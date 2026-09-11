import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import app.repository_indexing as indexing_module
from app.code_unit_persistence import persist_code_units
from app.database import create_database_engine
from app.embedding_provider import EmbeddingProvider
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.indexing_job import IndexingJob
from app.models.repository import Repository
from app.repository_clone import ClonedRepository, RepositoryCloneError
from app.repository_files import RepositoryFileCandidate
from app.repository_indexing import RepositoryIndexingError, index_repository

DATABASE_URL = os.getenv("DATABASE_URL")
GITHUB_SOURCE = "https://github.com/example/project.git"


@pytest.fixture
def database_session() -> Iterator[Session]:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")

    engine = create_database_engine(DATABASE_URL)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:
                yield session
        finally:
            transaction.rollback()
    engine.dispose()


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        dimension: int = EMBEDDING_DIMENSION,
        failure: Exception | None = None,
        omit_last_vector: bool = False,
    ) -> None:
        self._dimension = dimension
        self._failure = failure
        self._omit_last_vector = omit_last_vector
        self.batches: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        batch = tuple(texts)
        self.batches.append(batch)
        if self._failure is not None:
            raise self._failure

        vectors = [_vector_for_text(text) for text in batch]
        if self._omit_last_vector:
            return vectors[:-1]
        return vectors


def _use_provider(provider: EmbeddingProvider) -> EmbeddingProvider:
    return provider


def _vector_for_text(text: str) -> list[float]:
    marker = float(sum(ord(character) for character in text))
    return [marker, float(len(text)), *([0.0] * (EMBEDDING_DIMENSION - 2))]


def _repository(
    session: Session,
    *,
    source: str = GITHUB_SOURCE,
) -> Repository:
    stored_source = (
        f"https://github.com/example/{uuid4()}.git"
        if source == GITHUB_SOURCE
        else source
    )
    repository = Repository(
        name="Indexing test repository",
        source=stored_source,
    )
    session.add(repository)
    session.flush()
    return repository


def _fixture_files(sentinel: Path) -> dict[str, bytes]:
    return {
        "src/example.py": (
            "from pathlib import Path\n"
            f"Path({str(sentinel)!r}).write_text('executed')\n\n"
            "def python_symbol():\n"
            "    return 'python'\n"
        ).encode(),
        "src/example.js": b"export function javascriptSymbol() { return 'js'; }\n",
        "src/example.ts": b"export class TypeScriptSymbol {}\n",
        "README.md": b"# Example repository\n\nDocumentation body.\n",
        "config/example.yaml": b"enabled: true\n",
        "notes.unsupported": b"safe metadata without a parser\n",
        "node_modules/excluded.js": b"function excluded() {}\n",
        "binary.dat": b"binary\x00content",
        ".env": b"SECRET=not-indexed\n",
    }


def _install_offline_clone(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes],
    *,
    during_clone: Callable[[], None] | None = None,
) -> list[Path]:
    destinations: list[Path] = []

    def fake_clone(source: object, workspace_root: Path) -> ClonedRepository:
        if during_clone is not None:
            during_clone()
        workspace_root.mkdir(parents=True, exist_ok=True)
        destination = Path(
            tempfile.mkdtemp(prefix="repository-", dir=workspace_root.resolve())
        )
        for relative_path, content in files.items():
            path = destination / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        destinations.append(destination)
        return ClonedRepository(source=GITHUB_SOURCE, path=destination)

    monkeypatch.setattr(indexing_module, "clone_repository", fake_clone)
    return destinations


def _index(
    session: Session,
    repository: Repository,
    provider: EmbeddingProvider,
    workspace_root: Path,
) -> IndexingJob:
    return index_repository(
        session,
        repository_id=repository.id,
        embedding_provider=provider,
        workspace_root=workspace_root,
    )


def test_missing_repository_and_wrong_dimension_fail_before_job_or_clone(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone_calls = 0

    def unexpected_clone(source: object, workspace_root: Path) -> None:
        nonlocal clone_calls
        clone_calls += 1

    monkeypatch.setattr(indexing_module, "clone_repository", unexpected_clone)

    with pytest.raises(ValueError, match="Repository does not exist"):
        index_repository(
            database_session,
            repository_id=uuid4(),
            embedding_provider=FakeEmbeddingProvider(),
            workspace_root=tmp_path,
        )

    repository = _repository(database_session)
    with pytest.raises(ValueError, match="dimension"):
        _index(
            database_session,
            repository,
            FakeEmbeddingProvider(dimension=3),
            tmp_path,
        )

    invalid_source_repository = _repository(
        database_session,
        source=" https://github.com/example/project.git",
    )
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        _index(
            database_session,
            invalid_source_repository,
            FakeEmbeddingProvider(),
            tmp_path,
        )

    assert clone_calls == 0
    assert database_session.scalar(select(func.count()).select_from(IndexingJob)) == 0


@pytest.mark.parametrize("status", ["pending", "running", "completed"])
def test_blocking_job_status_rejects_new_index_before_clone(
    status: str,
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(database_session)
    database_session.add(IndexingJob(repository_id=repository.id, status=status))
    database_session.flush()

    def unexpected_clone(source: object, workspace_root: Path) -> None:
        pytest.fail("Clone must not run when an existing job blocks indexing")

    monkeypatch.setattr(indexing_module, "clone_repository", unexpected_clone)

    expected = (
        "active indexing job" if status != "completed" else "already been indexed"
    )
    with pytest.raises(ValueError, match=expected):
        _index(database_session, repository, FakeEmbeddingProvider(), tmp_path)

    assert (
        database_session.scalar(
            select(func.count())
            .select_from(IndexingJob)
            .where(IndexingJob.repository_id == repository.id)
        )
        == 1
    )


def test_existing_code_unit_blocks_full_reindex(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(database_session)
    file = File(repository_id=repository.id, path="src/existing.py")
    database_session.add(file)
    database_session.flush()
    persist_code_units(
        database_session,
        file.id,
        [
            indexing_module.PythonCodeParser().parse(
                content="def existing(): pass\n",
                relative_path=file.path,
            )[0]
        ],
    )

    def unexpected_clone(source: object, workspace_root: Path) -> None:
        pytest.fail("Clone must not run when CodeUnits already exist")

    monkeypatch.setattr(indexing_module, "clone_repository", unexpected_clone)

    with pytest.raises(ValueError, match="persisted CodeUnits"):
        _index(database_session, repository, FakeEmbeddingProvider(), tmp_path)


def test_small_repository_indexes_end_to_end_in_bounded_ordered_batches(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(database_session)
    existing_readme = File(repository_id=repository.id, path="README.md")
    database_session.add(existing_readme)
    database_session.flush()
    sentinel = tmp_path / "repository-code-executed"
    destinations = _install_offline_clone(
        monkeypatch,
        _fixture_files(sentinel),
        during_clone=lambda: _assert_running(database_session, repository.id),
    )
    monkeypatch.setattr(indexing_module, "EMBEDDING_BATCH_SIZE", 4)
    provider = FakeEmbeddingProvider()
    assert _use_provider(provider) is provider
    workspace = tmp_path / "workspace"
    sibling = workspace / "keep.txt"
    workspace.mkdir()
    sibling.write_text("keep", encoding="utf-8")

    job = _index(database_session, repository, provider, workspace)

    stored_files = list(
        database_session.scalars(
            select(File).where(File.repository_id == repository.id).order_by(File.path)
        )
    )
    units = list(
        database_session.scalars(
            select(CodeUnit)
            .join(File, CodeUnit.file_id == File.id)
            .where(File.repository_id == repository.id)
            .order_by(File.path, CodeUnit.start_line, CodeUnit.end_line.desc())
        )
    )

    assert job.status == "completed"
    assert {file.path for file in stored_files} == {
        "README.md",
        "config/example.yaml",
        "notes.unsupported",
        "src/example.js",
        "src/example.py",
        "src/example.ts",
    }
    assert existing_readme in stored_files
    assert {unit.language for unit in units} == {
        "config",
        "documentation",
        "javascript",
        "python",
        "typescript",
    }
    assert all(unit.embedding is not None for unit in units)
    assert all(len(unit.embedding or []) == EMBEDDING_DIMENSION for unit in units)
    for unit in units:
        assert unit.embedding is not None
        assert unit.embedding[0] == pytest.approx(_vector_for_text(unit.content)[0])
        assert unit.embedding[1] == pytest.approx(float(len(unit.content)))

    scanner_path_order = {
        path: index
        for index, path in enumerate(
            [
                "README.md",
                "config/example.yaml",
                "src/example.js",
                "src/example.py",
                "src/example.ts",
            ]
        )
    }
    expected_units = sorted(
        units,
        key=lambda unit: (
            scanner_path_order[unit.file.path],
            unit.start_line,
            unit.end_line,
        ),
    )
    flattened_inputs = [text for batch in provider.batches for text in batch]
    assert flattened_inputs == [unit.content for unit in expected_units]
    assert all(len(batch) <= 4 for batch in provider.batches)
    assert 0 < len(provider.batches[-1]) < 4
    assert not sentinel.exists()
    assert destinations and not destinations[0].exists()
    assert sibling.read_text(encoding="utf-8") == "keep"

    file_count = len(stored_files)
    unit_count = len(units)
    with pytest.raises(ValueError, match="already been indexed"):
        _index(database_session, repository, provider, workspace)
    assert (
        database_session.scalar(
            select(func.count())
            .select_from(File)
            .where(File.repository_id == repository.id)
        )
        == file_count
    )
    assert (
        database_session.scalar(
            select(func.count())
            .select_from(CodeUnit)
            .join(File, CodeUnit.file_id == File.id)
            .where(File.repository_id == repository.id)
        )
        == unit_count
    )


def _assert_running(session: Session, repository_id: object) -> None:
    assert (
        session.scalar(
            select(IndexingJob.status).where(IndexingJob.repository_id == repository_id)
        )
        == "running"
    )


@pytest.mark.parametrize(
    "provider",
    [
        FakeEmbeddingProvider(failure=RuntimeError("embedding failed")),
        FakeEmbeddingProvider(omit_last_vector=True),
    ],
)
def test_embedding_failure_rolls_back_derived_data_and_marks_job_failed(
    provider: FakeEmbeddingProvider,
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(database_session)
    destinations = _install_offline_clone(
        monkeypatch,
        {"src/example.py": b"def indexed_before_failure(): pass\n"},
    )

    with pytest.raises((RuntimeError, RepositoryIndexingError)):
        _index(database_session, repository, provider, tmp_path / "workspace")

    job = database_session.scalar(
        select(IndexingJob).where(IndexingJob.repository_id == repository.id)
    )
    assert job is not None
    assert job.status == "failed"
    assert (
        database_session.scalar(
            select(func.count())
            .select_from(File)
            .where(File.repository_id == repository.id)
        )
        == 0
    )
    assert destinations and not destinations[0].exists()


@pytest.mark.parametrize(
    ("relative_path", "content", "message"),
    [
        ("src/invalid.py", b"\xff", "not valid UTF-8"),
        ("src/invalid.py", b"def broken(\n", "syntax errors"),
    ],
)
def test_decode_and_parser_failures_are_fatal_and_clean_the_clone(
    relative_path: str,
    content: bytes,
    message: str,
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(database_session)
    destinations = _install_offline_clone(
        monkeypatch,
        {relative_path: content},
    )

    with pytest.raises((RepositoryIndexingError, RuntimeError), match=message):
        _index(
            database_session,
            repository,
            FakeEmbeddingProvider(),
            tmp_path / "workspace",
        )

    job = database_session.scalar(
        select(IndexingJob).where(IndexingJob.repository_id == repository.id)
    )
    assert job is not None and job.status == "failed"
    assert destinations and not destinations[0].exists()


def test_file_changed_after_discovery_fails_before_parsing(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(database_session)
    destinations = _install_offline_clone(
        monkeypatch,
        {"src/changing.py": b"def before_change(): pass\n"},
    )
    real_discover = indexing_module.discover_repository_files

    def discover_then_change(
        repository_root: Path,
    ) -> list[RepositoryFileCandidate]:
        candidates = real_discover(repository_root)
        (repository_root / "src/changing.py").write_bytes(
            b"def after_change_with_a_different_size(): pass\n"
        )
        return candidates

    monkeypatch.setattr(
        indexing_module,
        "discover_repository_files",
        discover_then_change,
    )

    with pytest.raises(RepositoryIndexingError, match="changed during indexing"):
        _index(
            database_session,
            repository,
            FakeEmbeddingProvider(),
            tmp_path / "workspace",
        )

    job = database_session.scalar(
        select(IndexingJob).where(IndexingJob.repository_id == repository.id)
    )
    assert job is not None and job.status == "failed"
    assert destinations and not destinations[0].exists()


def test_failed_job_allows_retry_when_no_partial_units_remain(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(database_session)
    database_session.add(IndexingJob(repository_id=repository.id, status="failed"))
    database_session.flush()
    _install_offline_clone(
        monkeypatch,
        {"README.md": b"# Retry\n\nSuccessful retry.\n"},
    )

    job = _index(
        database_session,
        repository,
        FakeEmbeddingProvider(),
        tmp_path / "workspace",
    )

    assert job.status == "completed"
    assert sorted(
        database_session.scalars(
            select(IndexingJob.status).where(IndexingJob.repository_id == repository.id)
        )
    ) == ["completed", "failed"]


def test_supported_but_unclonable_source_marks_started_job_failed(
    database_session: Session,
    tmp_path: Path,
) -> None:
    repository = _repository(database_session, source="../local-repository")

    with pytest.raises(RepositoryCloneError, match="validated HTTPS"):
        _index(
            database_session,
            repository,
            FakeEmbeddingProvider(),
            tmp_path / "workspace",
        )

    job = database_session.scalar(
        select(IndexingJob).where(IndexingJob.repository_id == repository.id)
    )
    assert job is not None and job.status == "failed"
    assert not (tmp_path / "workspace").exists()


def test_cleanup_failure_rolls_back_derived_data_and_preserves_workspace(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(database_session)
    destinations = _install_offline_clone(
        monkeypatch,
        {"README.md": b"# Cleanup failure\n"},
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sibling = workspace / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")

    def failed_remove(clone_path: Path, workspace_root: Path) -> None:
        raise RepositoryIndexingError("Cloned repository cleanup failed")

    monkeypatch.setattr(indexing_module, "_remove_owned_clone", failed_remove)

    with pytest.raises(RepositoryIndexingError, match="cleanup failed"):
        _index(database_session, repository, FakeEmbeddingProvider(), workspace)

    job = database_session.scalar(
        select(IndexingJob).where(IndexingJob.repository_id == repository.id)
    )
    assert job is not None and job.status == "failed"
    assert (
        database_session.scalar(
            select(func.count())
            .select_from(File)
            .where(File.repository_id == repository.id)
        )
        == 0
    )
    assert destinations[0].exists()
    assert workspace.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"


def test_indexing_and_cleanup_failure_preserves_primary_error_and_rolls_back(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(database_session)
    destinations = _install_offline_clone(
        monkeypatch,
        {"src/example.py": b"def persisted_before_failure(): pass\n"},
    )
    indexing_error = RuntimeError("embedding failed")
    provider = FakeEmbeddingProvider(failure=indexing_error)

    def failed_remove(clone_path: Path, workspace_root: Path) -> None:
        raise RepositoryIndexingError(f"cleanup failed for {clone_path}")

    monkeypatch.setattr(indexing_module, "_remove_owned_clone", failed_remove)

    with pytest.raises(RuntimeError, match="embedding failed") as captured_error:
        _index(
            database_session,
            repository,
            provider,
            tmp_path / "workspace",
        )

    assert captured_error.value is indexing_error
    assert provider.batches
    assert destinations
    clone_path = destinations[0]
    notes = getattr(captured_error.value, "__notes__", [])
    assert any("cleanup also failed" in note.casefold() for note in notes)
    assert str(captured_error.value) == "embedding failed"
    assert str(clone_path) not in str(captured_error.value)
    assert all(str(clone_path) not in note for note in notes)

    job = database_session.scalar(
        select(IndexingJob).where(IndexingJob.repository_id == repository.id)
    )
    assert job is not None and job.status == "failed"
    assert (
        database_session.scalar(
            select(func.count())
            .select_from(File)
            .where(File.repository_id == repository.id)
        )
        == 0
    )
    assert (
        database_session.scalar(
            select(func.count())
            .select_from(CodeUnit)
            .join(File, CodeUnit.file_id == File.id)
            .where(File.repository_id == repository.id)
        )
        == 0
    )


def test_pipeline_does_not_commit_and_caller_can_roll_back_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(DATABASE_URL)
    repository_id = uuid4()

    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection) as session:
            repository = Repository(
                id=repository_id,
                name="Rollback repository",
                source=f"https://github.com/example/{repository_id}.git",
            )
            session.add(repository)
            session.flush()
            _install_offline_clone(
                monkeypatch,
                {"README.md": b"# Rollback\n"},
            )

            def unexpected_transaction_method() -> None:
                pytest.fail("Indexing pipeline must not own the outer transaction")

            monkeypatch.setattr(session, "commit", unexpected_transaction_method)
            monkeypatch.setattr(session, "rollback", unexpected_transaction_method)
            _index(session, repository, FakeEmbeddingProvider(), tmp_path / "workspace")

        transaction.rollback()

    with Session(engine) as verification_session:
        assert verification_session.get(Repository, repository_id) is None
    engine.dispose()


def test_caller_can_commit_failed_status_without_partial_derived_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(DATABASE_URL)
    repository_id = uuid4()

    try:
        with Session(engine) as session:
            repository = Repository(
                id=repository_id,
                name="Committed failure repository",
                source=f"https://github.com/example/{repository_id}.git",
            )
            session.add(repository)
            session.commit()
            _install_offline_clone(
                monkeypatch,
                {"src/example.py": b"def persisted_then_rolled_back(): pass\n"},
            )

            with pytest.raises(RuntimeError, match="embedding failed"):
                _index(
                    session,
                    repository,
                    FakeEmbeddingProvider(failure=RuntimeError("embedding failed")),
                    tmp_path / "workspace",
                )
            session.commit()

        with Session(engine) as verification_session:
            job = verification_session.scalar(
                select(IndexingJob).where(IndexingJob.repository_id == repository_id)
            )
            assert job is not None and job.status == "failed"
            assert (
                verification_session.scalar(
                    select(func.count())
                    .select_from(File)
                    .where(File.repository_id == repository_id)
                )
                == 0
            )
    finally:
        with Session(engine) as cleanup_session:
            repository_to_delete = cleanup_session.get(Repository, repository_id)
            if repository_to_delete is not None:
                cleanup_session.delete(repository_to_delete)
                cleanup_session.commit()
        engine.dispose()
