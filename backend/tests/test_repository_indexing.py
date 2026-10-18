import hashlib
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import app.repository_indexing as indexing_module
from app.code_parser import ParsedCodeUnit
from app.code_relationships import FileCallAnalysis
from app.code_unit_persistence import persist_code_units
from app.database import create_database_engine
from app.embedding_provider import EmbeddingProvider
from app.hybrid_search import HybridSearchResult, search_code_units_hybrid
from app.import_relationships import ImportReference
from app.lexical_search import LexicalSearchResult, search_code_units_lexically
from app.metrics import Metrics, use_metrics
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.indexing_job import IndexingJob
from app.models.relationship import Relationship
from app.models.repository import Repository
from app.repository_clone import ClonedRepository, RepositoryCloneError
from app.repository_files import RepositoryFileCandidate
from app.repository_indexing import RepositoryIndexingError, index_repository
from app.reranked_search import RerankedSearchResult, search_code_units_reranked
from app.semantic_search import SemanticSearchResult, search_code_units_semantically

DATABASE_URL = os.getenv("DATABASE_URL")
GITHUB_SOURCE = "https://github.com/example/project.git"


class EqualScoreReranker:
    model_name = "test-equal-score"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, tuple(documents)))
        return [1.0] * len(documents)


def _edge_case_searches(
    session: Session,
    repository: Repository,
    query: str,
    reranker: EqualScoreReranker,
) -> dict[
    str,
    Sequence[
        SemanticSearchResult
        | LexicalSearchResult
        | HybridSearchResult
        | RerankedSearchResult
    ],
]:
    vector = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
    return {
        "semantic": search_code_units_semantically(
            session, repository_id=repository.id, query_vector=vector, limit=100
        ),
        "lexical": search_code_units_lexically(
            session, repository_id=repository.id, query=query, limit=100
        ),
        "hybrid": search_code_units_hybrid(
            session,
            repository_id=repository.id,
            query=query,
            query_vector=vector,
            limit=100,
        ),
        "reranked": search_code_units_reranked(
            session,
            repository_id=repository.id,
            query=query,
            query_vector=vector,
            reranker=reranker,
            limit=100,
        ),
    }


def _edge_case_units(session: Session, repository: Repository) -> list[CodeUnit]:
    return list(
        session.scalars(
            select(CodeUnit)
            .join(File, CodeUnit.file_id == File.id)
            .where(File.repository_id == repository.id)
            .order_by(File.path, CodeUnit.start_line, CodeUnit.id)
        )
    )


def test_duplicate_symbols_survive_indexing_and_all_retrieval_stages(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"def repeated():\n    return 1\n\ndef repeated():\n    return 2\n"
    _install_offline_clone(
        monkeypatch,
        {
            "a.py": source,
            "b.py": b"def repeated():\n    return 3\n",
        },
    )
    repository = _repository(database_session)
    other = _repository(database_session)
    provider = FakeEmbeddingProvider()
    _index(database_session, other, FakeEmbeddingProvider(), tmp_path / "other")
    _index(database_session, repository, provider, tmp_path / "target")
    units = _edge_case_units(database_session, repository)
    expected = [
        ("a.py", 1, 2, "repeated", "def repeated():\n    return 1"),
        ("a.py", 4, 5, "repeated", "def repeated():\n    return 2"),
        ("b.py", 1, 2, "repeated", "def repeated():\n    return 3"),
    ]
    assert [
        (u.file.path, u.start_line, u.end_line, u.symbol_name, u.content) for u in units
    ] == expected
    expected_by_id = {
        u.id: metadata for u, metadata in zip(units, expected, strict=True)
    }
    assert len(expected_by_id) == 3
    other_ids = {u.id for u in _edge_case_units(database_session, other)}
    assert len(other_ids) == 3 and other_ids.isdisjoint(expected_by_id)
    reranker = EqualScoreReranker()
    results = _edge_case_searches(database_session, repository, "repeated", reranker)
    again = _edge_case_searches(database_session, repository, "repeated", reranker)
    for mode, rows in results.items():
        assert len(rows) == 3 and {r.code_unit_id for r in rows} == set(expected_by_id)
        assert [r.code_unit_id for r in rows] == [r.code_unit_id for r in again[mode]]
        for r in rows:
            assert (r.path, r.start_line, r.end_line, r.symbol_name, r.content) == (
                expected_by_id[r.code_unit_id]
            )
            assert r.file_id == next(u.file_id for u in units if u.id == r.code_unit_id)
    hybrid = results["hybrid"]
    assert all(
        isinstance(r, HybridSearchResult)
        and r.semantic_rank is not None
        and r.lexical_rank is not None
        for r in hybrid
    )
    assert [r.code_unit_id for r in results["reranked"]] == [
        r.code_unit_id for r in hybrid
    ]
    expected_documents = tuple(
        f"Path: {r.path}\nSymbol: repeated\n\nContent:\n{r.content}" for r in hybrid
    )
    assert reranker.calls == [("repeated", expected_documents)] * 2
    assert all(
        isinstance(r, RerankedSearchResult)
        and r.rerank_score == 1.0
        and r.original_hybrid_rank == rank
        for rank, r in enumerate(results["reranked"], 1)
    )
    before_batches = list(provider.batches)
    generation = repository.index_generation
    _index(database_session, repository, provider, tmp_path / "target")
    assert {u.id for u in _edge_case_units(database_session, repository)} == set(
        expected_by_id
    )
    assert (
        provider.batches == before_batches and repository.index_generation == generation
    )


@pytest.mark.parametrize("content", [b"", b" \t\r\n\n"])
def test_empty_supported_files_index_as_metadata_without_embeddings(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
) -> None:
    files = {
        path: content
        for path in ["empty.py", "empty.js", "empty.ts", "notes.md", "settings.yaml"]
    }
    destinations = _install_offline_clone(monkeypatch, files)
    repository = _repository(database_session)
    provider = FakeEmbeddingProvider()
    assert (
        _index(database_session, repository, provider, tmp_path / "workspace").status
        == "completed"
    )
    stored = list(
        database_session.scalars(
            select(File).where(File.repository_id == repository.id)
        )
    )
    assert {f.path: f.content_hash for f in stored} == {
        path: hashlib.sha256(raw).hexdigest() for path, raw in files.items()
    }
    assert (
        _edge_case_units(database_session, repository) == [] and provider.batches == []
    )
    reranker = EqualScoreReranker()
    assert all(
        not rows
        for rows in _edge_case_searches(
            database_session, repository, "empty", reranker
        ).values()
    )
    assert reranker.calls == []
    assert destinations and all(not p.exists() for p in destinations)


def test_sixty_five_definitions_keep_full_spans_and_default_embedding_batches(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definitions = [
        f'def entry_{i}():\n    return "{"界" * 2_100 if i == 0 else i}"'
        for i in range(65)
    ]
    _install_offline_clone(
        monkeypatch, {"many.py": ("\n\n".join(definitions) + "\n").encode()}
    )
    repository = _repository(database_session)
    provider = FakeEmbeddingProvider()
    assert (
        _index(database_session, repository, provider, tmp_path / "workspace").status
        == "completed"
    )
    units = _edge_case_units(database_session, repository)
    assert len(units) == 65 and len({u.id for u in units}) == 65
    assert [u.content for u in units] == definitions
    assert [(u.symbol_name, u.start_line, u.end_line) for u in units] == [
        (f"entry_{i}", 3 * i + 1, 3 * i + 2) for i in range(65)
    ]
    assert [len(batch) for batch in provider.batches] == [64, 1]
    assert [text for batch in provider.batches for text in batch] == definitions
    assert all(u.embedding is not None for u in units)
    for rows in _edge_case_searches(
        database_session, repository, "entry_", EqualScoreReranker()
    ).values():
        assert len(rows) == 65 and {r.code_unit_id for r in rows} == {
            u.id for u in units
        }
        assert {r.code_unit_id: r.content for r in rows} == {
            u.id: u.content for u in units
        }


@pytest.mark.parametrize("path", ["notes.md", "settings.yaml"])
def test_long_text_chunks_survive_indexing_losslessly(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    # One physical Unicode line: both chunks intentionally have the same range.
    content = "界" * 2_001
    _install_offline_clone(monkeypatch, {path: content.encode()})
    repository = _repository(database_session)
    provider = FakeEmbeddingProvider()
    _index(database_session, repository, provider, tmp_path / "workspace")
    units = _edge_case_units(database_session, repository)
    assert len(units) == 2 and len({u.id for u in units}) == 2
    assert sorted(u.content for u in units) == ["界", "界" * 2_000]
    assert all((u.start_line, u.end_line) == (1, 1) for u in units)
    assert provider.batches == [("界" * 2_000, "界")]
    assert "".join(provider.batches[0]) == content
    assert all(u.embedding is not None for u in units)
    for rows in _edge_case_searches(
        database_session, repository, "界", EqualScoreReranker()
    ).values():
        assert len(rows) == 2 and {r.code_unit_id for r in rows} == {
            u.id for u in units
        }
        assert sorted(r.content for r in rows) == ["界", "界" * 2_000]


def test_discovery_byte_limit_controls_indexing_without_truncation(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 1_048_576
    definition = b"def boundary():\n    return 1\n"
    admitted = (
        b"#"
        + ("é" * ((limit - len(definition) - 3) // 2)).encode()
        + b"\n"
        + definition
    )
    excluded = admitted + b" " * (limit + 1 - len(admitted))
    assert len(admitted) < limit < len(excluded)
    assert len(admitted.decode()) < len(admitted)
    _install_offline_clone(monkeypatch, {"below.py": admitted, "above.py": excluded})
    parsed: list[str] = []
    original = indexing_module.PythonCodeParser.parse

    def record_parse(
        self: indexing_module.PythonCodeParser, *, content: str, relative_path: str
    ) -> list[ParsedCodeUnit]:
        parsed.append(relative_path)
        return original(self, content=content, relative_path=relative_path)

    monkeypatch.setattr(indexing_module.PythonCodeParser, "parse", record_parse)
    repository = _repository(database_session)
    provider = FakeEmbeddingProvider()
    assert (
        _index(database_session, repository, provider, tmp_path / "workspace").status
        == "completed"
    )
    stored = list(
        database_session.scalars(
            select(File).where(File.repository_id == repository.id)
        )
    )
    assert [(f.path, f.content_hash) for f in stored] == [
        ("below.py", hashlib.sha256(admitted).hexdigest())
    ]
    assert parsed == ["below.py"]
    units = _edge_case_units(database_session, repository)
    assert len(units) == 1
    assert (units[0].content, units[0].start_line, units[0].end_line) == (
        definition.decode().rstrip("\n"),
        2,
        3,
    )
    assert provider.batches == [(definition.decode().rstrip("\n"),)]


@pytest.mark.parametrize("mixed", [False, True])
def test_unsupported_languages_remain_metadata_only(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mixed: bool,
) -> None:
    unsupported = {
        path: b"def searchable():\n    return 1\n"
        for path in ["module.rs", "module.go", "module.tsx", "UNKNOWN_SOURCE"]
    }
    files = dict(unsupported)
    if mixed:
        files["supported.py"] = b"def searchable():\n    return 2\n"
    _install_offline_clone(monkeypatch, files)
    repository = _repository(database_session)
    provider = FakeEmbeddingProvider()
    assert (
        _index(database_session, repository, provider, tmp_path / "workspace").status
        == "completed"
    )
    stored = list(
        database_session.scalars(
            select(File).where(File.repository_id == repository.id)
        )
    )
    assert {f.path: f.content_hash for f in stored} == {
        p: hashlib.sha256(raw).hexdigest() for p, raw in files.items()
    }
    units = _edge_case_units(database_session, repository)
    assert len(units) == int(mixed)
    assert provider.batches == ([("def searchable():\n    return 2",)] if mixed else [])
    for rows in _edge_case_searches(
        database_session, repository, "searchable", EqualScoreReranker()
    ).values():
        assert len(rows) == int(mixed)
        assert {r.code_unit_id for r in rows} == {u.id for u in units}
        assert all(r.path == "supported.py" for r in rows)


@pytest.fixture
def metrics() -> Iterator[Metrics]:
    value = Metrics()
    with use_metrics(value):
        yield value


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


@pytest.mark.parametrize("status", ["pending", "completed", "failed"])
def test_reserved_execution_rejects_unclaimed_job(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    repository = _repository(database_session)
    job = IndexingJob(repository_id=repository.id, status=status)
    database_session.add(job)
    database_session.flush()
    monkeypatch.setattr(
        indexing_module,
        "clone_repository",
        lambda *args: pytest.fail("Unclaimed job must not clone"),
    )
    with pytest.raises(ValueError, match="not running"):
        indexing_module.execute_reserved_indexing_job(
            database_session,
            repository_id=repository.id,
            job_id=job.id,
            embedding_provider=FakeEmbeddingProvider(),
            workspace_root=tmp_path,
        )
    assert job.status == status


@pytest.mark.parametrize("fail_relationships", [False, True])
def test_import_relationships_share_indexing_transaction(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_relationships: bool,
) -> None:
    repository = _repository(database_session)
    _install_offline_clone(
        monkeypatch,
        {
            "a.py": b"import z\nimport z\n",
            "z.py": b"",
        },
    )
    provider = FakeEmbeddingProvider()
    original = indexing_module.persist_import_relationships
    if fail_relationships:
        # Exercise a failure after relationships have actually been flushed.
        def persist_then_fail(
            session: Session,
            *,
            repository_id: UUID,
            files_by_path: Mapping[str, File],
            references: Sequence[ImportReference],
        ) -> None:
            original(
                session,
                repository_id=repository_id,
                files_by_path=files_by_path,
                references=references,
            )
            raise RuntimeError("relationship failure")

        monkeypatch.setattr(
            indexing_module,
            "persist_import_relationships",
            persist_then_fail,
        )
        with pytest.raises(RuntimeError, match="relationship failure"):
            _index(database_session, repository, provider, tmp_path)
    else:
        job = _index(database_session, repository, provider, tmp_path)
        assert job.status == "completed"
    edges = database_session.scalars(
        select(Relationship).where(Relationship.repository_id == repository.id)
    ).all()
    assert len(edges) == (0 if fail_relationships else 1)
    files = database_session.scalars(
        select(File).where(File.repository_id == repository.id)
    ).all()
    if fail_relationships:
        assert files == []
    else:
        paths = {file.id: file.path for file in files}
        assert (paths[edges[0].source_file_id], paths[edges[0].target_file_id]) == (
            "a.py",
            "z.py",
        )
        units = database_session.scalars(
            select(CodeUnit).join(File).where(File.repository_id == repository.id)
        ).all()
        assert len(units) == 2
        assert all(unit.embedding is not None for unit in units)
    assert provider.batches == [("import z", "import z")]


def _vector_for_text(text: str) -> list[float]:
    marker = float(sum(ord(character) for character in text))
    return [marker, float(len(text)), *([0.0] * (EMBEDDING_DIMENSION - 2))]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("fail_calls", [False, True])
def test_calls_share_indexing_savepoint_and_preserve_embeddings(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reverse: bool,
    fail_calls: bool,
) -> None:
    repository = _repository(database_session)
    _install_offline_clone(
        monkeypatch,
        {
            "a.py": b"from z import helper\ndef caller():\n helper()\n helper()\n",
            "z.py": b"def helper(): pass\n",
        },
    )
    discover = indexing_module.discover_repository_files
    if reverse:
        monkeypatch.setattr(
            indexing_module,
            "discover_repository_files",
            lambda path, **kwargs: list(reversed(discover(path, **kwargs))),
        )
    original = indexing_module.persist_call_relationships
    observed: list[int] = []

    def persist(
        session: Session,
        *,
        repository_id: UUID,
        files_by_path: Mapping[str, File],
        analyses: Sequence[FileCallAnalysis],
    ) -> None:
        original(
            session,
            repository_id=repository_id,
            files_by_path=files_by_path,
            analyses=analyses,
        )
        count = session.scalar(
            select(func.count())
            .select_from(CodeRelationship)
            .where(CodeRelationship.repository_id == repository_id)
        )
        assert count == 1
        observed.append(count)
        if fail_calls:
            raise RuntimeError("call persistence failure")

    monkeypatch.setattr(indexing_module, "persist_call_relationships", persist)
    provider = FakeEmbeddingProvider()
    if fail_calls:
        with pytest.raises(RuntimeError, match="call persistence failure"):
            _index(database_session, repository, provider, tmp_path)
    else:
        _index(database_session, repository, provider, tmp_path)
    assert observed == [1]
    expected = [
        "from z import helper",
        "def caller():\n helper()\n helper()",
        "def helper(): pass",
    ]
    if reverse:
        expected = [expected[-1], *expected[:-1]]
    assert provider.batches == [tuple(expected)]
    units = database_session.scalars(
        select(CodeUnit).join(File).where(File.repository_id == repository.id)
    ).all()
    assert len(units) == (0 if fail_calls else 3)
    for model in (File, Relationship, CodeRelationship):
        count = database_session.scalar(
            select(func.count())
            .select_from(model)
            .where(model.repository_id == repository.id)
        )
        assert count == (0 if fail_calls else 2 if model is File else 1)
    if not fail_calls:
        edge = database_session.scalar(
            select(CodeRelationship).where(
                CodeRelationship.repository_id == repository.id
            )
        )
        assert edge is not None
        by_id = {unit.id: unit for unit in units}
        assert by_id[edge.source_code_unit_id].symbol_name == "caller"
        assert by_id[edge.target_code_unit_id].symbol_name == "helper"


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
    job_count = select(func.count()).select_from(IndexingJob)
    jobs_before = database_session.scalar(job_count)
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
    assert database_session.scalar(job_count) == jobs_before

    repository = _repository(database_session)
    with pytest.raises(ValueError, match="dimension"):
        _index(
            database_session,
            repository,
            FakeEmbeddingProvider(dimension=3),
            tmp_path,
        )
    assert database_session.scalar(job_count) == jobs_before

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
    assert database_session.scalar(job_count) == jobs_before

    assert clone_calls == 0


@pytest.mark.parametrize("status", ["pending", "running"])
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

    with pytest.raises(ValueError, match="active indexing job"):
        _index(database_session, repository, FakeEmbeddingProvider(), tmp_path)

    assert (
        database_session.scalar(
            select(func.count())
            .select_from(IndexingJob)
            .where(IndexingJob.repository_id == repository.id)
        )
        == 1
    )


def test_existing_unverified_code_unit_is_replaced_on_reindex(
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

    old_ids = set(
        database_session.scalars(select(CodeUnit.id).where(CodeUnit.file_id == file.id))
    )
    _install_offline_clone(monkeypatch, {file.path: b"def replacement(): pass\n"})
    job = _index(database_session, repository, FakeEmbeddingProvider(), tmp_path)
    units = database_session.scalars(
        select(CodeUnit).where(CodeUnit.file_id == file.id)
    ).all()
    assert job.status == "completed" and file.content_hash is not None
    assert [unit.symbol_name for unit in units] == ["replacement"]
    assert old_ids.isdisjoint(unit.id for unit in units)


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
    batches_before = list(provider.batches)
    second_job = _index(database_session, repository, provider, workspace)
    assert second_job.id != job.id and second_job.status == "completed"
    assert provider.batches == batches_before
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
            select(IndexingJob.status)
            .where(IndexingJob.repository_id == repository_id)
            .where(IndexingJob.status == "running")
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
        *,
        strict_errors: bool = False,
    ) -> list[RepositoryFileCandidate]:
        candidates = real_discover(repository_root, strict_errors=strict_errors)
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
    metrics: Metrics,
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
    # Service success is measured even though the caller discarded all DB work.
    assert (
        metrics.registry.get_sample_value(
            "repomind_indexing_executions_total",
            {"mode": "direct", "outcome": "success"},
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value("repomind_indexing_files_hashed_total") == 1
    )
    assert (
        metrics.registry.get_sample_value("repomind_indexing_code_units_parsed_total")
        == 1
    )


@pytest.mark.parametrize("collector_failure", [False, True])
def test_work_metrics_count_hashes_but_not_reused_units(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metrics: Metrics,
    collector_failure: bool,
) -> None:
    repository = _repository(database_session)
    _install_offline_clone(monkeypatch, {"a.py": b"def helper(): return 1\n"})
    if collector_failure:

        def broken(*args: object, **kwargs: object) -> None:
            raise RuntimeError("telemetry only")

        monkeypatch.setattr(metrics.files_hashed, "inc", broken)
        monkeypatch.setattr(metrics.units_parsed, "inc", broken)
        monkeypatch.setattr(metrics.indexing_duration, "labels", broken)
        monkeypatch.setattr(metrics.indexing_executions, "labels", broken)
    first = _index(
        database_session, repository, FakeEmbeddingProvider(), tmp_path / "first"
    )
    ids = list(
        database_session.scalars(
            select(File.id).where(File.repository_id == repository.id)
        )
    )
    second = _index(
        database_session, repository, FakeEmbeddingProvider(), tmp_path / "second"
    )
    assert first.status == second.status == "completed"
    assert (
        list(
            database_session.scalars(
                select(File.id).where(File.repository_id == repository.id)
            )
        )
        == ids
    )
    if not collector_failure:
        assert (
            metrics.registry.get_sample_value("repomind_indexing_files_hashed_total")
            == 2
        )
        assert (
            metrics.registry.get_sample_value(
                "repomind_indexing_code_units_parsed_total"
            )
            == 1
        )
        assert (
            metrics.registry.get_sample_value(
                "repomind_indexing_executions_total",
                {"mode": "direct", "outcome": "success"},
            )
            == 2
        )


def test_failed_direct_execution_keeps_work_metrics(
    database_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metrics: Metrics,
) -> None:
    repository = _repository(database_session)
    _install_offline_clone(monkeypatch, {"a.py": b"def helper(): return 1\n"})
    error = RuntimeError("embedding failed")
    with pytest.raises(RuntimeError) as caught:
        _index(
            database_session,
            repository,
            FakeEmbeddingProvider(failure=error),
            tmp_path / "workspace",
        )
    assert caught.value is error
    assert (
        list(
            database_session.scalars(
                select(File).where(File.repository_id == repository.id)
            )
        )
        == []
    )
    assert (
        metrics.registry.get_sample_value("repomind_indexing_files_hashed_total") == 1
    )
    assert (
        metrics.registry.get_sample_value("repomind_indexing_code_units_parsed_total")
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "repomind_indexing_executions_total", {"mode": "direct", "outcome": "error"}
        )
        == 1
    )


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
