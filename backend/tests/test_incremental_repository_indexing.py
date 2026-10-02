import hashlib
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app import repository_indexing as indexing
from app.database import create_database_engine
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.indexing_job import IndexingJob
from app.models.relationship import Relationship
from app.models.repository import Repository
from app.repository_clone import ClonedRepository
from app.repository_files import RepositoryFileCandidate


class FakeEmbeddingProvider:
    dimension = 384

    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(tuple(texts))
        return [[float(len(t)), 1.0] + [0.0] * 382 for t in texts]


@dataclass
class IndexingFixture:
    session: Session
    repository: Repository
    workspace: Path
    files: dict[str, bytes]
    provider: FakeEmbeddingProvider = field(default_factory=FakeEmbeddingProvider)
    clones: list[Path] = field(default_factory=list)

    def run(self) -> IndexingJob:
        return indexing.index_repository(
            self.session,
            repository_id=self.repository.id,
            embedding_provider=self.provider,
            workspace_root=self.workspace,
        )


@pytest.fixture
def fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[IndexingFixture]:
    url = os.getenv("DATABASE_URL")
    if url is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(url)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:
                repo = Repository(
                    name="incremental",
                    source=f"https://github.com/example/{uuid4()}.git",
                )
                session.add(repo)
                session.flush()
                state = IndexingFixture(
                    session,
                    repo,
                    tmp_path / "workspace",
                    {
                        "api.py": (
                            b"from service import helper\ndef caller():\n"
                            b"    return helper()\n"
                        ),
                        "service.py": b"def helper():\n    return 1\n",
                        "nested/README.md": b"# Notes\n",
                        "empty.py": b"",
                        "notes.unsupported": b"metadata\xff",
                    },
                )

                def clone(source: Any, workspace_root: Path) -> ClonedRepository:
                    workspace_root.mkdir(exist_ok=True, parents=True)
                    destination = Path(
                        tempfile.mkdtemp(prefix="repository-", dir=workspace_root)
                    )
                    for path, content in state.files.items():
                        target = destination / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                    state.clones.append(destination)
                    return ClonedRepository(source=repo.source, path=destination)

                monkeypatch.setattr(indexing, "clone_repository", clone)
                yield state
        finally:
            transaction.rollback()
    engine.dispose()


def _snapshot(
    session: Session, repository_id: UUID
) -> dict[str, list[tuple[Any, ...]]]:
    result = {}
    for model in (File, CodeUnit, Relationship, CodeRelationship):
        table = model.__table__
        predicate = (
            table.c.file_id.in_(
                select(File.id).where(File.repository_id == repository_id)
            )
            if model is CodeUnit
            else table.c.repository_id == repository_id
        )
        rows = session.execute(select(table).where(predicate).order_by(table.c.id))
        result[model.__tablename__] = [
            tuple(
                tuple(value.tolist()) if hasattr(value, "tolist") else value
                for value in row
            )
            for row in rows
        ]
    return result


def _units(fixture: IndexingFixture) -> dict[str, list[tuple[Any, ...]]]:
    rows = fixture.session.execute(
        select(File.path, CodeUnit.id, CodeUnit.content, CodeUnit.embedding)
        .join(CodeUnit, CodeUnit.file_id == File.id)
        .where(File.repository_id == fixture.repository.id)
        .order_by(CodeUnit.id)
    )
    result: dict[str, list[tuple[Any, ...]]] = {}
    for path, uid, content, embedding in rows:
        result.setdefault(path, []).append((uid, content, tuple(embedding)))
    return result


@pytest.mark.parametrize(
    "raw", [b"", b"hello\n", b"hello\r\n", "日本語".encode(), b"\xff"]
)
def test_raw_hash_identity(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "source.py"
    path.write_bytes(raw)
    candidate = RepositoryFileCandidate(path, "source.py", len(raw))
    data = indexing._read_candidate_bytes(candidate)
    assert data == raw
    assert hashlib.sha256(data).hexdigest() == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "current,old,refresh,removed",
    [
        ({}, {}, set(), set()),
        ({"a": "h"}, {}, {"a"}, set()),
        ({"a": "h"}, {"a": None}, {"a"}, set()),
        ({"a": "h"}, {"a": "h"}, set(), set()),
        ({"a": "new"}, {"a": "old"}, {"a"}, set()),
        ({}, {"a": "h"}, set(), {"a"}),
        ({"b": "h"}, {"a": "h"}, {"b"}, {"a"}),
    ],
)
def test_classification(
    current: dict[str, str],
    old: dict[str, str | None],
    refresh: set[str],
    removed: set[str],
) -> None:
    assert indexing._classify_snapshot(current, old) == (refresh, removed)


def test_identical_second_run_preserves_entire_snapshot_without_work(
    fixture: IndexingFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = fixture.run()
    before = _snapshot(fixture.session, fixture.repository.id)
    assert fixture.repository.index_generation == 1
    assert (
        before["code_units"]
        and before["relationships"]
        and before["code_relationships"]
    )
    files = fixture.session.scalars(
        select(File).where(File.repository_id == fixture.repository.id)
    ).all()
    assert {f.path: f.content_hash for f in files} == {
        p: hashlib.sha256(b).hexdigest() for p, b in fixture.files.items()
    }
    batches = list(fixture.provider.batches)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("A no-change run must not parse, embed, or rebuild graphs")

    for name in (
        "PythonCodeParser",
        "JavaScriptCodeParser",
        "TypeScriptCodeParser",
        "DocumentationTextParser",
        "ConfigurationTextParser",
        "analyze_calls",
        "extract_imports",
        "persist_code_units",
        "persist_import_relationships",
        "persist_call_relationships",
    ):
        monkeypatch.setattr(indexing, name, forbidden)
    monkeypatch.setattr(fixture.provider, "embed", forbidden)
    second = fixture.run()
    assert first.id != second.id and first.status == second.status == "completed"
    assert _snapshot(fixture.session, fixture.repository.id) == before
    assert fixture.provider.batches == batches
    assert fixture.repository.index_generation == 1
    assert len(fixture.clones) == 2 and all(
        not path.exists() for path in fixture.clones
    )


@pytest.mark.parametrize(
    "change,imports,calls",
    [
        ("modified", 1, 1),
        ("removed_function", 1, 0),
        ("new", 2, 2),
        ("deleted", 0, 0),
        ("rename", 0, 0),
        ("ambiguous", 0, 0),
    ],
)
def test_incremental_reconciliation_preserves_unchanged_units(
    fixture: IndexingFixture,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    imports: int,
    calls: int,
) -> None:
    fixture.run()
    old_units = _units(fixture)
    old_files = {
        f.path: f.id
        for f in fixture.session.scalars(
            select(File).where(File.repository_id == fixture.repository.id)
        )
    }
    fixture.provider.batches.clear()
    if change == "modified":
        fixture.files["service.py"] = b"def helper():\n    return 2\n"
        refreshed = {"service.py"}
    elif change == "removed_function":
        fixture.files["service.py"] = b"def other(): pass\n"
        refreshed = {"service.py"}
    elif change == "new":
        fixture.files["new.py"] = (
            b"from service import helper\ndef another(): return helper()\n"
        )
        refreshed = {"new.py"}
    elif change == "ambiguous":
        fixture.files["service/__init__.py"] = b"def helper(): pass\n"
        refreshed = {"service/__init__.py"}
    else:
        raw = fixture.files.pop("service.py")
        refreshed = set()
        if change == "rename":
            fixture.files["moved.py"] = raw
            refreshed = {"moved.py"}
    parsed: list[str] = []
    original = indexing.PythonCodeParser.parse

    def parse(self: Any, *, content: str, relative_path: str) -> Any:
        parsed.append(relative_path)
        return original(self, content=content, relative_path=relative_path)

    monkeypatch.setattr(indexing.PythonCodeParser, "parse", parse)
    assert fixture.run().status == "completed"
    assert set(parsed) == refreshed and len(parsed) == len(refreshed)
    new_units = _units(fixture)
    assert new_units["api.py"] == old_units["api.py"]
    assert new_units["nested/README.md"] == old_units["nested/README.md"]
    current_files = {
        f.path: f.id
        for f in fixture.session.scalars(
            select(File).where(File.repository_id == fixture.repository.id)
        )
    }
    assert set(current_files) == set(fixture.files)
    for path in set(old_files) & set(current_files):
        assert current_files[path] == old_files[path]
    if change in {"modified", "removed_function"}:
        assert {row[0] for row in old_units["service.py"]}.isdisjoint(
            row[0] for row in new_units["service.py"]
        )
    expected_content = [row[1] for path in refreshed for row in new_units.get(path, [])]
    assert sorted(
        text for batch in fixture.provider.batches for text in batch
    ) == sorted(expected_content)
    snapshot = _snapshot(fixture.session, fixture.repository.id)
    assert len(snapshot["relationships"]) == imports
    assert len(snapshot["code_relationships"]) == calls
    if calls:
        caller_id = next(
            row[0] for row in new_units["api.py"] if row[1].startswith("def caller")
        )
        assert (
            fixture.session.scalar(
                select(CodeRelationship.id).where(
                    CodeRelationship.repository_id == fixture.repository.id,
                    CodeRelationship.source_code_unit_id == caller_id,
                )
            )
            is not None
        )


def test_legacy_null_hash_refreshes_units_not_partial_content(
    fixture: IndexingFixture,
) -> None:
    fixture.run()
    before = _units(fixture)
    file = fixture.session.scalar(
        select(File).where(
            File.repository_id == fixture.repository.id, File.path == "service.py"
        )
    )
    assert file is not None
    fid = file.id
    file.content_hash = None
    fixture.session.flush()
    fixture.provider.batches.clear()
    fixture.run()
    assert (
        file.id == fid
        and file.content_hash == hashlib.sha256(fixture.files[file.path]).hexdigest()
    )
    assert _units(fixture)["service.py"] != before["service.py"]
    assert _units(fixture)["api.py"] == before["api.py"]
    assert fixture.provider.batches == [("def helper():\n    return 1",)]


def test_line_endings_are_content_changes(fixture: IndexingFixture) -> None:
    fixture.run()
    before = _units(fixture)
    fixture.provider.batches.clear()
    fixture.files["service.py"] = fixture.files["service.py"].replace(b"\n", b"\r\n")
    fixture.run()
    assert fixture.provider.batches
    assert _units(fixture)["service.py"] != before["service.py"]
    assert _units(fixture)["api.py"] == before["api.py"]


@pytest.mark.parametrize(
    "phase",
    [
        "discover",
        "hash",
        "reread",
        "parse",
        "embed",
        "imports",
        "calls",
        "persist",
        "cleanup",
    ],
)
def test_failure_preserves_previous_snapshot(
    fixture: IndexingFixture, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    first = fixture.run()
    before = _snapshot(fixture.session, fixture.repository.id)
    fixture.files["service.py"] = b"def replacement(): pass\n"
    fixture.files.pop("nested/README.md")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected failure")

    if phase == "embed":
        monkeypatch.setattr(fixture.provider, "embed", fail)
    elif phase == "parse":
        monkeypatch.setattr(indexing.PythonCodeParser, "parse", fail)
    elif phase == "reread":
        original_read = indexing._read_candidate_bytes
        reads: dict[str, int] = {}

        def inconsistent(candidate: RepositoryFileCandidate) -> bytes:
            raw = original_read(candidate)
            reads[candidate.relative_path] = reads.get(candidate.relative_path, 0) + 1
            return raw + b" " if reads[candidate.relative_path] == 2 else raw

        monkeypatch.setattr(indexing, "_read_candidate_bytes", inconsistent)
    else:
        name = {
            "discover": "discover_repository_files",
            "hash": "_read_candidate_bytes",
            "imports": "persist_import_relationships",
            "calls": "persist_call_relationships",
            "persist": "persist_code_units",
            "cleanup": "_remove_owned_clone",
        }[phase]
        original_function = getattr(indexing, name)

        def fail_after_work(*args: Any, **kwargs: Any) -> Any:
            original_function(*args, **kwargs)
            raise RuntimeError("injected failure")

        monkeypatch.setattr(
            indexing,
            name,
            fail_after_work if phase in {"imports", "calls", "persist"} else fail,
        )
    with pytest.raises((RuntimeError, indexing.RepositoryIndexingError)):
        fixture.run()
    assert _snapshot(fixture.session, fixture.repository.id) == before
    assert (
        fixture.session.scalar(
            select(Repository.index_generation).where(
                Repository.id == fixture.repository.id
            )
        )
        == 1
    )
    jobs = fixture.session.scalars(
        select(IndexingJob).where(IndexingJob.repository_id == fixture.repository.id)
    ).all()
    assert len(jobs) == 2 and sorted(j.status for j in jobs) == ["completed", "failed"]
    assert next(j for j in jobs if j.id == first.id).status == "completed"
    assert (
        all(not path.exists() for path in fixture.clones)
        if phase != "cleanup"
        else fixture.clones[-1].exists()
    )


@pytest.mark.parametrize("change", ["changed", "deleted", "legacy", "added"])
def test_generation_advances_once_per_changed_snapshot(
    fixture: IndexingFixture, change: str
) -> None:
    fixture.run()
    assert fixture.repository.index_generation == 1
    if change == "changed":
        fixture.files["service.py"] = b"def helper(): return 2\n"
    elif change == "deleted":
        fixture.files.clear()
    elif change == "added":
        fixture.files["new.py"] = b"def new(): pass\n"
    else:
        file = fixture.session.scalar(
            select(File).where(File.repository_id == fixture.repository.id)
        )
        assert file is not None
        file.content_hash = None
        fixture.session.flush()
    fixture.run()
    assert fixture.repository.index_generation == 2
    fixture.run()
    assert fixture.repository.index_generation == 2


def test_empty_to_empty_generation_and_caller_rollback(
    fixture: IndexingFixture,
) -> None:
    fixture.files.clear()
    fixture.run()
    assert fixture.repository.index_generation == 0
    outer = fixture.session.begin_nested()
    fixture.files["one.py"] = b"def one(): pass\n"
    fixture.run()
    assert fixture.repository.index_generation == 1
    outer.rollback()
    assert (
        fixture.session.scalar(
            select(Repository.index_generation).where(
                Repository.id == fixture.repository.id
            )
        )
        == 0
    )


@pytest.mark.parametrize("policy", ["binary", "oversize"])
def test_intentional_exclusion_removes_old_derived_data(
    fixture: IndexingFixture, policy: str
) -> None:
    fixture.run()
    fixture.files["service.py"] = (
        b"\x00binary" if policy == "binary" else b"x" * 1_048_577
    )
    fixture.provider.batches.clear()
    fixture.run()
    assert "service.py" not in _units(fixture)
    assert fixture.provider.batches == []
    snapshot = _snapshot(fixture.session, fixture.repository.id)
    assert snapshot["relationships"] == snapshot["code_relationships"] == []


def test_strict_scan_failure_does_not_delete_snapshot(
    fixture: IndexingFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture.run()
    before = _snapshot(fixture.session, fixture.repository.id)
    original = os.scandir

    def blocked(path: Any) -> Any:
        if Path(path).name == "nested":
            raise PermissionError("sensitive path")
        return original(path)

    monkeypatch.setattr(os, "scandir", blocked)
    with pytest.raises(RuntimeError, match="directory could not be scanned"):
        fixture.run()
    assert _snapshot(fixture.session, fixture.repository.id) == before


def test_binary_sniff_mutation_preserves_previous_snapshot(
    fixture: IndexingFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture.files["source.py"] = fixture.files.pop("service.py")
    fixture.files["api.py"] = fixture.files["api.py"].replace(b"service", b"source")
    first = fixture.run()
    before = _snapshot(fixture.session, fixture.repository.id)
    assert all(before.values())
    fixture.provider.batches.clear()
    real_open = os.open
    real_read = os.read
    target_fd: int | None = None
    mutated = False

    def track_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal target_fd
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path).name == "source.py":
            target_fd = fd
        return fd

    def mutate_during_read(fd: int, size: int) -> bytes:
        nonlocal mutated
        if fd == target_fd and not mutated:
            (fixture.clones[-1] / "source.py").write_bytes(b"\x00changed source\n")
            mutated = True
        return real_read(fd, size)

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "read", mutate_during_read)
    with pytest.raises(RuntimeError, match="changed during discovery"):
        fixture.run()
    assert mutated
    assert _snapshot(fixture.session, fixture.repository.id) == before
    assert fixture.provider.batches == []
    jobs = fixture.session.scalars(
        select(IndexingJob).where(IndexingJob.repository_id == fixture.repository.id)
    ).all()
    assert len(jobs) == 2
    assert first.status == "completed"
    assert next(job for job in jobs if job.id != first.id).status == "failed"


def test_repositories_with_identical_hashes_do_not_share_units(
    fixture: IndexingFixture,
) -> None:
    fixture.run()
    original_id = fixture.repository.id
    original = _snapshot(fixture.session, original_id)
    other = Repository(name="other", source=f"https://github.com/example/{uuid4()}.git")
    fixture.session.add(other)
    fixture.session.flush()
    fixture.repository = other
    fixture.provider.batches.clear()
    fixture.run()
    assert fixture.provider.batches
    newer = _snapshot(fixture.session, other.id)
    assert {row[0] for row in original["code_units"]}.isdisjoint(
        row[0] for row in newer["code_units"]
    )
    fixture.files.clear()
    fixture.run()
    assert all(rows == [] for rows in _snapshot(fixture.session, other.id).values())
    assert _snapshot(fixture.session, original_id) == original


def test_repository_lock_is_held_until_outer_transaction_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = os.getenv("DATABASE_URL")
    if url is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(url)
    ids = [uuid4(), uuid4()]
    try:
        with Session(engine) as setup:
            setup.add_all(
                [
                    Repository(
                        id=rid,
                        name="lock",
                        source=f"https://github.com/example/{rid}.git",
                    )
                    for rid in ids
                ]
            )
            setup.commit()

        def clone(source: Any, workspace_root: Path) -> ClonedRepository:
            workspace_root.mkdir(parents=True, exist_ok=True)
            return ClonedRepository(
                source=source.value, path=Path(tempfile.mkdtemp(dir=workspace_root))
            )

        monkeypatch.setattr(indexing, "clone_repository", clone)
        with Session(engine) as owner, Session(engine) as contender:
            indexing.index_repository(
                owner,
                repository_id=ids[0],
                embedding_provider=FakeEmbeddingProvider(),
                workspace_root=tmp_path,
            )
            contender.execute(text("SET LOCAL lock_timeout = '200ms'"))
            # A different repository is not globally serialized.
            indexing.index_repository(
                contender,
                repository_id=ids[1],
                embedding_provider=FakeEmbeddingProvider(),
                workspace_root=tmp_path,
            )
            with pytest.raises(OperationalError) as caught:
                indexing.index_repository(
                    contender,
                    repository_id=ids[0],
                    embedding_provider=FakeEmbeddingProvider(),
                    workspace_root=tmp_path,
                )
            assert getattr(caught.value.orig, "sqlstate", None) == "55P03"
            contender.rollback()
            owner.rollback()
            # Lock is released only by the caller, not when indexing returns.
            assert (
                indexing.index_repository(
                    contender,
                    repository_id=ids[0],
                    embedding_provider=FakeEmbeddingProvider(),
                    workspace_root=tmp_path,
                ).status
                == "completed"
            )
            contender.rollback()
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(delete(Repository).where(Repository.id.in_(ids)))
            cleanup.commit()
        engine.dispose()
