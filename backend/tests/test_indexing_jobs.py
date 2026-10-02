import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from types import ModuleType
from typing import Any
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from redis import Redis
from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from app import indexing_jobs as jobs
from app import repository_indexing as indexing
from app.database import create_database_engine, create_session_factory
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.indexing_job import IndexingJob
from app.models.relationship import Relationship
from app.models.repository import Repository
from app.redis_cache import RedisCache
from app.repository_clone import ClonedRepository
from app.search_api import search


class FakeProvider:
    dimension = 384

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return [[1.0] + [0.0] * 383 for _ in texts]


@dataclass
class State:
    factory: Any
    ids: list[UUID]
    provider: FakeProvider = field(default_factory=FakeProvider)
    clones: list[Path] = field(default_factory=list)
    observed: list[str] = field(default_factory=list)
    files: dict[str, bytes] = field(
        default_factory=lambda: {
            "api.py": b"from service import helper\ndef caller(): return helper()\n",
            "service.py": b"def helper(): return 1\n",
        }
    )

    def reserve(self, index: int = 0) -> UUID:
        with self.factory() as session:
            job = jobs.reserve_indexing_job(session, self.ids[index])
            session.commit()
            return job.id

    def status(self, job_id: UUID) -> str | None:
        with self.factory() as session:
            return session.scalar(
                select(IndexingJob.status).where(IndexingJob.id == job_id)
            )

    def generation(self) -> int | None:
        with self.factory() as session:
            return session.scalar(
                select(Repository.index_generation).where(Repository.id == self.ids[0])
            )


@pytest.fixture
def state(monkeypatch: pytest.MonkeyPatch) -> Iterator[State]:
    url = os.getenv("DATABASE_URL")
    if url is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(url)
    factory = create_session_factory(engine)
    ids = [uuid4(), uuid4()]
    with factory() as session:
        session.add_all(
            [
                Repository(
                    id=rid, name="jobs", source=f"https://github.com/example/{rid}"
                )
                for rid in ids
            ]
        )
        session.commit()
    value = State(factory, ids)
    monkeypatch.setattr(jobs, "get_application_session_factory", lambda: factory)
    monkeypatch.setattr(jobs, "get_indexing_embedding_provider", lambda: value.provider)

    def clone(source: Any, workspace: Path) -> ClonedRepository:
        path = Path(tempfile.mkdtemp(dir=workspace))
        for name, content in value.files.items():
            (path / name).write_bytes(content)
        value.clones.append(path)
        with factory() as observer:
            value.observed.extend(
                observer.scalars(
                    select(IndexingJob.status).where(
                        IndexingJob.repository_id == ids[0],
                        IndexingJob.status == "running",
                    )
                )
            )
        return ClonedRepository(source.value, path)

    monkeypatch.setattr(indexing, "clone_repository", clone)
    try:
        yield value
    finally:
        with factory() as session:
            session.execute(delete(Repository).where(Repository.id.in_(ids)))
            session.commit()
        engine.dispose()


def snapshot(state: State) -> dict[str, list[tuple[Any, ...]]]:
    result = {}
    with state.factory() as session:
        for model in (File, CodeUnit, Relationship, CodeRelationship):
            table = model.__table__
            predicate = (
                table.c.file_id.in_(
                    select(File.id).where(File.repository_id == state.ids[0])
                )
                if model is CodeUnit
                else table.c.repository_id == state.ids[0]
            )
            result[model.__tablename__] = [
                tuple(tuple(v.tolist()) if hasattr(v, "tolist") else v for v in row)
                for row in session.execute(
                    select(table).where(predicate).order_by(table.c.id)
                )
            ]
    return result


@pytest.mark.parametrize("status", ["pending", "running", "completed", "failed"])
def test_reservation_respects_history(state: State, status: str) -> None:
    with state.factory() as session:
        session.add(IndexingJob(repository_id=state.ids[0], status=status))
        session.commit()
    if status in {"pending", "running"}:
        with pytest.raises(jobs.IndexingConflictError):
            state.reserve()
    else:
        assert state.status(state.reserve()) == "pending"
    assert state.provider.calls == [] and state.clones == []


@pytest.mark.parametrize(
    "source",
    [
        "/local/repository",
        "https://example.com/repo",
        "https://github.com/a/b?secret=1",
    ],
)
def test_unsupported_source_cannot_reserve(state: State, source: str) -> None:
    with state.factory() as session:
        repo = session.get(Repository, state.ids[0])
        assert repo is not None
        repo.source = source
        session.commit()
    with pytest.raises(jobs.InvalidIndexingSourceError):
        state.reserve()
    with state.factory() as session:
        assert (
            session.scalars(
                select(IndexingJob).where(IndexingJob.repository_id == state.ids[0])
            ).all()
            == []
        )


def test_missing_repository_cannot_reserve(state: State) -> None:
    with state.factory() as session, pytest.raises(jobs.RepositoryNotFoundError):
        jobs.reserve_indexing_job(session, uuid4())


def test_reservation_lock_and_commit_visibility(state: State) -> None:
    with state.factory() as owner, state.factory() as contender:
        job = jobs.reserve_indexing_job(owner, state.ids[0])
        assert state.status(job.id) is None
        with pytest.raises(jobs.IndexingConflictError):
            jobs.reserve_indexing_job(contender, state.ids[0])
        contender.rollback()
        other = jobs.reserve_indexing_job(contender, state.ids[1])
        contender.commit()
        assert state.status(other.id) == "pending"
        owner.commit()
        assert state.status(job.id) == "pending"
        with pytest.raises(jobs.IndexingConflictError):
            jobs.reserve_indexing_job(contender, state.ids[0])


def test_runner_uses_exact_job_and_commits_visible_running(state: State) -> None:
    assert state.generation() == 0
    job_id = state.reserve()
    jobs.run_indexing_job(job_id)
    assert state.observed == ["running"]
    assert state.status(job_id) == "completed"
    assert state.generation() == 1
    before = snapshot(state)
    assert all(before.values())
    assert state.provider.calls
    assert all(not p.exists() and not p.parent.exists() for p in state.clones)
    jobs.run_indexing_job(job_id)
    assert len(state.clones) == 1
    with state.factory() as session:
        assert list(
            session.scalars(
                select(IndexingJob.id).where(IndexingJob.repository_id == state.ids[0])
            )
        ) == [job_id]
    state.provider.calls.clear()
    jobs.run_indexing_job(state.reserve())
    assert state.provider.calls == []
    assert snapshot(state) == before
    assert state.generation() == 1


@pytest.mark.parametrize("status", ["running", "completed", "failed", "missing"])
def test_unclaimable_jobs_do_not_acquire_provider(
    state: State, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    job_id = uuid4()
    if status != "missing":
        with state.factory() as session:
            session.add(
                IndexingJob(id=job_id, repository_id=state.ids[0], status=status)
            )
            session.commit()
    monkeypatch.setattr(
        jobs,
        "get_indexing_embedding_provider",
        lambda: pytest.fail("Unexpected provider acquisition"),
    )
    jobs.run_indexing_job(job_id)
    assert state.status(job_id) == (None if status == "missing" else status)


@pytest.mark.parametrize(
    "phase",
    [
        "provider",
        "clone",
        "embed",
        "imports",
        "calls",
        "cleanup",
        "workspace",
        "commit",
        "database",
    ],
)
def test_background_failure_preserves_snapshot(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    phase: str,
) -> None:
    first = state.reserve()
    jobs.run_indexing_job(first)
    before = snapshot(state)
    assert all(before.values())
    state.files["service.py"] = b"def helper(): return 42\n"
    second = state.reserve()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("PRIVATE SOURCE credential SECRET")

    if phase == "provider":
        monkeypatch.setattr(jobs, "get_indexing_embedding_provider", fail)
    elif phase == "embed":
        monkeypatch.setattr(state.provider, "embed", fail)
    elif phase == "workspace":

        class FailingWorkspace(TemporaryDirectory[str]):
            def __exit__(self, *args: Any) -> None:
                super().__exit__(*args)
                fail()

        monkeypatch.setattr(jobs, "TemporaryDirectory", FailingWorkspace)
    elif phase == "commit":
        real_commit = Session.commit

        def commit(session: Session) -> None:
            if (
                session.scalar(
                    select(IndexingJob.status).where(IndexingJob.id == second)
                )
                == "completed"
            ):
                fail()
            real_commit(session)

        monkeypatch.setattr(Session, "commit", commit)
    elif phase == "database":
        execute = jobs.execute_reserved_indexing_job

        def abort_transaction(session: Session, **kwargs: Any) -> Any:
            execute(session, **kwargs)
            session.execute(text("SELECT 1 / 0"))

        monkeypatch.setattr(jobs, "execute_reserved_indexing_job", abort_transaction)
    else:
        name = {
            "clone": "clone_repository",
            "imports": "persist_import_relationships",
            "calls": "persist_call_relationships",
            "cleanup": "_remove_owned_clone",
        }[phase]
        original = getattr(indexing, name)

        def after_work(*args: Any, **kwargs: Any) -> Any:
            original(*args, **kwargs)
            fail()

        monkeypatch.setattr(
            indexing,
            name,
            after_work if phase in {"imports", "calls", "cleanup"} else fail,
        )
    jobs.run_indexing_job(second)
    assert state.status(first) == "completed"
    assert state.status(second) == "failed"
    assert snapshot(state) == before
    assert state.generation() == 1
    assert "SECRET" not in caplog.text and "PRIVATE SOURCE" not in caplog.text


def test_failure_update_does_not_overwrite_completed(state: State) -> None:
    job_id = state.reserve()
    jobs.run_indexing_job(job_id)
    jobs._mark_failed(job_id, state.ids[0])
    assert state.status(job_id) == "completed"


def test_claim_has_one_winner(state: State) -> None:
    job_id = state.reserve()
    with state.factory() as one, state.factory() as two:
        assert jobs._claim_job(one, job_id) == state.ids[0]
        assert state.status(job_id) == "running"
        assert jobs._claim_job(two, job_id) is None


@pytest.mark.parametrize("failure", [False, True])
def test_runner_closes_independent_sessions(
    state: State, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    job_id = state.reserve()
    sessions: list[Session] = []
    closed: list[Session] = []

    class TrackedSession(Session):
        def close(self) -> None:
            closed.append(self)
            super().close()

    def factory() -> Session:
        session = TrackedSession(bind=state.factory.kw["bind"], expire_on_commit=False)
        sessions.append(session)
        return session

    monkeypatch.setattr(jobs, "get_application_session_factory", lambda: factory)
    if failure:

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("embedding failure")

        monkeypatch.setattr(state.provider, "embed", fail)
    jobs.run_indexing_job(job_id)
    assert len(sessions) == (3 if failure else 2)
    assert len({id(session) for session in sessions}) == len(sessions)
    assert closed == sessions
    assert state.status(job_id) == ("failed" if failure else "completed")


def test_simultaneous_claims_execute_only_once(state: State) -> None:
    job_id = state.reserve()
    barrier = Barrier(2)

    def claim() -> UUID | None:
        with state.factory() as session:
            barrier.wait(timeout=10)
            return jobs._claim_job(session, job_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results.count(state.ids[0]) == 1 and results.count(None) == 1
    assert state.status(job_id) == "running"


def test_unavailable_factory_is_logged_without_secrets(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail() -> Any:
        raise RuntimeError("SECRET database credentials")

    monkeypatch.setattr(jobs, "get_application_session_factory", fail)
    jobs.run_indexing_job(uuid4())
    jobs._mark_failed(uuid4(), uuid4())
    assert "SECRET" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_provider_adapter_preserves_contract() -> None:
    fake = FakeProvider()
    provider = jobs._SerializedEmbeddingProvider(fake)
    assert provider.dimension == 384
    assert len(provider.embed(["one", "two"])) == 2
    assert fake.calls == [("one", "two")]


def test_provider_initialization_is_cached_and_synchronized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("app.sentence_transformer_embedding_provider")
    constructed: list[FakeProvider] = []

    def construct() -> FakeProvider:
        provider = FakeProvider()
        constructed.append(provider)
        return provider

    monkeypatch.setattr(
        module, "SentenceTransformerEmbeddingProvider", construct, raising=False
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(jobs, "_provider", None)
    barrier = Barrier(4)

    def acquire() -> Any:
        barrier.wait(timeout=10)
        return jobs.get_indexing_embedding_provider()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(acquire) for _ in range(4)]
        providers = [f.result(timeout=10) for f in futures]
    assert len(constructed) == 1
    assert all(p is providers[0] for p in providers)
    assert constructed[0].calls == []


def test_failed_claim_commit_does_not_execute(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = state.reserve()

    def fail(session: Session) -> None:
        raise RuntimeError("commit unavailable")

    monkeypatch.setattr(Session, "commit", fail)
    monkeypatch.setattr(
        jobs,
        "get_indexing_embedding_provider",
        lambda: pytest.fail("Uncommitted claim must not execute"),
    )
    jobs.run_indexing_job(job_id)
    assert state.status(job_id) == "pending"
    assert state.clones == []
    monkeypatch.undo()


def test_committed_completion_survives_ambiguous_commit_error(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = state.reserve()
    original = Session.commit

    def commit(session: Session) -> None:
        completed = (
            session.scalar(select(IndexingJob.status).where(IndexingJob.id == job_id))
            == "completed"
        )
        original(session)
        if completed:
            raise RuntimeError("connection lost after commit")

    monkeypatch.setattr(Session, "commit", commit)
    jobs.run_indexing_job(job_id)
    assert state.status(job_id) == "completed"
    assert all(snapshot(state).values())
    assert state.generation() == 1
    monkeypatch.undo()


def test_cache_survives_no_change_and_failure_but_not_changed_commit(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    values: dict[str, bytes] = {}
    client = Mock(spec=Redis)
    client.get.side_effect = values.get
    client.set.side_effect = lambda key, value, **kwargs: values.__setitem__(key, value)
    cache = RedisCache(client)
    jobs.run_indexing_job(state.reserve())

    def query() -> Any:
        with state.factory() as session:
            return search(
                repository_id=state.ids[0],
                q="helper",
                session=session,
                provider_factory=lambda: state.provider,
                cache=cache,
                limit=10,
            )

    first = query()
    assert first.items and len(values) == 1
    jobs.run_indexing_job(state.reserve())
    state.provider.calls.clear()
    assert query() == first
    assert state.provider.calls == []
    state.files["service.py"] = b"def helper(): return 42\n"
    original = state.provider.embed

    def fail(texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedding failure")

    monkeypatch.setattr(state.provider, "embed", fail)
    failed = state.reserve()
    jobs.run_indexing_job(failed)
    assert state.status(failed) == "failed" and state.generation() == 1
    assert query() == first  # Cached response never touches the failing provider.
    monkeypatch.setattr(state.provider, "embed", original)
    jobs.run_indexing_job(state.reserve())
    assert state.generation() == 2
    state.provider.calls.clear()
    latest = query()
    assert state.provider.calls == [("helper",)]
    assert any("42" in item.content for item in latest.items)
    assert len(values) == 2


def test_generation_and_snapshot_become_visible_together_after_cleanup(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs.run_indexing_job(state.reserve())
    old_snapshot = snapshot(state)
    assert all(old_snapshot.values()) and state.generation() == 1
    state.files["service.py"] = b"def helper(): return 42\n"
    original = indexing._remove_owned_clone
    observed: list[int | None] = []

    def cleanup(path: Path, workspace: Path) -> None:
        # The writer has flushed its new snapshot/generation by this point.
        # Independent connections must still observe the last committed pair.
        observed.append(state.generation())
        assert snapshot(state) == old_snapshot
        original(path, workspace)

    monkeypatch.setattr(indexing, "_remove_owned_clone", cleanup)
    job = state.reserve()
    jobs.run_indexing_job(job)
    assert observed == [1]
    assert state.status(job) == "completed" and state.generation() == 2
    assert snapshot(state) != old_snapshot
