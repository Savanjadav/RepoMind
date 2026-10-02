import asyncio
import os
from collections.abc import Iterator, MutableMapping
from typing import Any
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from redis import Redis
from redis.exceptions import ConnectionError, TimeoutError
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app import indexing_api as api
from app import indexing_jobs as jobs
from app.database import create_database_engine, create_session_factory
from app.main import app
from app.models.indexing_job import IndexingJob
from app.models.repository import Repository
from app.redis_cache import RedisCache, get_redis_cache


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, UUID, list[UUID]]]:
    url = os.getenv("DATABASE_URL")
    if url is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(url)
    factory = create_session_factory(engine)
    rid = uuid4()
    with factory() as session:
        session.add(
            Repository(
                id=rid, name="API indexing", source=f"https://github.com/example/{rid}"
            )
        )
        session.commit()
    calls: list[UUID] = []
    app.dependency_overrides[get_redis_cache] = lambda: None

    def runner(job_id: UUID) -> None:
        assert isinstance(job_id, UUID)
        with factory() as observer:
            assert (
                observer.scalar(
                    select(IndexingJob.status).where(IndexingJob.id == job_id)
                )
                == "pending"
            )
        calls.append(job_id)

    monkeypatch.setattr(api, "get_application_session_factory", lambda: factory)
    monkeypatch.setattr(api, "run_indexing_job", runner)
    monkeypatch.setattr(
        jobs,
        "get_indexing_embedding_provider",
        lambda: pytest.fail("API must not load a provider"),
    )
    try:
        yield factory, rid, calls
    finally:
        app.dependency_overrides.clear()
        with factory() as session:
            session.execute(delete(Repository).where(Repository.id == rid))
            session.commit()
        engine.dispose()


def test_post_commits_exact_job_before_scheduling(
    database: tuple[Any, UUID, list[UUID]],
) -> None:
    factory, rid, calls = database
    with TestClient(app) as client:
        response = client.post(f"/repositories/{rid}/index")
        assert response.status_code == 202
        job_id = UUID(response.json()["job_id"])
        assert response.json() == {
            "job_id": str(job_id),
            "repository_id": str(rid),
            "status": "pending",
        }
        assert (
            response.headers["location"]
            == f"/repositories/{rid}/indexing-jobs/{job_id}"
        )
        assert client.get(response.headers["location"]).json() == response.json()
    assert calls == [job_id]
    with factory() as session:
        assert list(
            session.scalars(
                select(IndexingJob.id).where(IndexingJob.repository_id == rid)
            )
        ) == [job_id]


@pytest.mark.parametrize("status", ["pending", "running", "completed", "failed"])
def test_status_and_history(
    database: tuple[Any, UUID, list[UUID]], status: str
) -> None:
    factory, rid, calls = database
    with factory() as session:
        job = IndexingJob(repository_id=rid, status=status)
        session.add(job)
        session.commit()
        job_id = job.id
    with TestClient(app) as client:
        response = client.get(f"/repositories/{rid}/indexing-jobs/{job_id}")
        assert response.status_code == 200
        assert response.json() == {
            "job_id": str(job_id),
            "repository_id": str(rid),
            "status": status,
        }
        assert calls == []
        response = client.post(f"/repositories/{rid}/index")
        assert response.status_code == (
            409 if status in {"pending", "running"} else 202
        )


def test_missing_and_mismatched_resources(
    database: tuple[Any, UUID, list[UUID]],
) -> None:
    factory, rid, calls = database
    with factory() as session:
        job = IndexingJob(repository_id=rid)
        session.add(job)
        session.commit()
        job_id = job.id
    with TestClient(app) as client:
        assert client.post(f"/repositories/{uuid4()}/index").status_code == 404
        assert (
            client.get(f"/repositories/{rid}/indexing-jobs/{uuid4()}").status_code
            == 404
        )
        assert (
            client.get(f"/repositories/{uuid4()}/indexing-jobs/{job_id}").status_code
            == 404
        )
        assert client.post("/repositories/not-a-uuid/index").status_code == 422
    assert calls == []


def test_unsupported_source(database: tuple[Any, UUID, list[UUID]]) -> None:
    factory, rid, calls = database
    with factory() as session:
        repo = session.get(Repository, rid)
        assert repo is not None
        repo.source = "/SECRET/local/path"
        session.commit()
    with TestClient(app) as client:
        response = client.post(f"/repositories/{rid}/index")
    assert response.status_code == 422
    assert "SECRET" not in response.text
    assert calls == []


def test_lock_contention_is_409(database: tuple[Any, UUID, list[UUID]]) -> None:
    factory, rid, calls = database
    with factory() as owner:
        owner.scalar(select(Repository).where(Repository.id == rid).with_for_update())
        with TestClient(app) as client:
            assert client.post(f"/repositories/{rid}/index").status_code == 409
    assert calls == []


@pytest.mark.parametrize("failure", ["commit", "schedule"])
def test_failed_acceptance_never_executes_task(
    database: tuple[Any, UUID, list[UUID]],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    factory, rid, calls = database

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("SECRET external exception")

    if failure == "commit":
        monkeypatch.setattr(Session, "commit", fail)
    else:
        monkeypatch.setattr(api.BackgroundTasks, "add_task", fail)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(f"/repositories/{rid}/index")
    assert response.status_code == 500
    assert "SECRET" not in response.text
    assert calls == []
    monkeypatch.undo()
    with factory() as session:
        assert (
            session.scalars(
                select(IndexingJob).where(IndexingJob.repository_id == rid)
            ).all()
            == []
        )


@pytest.mark.parametrize(
    "kind,expected", [("connection", 503), ("integrity", 500), ("defect", 500)]
)
def test_sanitized_request_errors(
    database: tuple[Any, UUID, list[UUID]],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    expected: int,
) -> None:
    _, rid, calls = database

    def fail(*args: Any, **kwargs: Any) -> Any:
        if kind == "connection":
            raise OperationalError("SECRET SQL", {}, OSError("SECRET credentials"))
        if kind == "integrity":
            raise IntegrityError("SECRET SQL", {}, ValueError("SECRET constraint"))
        raise AttributeError("SECRET defect")

    monkeypatch.setattr(api, "reserve_indexing_job", fail)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(f"/repositories/{rid}/index")
    assert response.status_code == expected
    assert "SECRET" not in response.text and calls == []


def test_final_response_body_precedes_runner(
    database: tuple[Any, UUID, list[UUID]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, rid, _ = database
    events: list[str] = []

    def runner(job_id: UUID) -> None:
        assert events == ["start", "body"]
        events.append("runner")

    monkeypatch.setattr(api, "run_indexing_job", runner)

    async def exercise() -> None:
        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                assert message["status"] == 202
                events.append("start")
            elif message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                events.append("body")

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": f"/repositories/{rid}/index",
                "raw_path": f"/repositories/{rid}/index".encode(),
                "query_string": b"",
                "headers": [],
                "server": ("test", 80),
                "client": ("test", 1),
                "root_path": "",
            },
            receive,
            send,
        )

    asyncio.run(exercise())
    assert events == ["start", "body", "runner"]


@pytest.mark.parametrize(
    "code,expected",
    [("08006", 503), ("57P01", 503), ("40001", 500), ("55P03", 500), ("42601", 500)],
)
def test_database_error_mapping_is_narrow(code: str, expected: int) -> None:
    class DriverError(Exception):
        sqlstate = code

    error = OperationalError("SECRET SQL", {}, DriverError("SECRET message"))
    mapped = api._request_error(error)
    assert mapped.status_code == expected
    assert "SECRET" not in str(mapped.detail)


def test_status_is_read_only(
    database: tuple[Any, UUID, list[UUID]], monkeypatch: pytest.MonkeyPatch
) -> None:
    factory, rid, calls = database
    with factory() as session:
        job = IndexingJob(repository_id=rid)
        session.add(job)
        session.commit()
        job_id = job.id

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Status must not write or commit")

    monkeypatch.setattr(Session, "commit", forbidden)
    monkeypatch.setattr(Session, "flush", forbidden)
    with TestClient(app) as client:
        assert (
            client.get(f"/repositories/{rid}/indexing-jobs/{job_id}").status_code == 200
        )
    assert calls == []
    monkeypatch.undo()


@pytest.mark.parametrize("mode", ["held", "acquired", "outage", "release_failure"])
def test_advisory_reservation_lease(
    database: tuple[Any, UUID, list[UUID]], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    factory, rid, calls = database
    client = Mock(spec=Redis)
    client.set.return_value = None if mode == "held" else True
    if mode == "outage":
        client.set.side_effect = ConnectionError("SECRET")
    if mode == "release_failure":
        client.eval.side_effect = TimeoutError("SECRET")
    cache = RedisCache(client)
    app.dependency_overrides[get_redis_cache] = lambda: cache

    def runner(job_id: UUID) -> None:
        if mode in {"acquired", "release_failure"}:
            assert client.eval.call_count == 1  # Released before dispatch.
        calls.append(job_id)

    monkeypatch.setattr(api, "run_indexing_job", runner)
    with TestClient(app) as http:
        response = http.post(f"/repositories/{rid}/index")
        assert response.status_code == (409 if mode == "held" else 202)
        if mode != "held":
            assert len(calls) == 1
            # Simulate expiry/reacquisition: Redis allows, PostgreSQL still blocks.
            assert http.post(f"/repositories/{rid}/index").status_code == 409
            assert len(calls) == 1
    with factory() as session:
        jobs_for_repo = list(
            session.scalars(
                select(IndexingJob.id).where(IndexingJob.repository_id == rid)
            )
        )
    assert len(jobs_for_repo) == (0 if mode == "held" else 1)


def test_failed_commit_releases_lease_without_dispatch(
    database: tuple[Any, UUID, list[UUID]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, rid, calls = database
    client = Mock(spec=Redis)
    client.set.return_value = True
    app.dependency_overrides[get_redis_cache] = lambda: RedisCache(client)

    def fail(session: Session) -> None:
        raise RuntimeError("commit failure")

    monkeypatch.setattr(Session, "commit", fail)
    with TestClient(app, raise_server_exceptions=False) as http:
        assert http.post(f"/repositories/{rid}/index").status_code == 500
    assert calls == []
    assert client.eval.call_count == 1
    monkeypatch.undo()
