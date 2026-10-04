import os
from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.database import create_database_engine, get_db_session
from app.main import app
from app.models import IndexingJob, Repository


@pytest.mark.parametrize("source", ["//[", "//[not-an-ip]/repo", "/tmp/repo"])
def test_local_source_rejected_before_persistence(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Session() as session:

        def reject_persistence(*args: object, **kwargs: object) -> None:
            pytest.fail("Invalid registration must not attempt persistence")

        for method in ("add", "flush", "commit", "execute"):
            monkeypatch.setattr(session, method, reject_persistence)

        def override() -> Iterator[Session]:
            yield session

        app.dependency_overrides[get_db_session] = override
        try:
            with TestClient(app) as client:
                response = client.post("/repositories", json={"source": source})
            assert response.status_code == 422
            assert response.json() == {
                "detail": "Use a valid GitHub HTTPS repository URL"
            }
            assert source not in response.text
            assert not session.new
        finally:
            app.dependency_overrides.pop(get_db_session, None)


@pytest.fixture
def api() -> Iterator[tuple[TestClient, Session]]:
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(url)
    with engine.connect() as connection:
        outer = connection.begin()
        with Session(connection, join_transaction_mode="create_savepoint") as session:

            def override() -> Iterator[Session]:
                yield session

            app.dependency_overrides[get_db_session] = override
            try:
                with TestClient(app) as client:
                    yield client, session
            finally:
                app.dependency_overrides.pop(get_db_session, None)
        outer.rollback()
    engine.dispose()


@pytest.mark.parametrize("suffix", ["", "/", ".git", ".git/"])
def test_create_canonical_metadata_only(
    api: tuple[TestClient, Session], suffix: str
) -> None:
    client, session = api
    name = f"Repo-{uuid4()}"
    before_jobs = session.scalar(select(func.count()).select_from(IndexingJob))
    response = client.post(
        "/repositories", json={"source": f"https://GitHub.com/Owner/{name}{suffix}"}
    )
    assert response.status_code == 201
    data = response.json()
    assert set(data) == {"id", "name", "source", "created_at"}
    assert data["source"] == f"https://github.com/owner/{name.lower()}"
    assert data["name"] == name.lower()
    assert datetime.fromisoformat(data["created_at"]).tzinfo is not None
    session.expunge_all()
    stored = session.get(Repository, UUID(data["id"]))
    assert stored is not None and stored.source == data["source"]
    assert stored.index_generation == 0 and stored.files == []
    assert stored.indexing_jobs == []
    assert session.scalar(select(func.count()).select_from(IndexingJob)) == before_jobs


@pytest.mark.parametrize(
    "source",
    [
        "http://github.com/a/b",
        "https://example.com/a/b",
        "https://user:SECRET@github.com/a/b",
        "https://github.com/a/b?secret=1",
        "https://github.com/a/b#secret",
        "https://github.com:8443/a/b",
        "https://github.com:/a/b",
        "https://github.com/a/b/tree/main",
        "https://github.com/a/../b",
        "https://github.com/a/%2e%2e",
        "https://github.com/a%2fb/c",
        "https://github.com/a/b%5cc",
        "https://github.com/a/b%252fc",
        "https://github.com//a/b",
        "https://github.com/a//b",
        "https://github.com/a/b//",
        "https://github.com/a",
        "https://github.com/",
        "https://[",
        "https://github.com/a/.git",
        "https://github.com/a/..git",
        "https://github.com/a/b\\c",
        "https://git\nhub.com/a/b",
        "https://github.com/a/b\x00",
        " https://github.com/a/b",
        "https://github.com/a/b ",
        "/tmp/repo",
        "",
        "x" * 2049,
        "https://github.com/-owner/repo",
        "https://github.com/a/" + "b" * 101,
    ],
)
def test_invalid_source_does_not_mutate(
    api: tuple[TestClient, Session], source: str
) -> None:
    client, session = api
    before = session.scalar(select(func.count()).select_from(Repository))
    assert client.post("/repositories", json={"source": source}).status_code == 422
    assert session.scalar(select(func.count()).select_from(Repository)) == before


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"source": None},
        {"source": 4},
        {"source": []},
        *[
            {"source": "https://github.com/a/b", key: "forbidden"}
            for key in ("id", "name", "source_type", "index_generation", "created_at")
        ],
    ],
)
def test_strict_request(api: tuple[TestClient, Session], body: object) -> None:
    assert api[0].post("/repositories", json=body).status_code == 422


def test_duplicate_database_constraint_and_session_recovery(
    api: tuple[TestClient, Session],
) -> None:
    client, session = api
    source = f"https://github.com/owner/repo-{uuid4()}"
    first = client.post("/repositories", json={"source": source})
    assert first.status_code == 201
    for equivalent in (source, source.upper() + ".git/", source + "/"):
        response = client.post("/repositories", json={"source": equivalent})
        assert response.status_code == 409
        assert response.json() == {"detail": "This repository is already registered"}
    # These requests execute real duplicate INSERTs, not a mocked pre-check.
    assert (
        session.scalar(
            select(func.count())
            .select_from(Repository)
            .where(Repository.source == source)
        )
        == 1
    )
    assert (
        client.post("/repositories", json={"source": source + "-other"}).status_code
        == 201
    )


def test_historical_sources_are_not_rewritten(api: tuple[TestClient, Session]) -> None:
    client, session = api
    name = f"Repo-{uuid4()}"
    old = Repository(name=name, source=f"https://github.com/Owner/{name}.git")
    session.add(old)
    session.flush()
    old_id, old_source = old.id, old.source
    response = client.post("/repositories", json={"source": old_source})
    # Raw-string uniqueness cannot establish historical semantic equivalence.
    assert response.status_code == 201
    assert UUID(response.json()["id"]) != old_id
    stored = session.get(Repository, old_id)
    assert stored is not None and stored.source == old_source


def test_list_order_pagination_and_public_fields(
    api: tuple[TestClient, Session],
) -> None:
    client, session = api
    # Far-future timestamps isolate ordering from pre-existing fixture rows.
    ids = sorted([uuid4(), uuid4()])
    for rid in ids:
        session.add(
            Repository(
                id=rid,
                name="historical",
                source=f"/local/{uuid4()}",
                created_at=datetime(9998, 1, 1, tzinfo=UTC),
                index_generation=7,
            )
        )
    session.flush()
    first = client.get("/repositories?limit=1").json()
    second = client.get("/repositories?limit=1&offset=1").json()
    assert first["limit"] == 1 and first["offset"] == 0
    assert first["items"][0]["id"] == str(ids[1])
    assert second["items"][0]["id"] == str(ids[0])
    assert set(first["items"][0]) == {"id", "name", "source", "created_at"}
    count = session.scalar(select(func.count()).select_from(Repository))
    assert count is not None
    assert client.get(f"/repositories?offset={count}").json() == {
        "items": [],
        "limit": 100,
        "offset": count,
    }


@pytest.mark.parametrize(
    "query", ["limit=0", "limit=101", "limit=x", "offset=-1", "offset=x"]
)
def test_list_invalid_bounds(api: tuple[TestClient, Session], query: str) -> None:
    assert api[0].get(f"/repositories?{query}").status_code == 422


@pytest.mark.parametrize("unavailable", [False, True])
def test_database_errors_are_sanitized(
    api: tuple[TestClient, Session], monkeypatch: pytest.MonkeyPatch, unavailable: bool
) -> None:
    client, session = api

    def fail() -> None:
        error = OperationalError if unavailable else IntegrityError
        raise error("SECRET SQL", {}, Exception("SECRET DATABASE"))

    monkeypatch.setattr(session, "flush", fail)
    response = client.post("/repositories", json={"source": "https://github.com/a/b"})
    assert response.status_code == (503 if unavailable else 500)
    assert "SECRET" not in response.text
