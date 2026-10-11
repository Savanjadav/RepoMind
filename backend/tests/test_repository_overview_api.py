import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session

from app.database import create_database_engine, get_db_session
from app.main import app
from app.models import CodeUnit, File, IndexingJob, Repository


@pytest.fixture
def api() -> Iterator[tuple[TestClient, Session]]:
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(url)
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection, autoflush=False) as session:

            def override() -> Iterator[Session]:
                yield session

            app.dependency_overrides[get_db_session] = override
            try:
                with TestClient(app) as client:
                    yield client, session
            finally:
                app.dependency_overrides.pop(get_db_session, None)
        transaction.rollback()
    engine.dispose()


def repository(session: Session) -> Repository:
    row = Repository(name="Overview", source=f"https://example.com/{uuid4()}")
    session.add(row)
    session.flush()
    return row


def units(session: Session, owner: Repository, languages: list[str]) -> None:
    file = File(repository_id=owner.id, path="same.py")
    session.add(file)
    session.flush()
    for index, language in enumerate(languages):
        session.add(
            CodeUnit(
                file_id=file.id,
                kind="function" if index % 2 else "document",
                language=language,
                content="SECRET source must not appear",
                symbol_name="same" if index % 2 else None,
                start_line=1,
                end_line=1,
            )
        )
    session.flush()


def test_invalid_and_unknown_repository(api: tuple[TestClient, Session]) -> None:
    client, _ = api
    assert client.get("/repositories/not-a-uuid/overview").status_code == 422
    response = client.get(f"/repositories/{uuid4()}/overview")
    assert response.status_code == 404
    assert response.json() == {"detail": "Repository not found"}


@pytest.mark.parametrize("files", [0, 1])
@pytest.mark.parametrize("status", [None, "pending", "running", "completed", "failed"])
def test_empty_overview(
    api: tuple[TestClient, Session], files: int, status: str | None
) -> None:
    client, session = api
    owner = repository(session)
    if files:
        session.add(File(repository_id=owner.id, path="unsupported.source"))
        session.flush()
    job = None
    if status is not None:
        job = IndexingJob(repository_id=owner.id, status=status)
        session.add(job)
        session.flush()
    response = client.get(f"/repositories/{owner.id}/overview")
    assert response.status_code == 200
    data = response.json()
    latest = data.pop("latest_job")
    if job is None:
        assert latest is None
    else:
        assert latest["job_id"] == str(job.id)
        assert latest["status"] == status
    assert data == {
        "repository_id": str(owner.id),
        "index_generation": "0",
        "snapshot_counts": {"files": files, "code_units": 0, "named_code_units": 0},
        "code_units_by_language": [],
        "code_units_by_kind": [],
    }


@pytest.mark.parametrize("status", ["pending", "running", "completed", "failed"])
def test_single_statement_retained_snapshot_and_isolation(
    api: tuple[TestClient, Session], status: str
) -> None:
    client, session = api
    owner, other = repository(session), repository(session)
    owner.index_generation = 9223372036854775807
    units(
        session,
        owner,
        ["python"] * 2 + ["rust"] * 4 + ["go"] * 3 + ["java"] * 2 + ["config"],
    )
    units(session, other, ["javascript", "typescript", "documentation"])
    session.add(File(repository_id=owner.id, path="no-units.txt"))
    now = datetime.now(UTC)
    session.add(
        IndexingJob(
            repository_id=owner.id,
            status="completed",
            created_at=now - timedelta(days=1),
        )
    )
    job = IndexingJob(repository_id=owner.id, status=status, created_at=now)
    session.add(job)
    session.flush()
    statements: list[str] = []

    def observe(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        statements.append(statement)

    connection = session.connection()
    event.listen(connection, "before_cursor_execute", observe)
    try:
        response = client.get(f"/repositories/{owner.id}/overview")
    finally:
        event.remove(connection, "before_cursor_execute", observe)
    assert response.status_code == 200
    assert len(statements) == 1
    assert statements[0].lstrip().startswith("WITH repository_files")
    assert "FOR UPDATE" not in statements[0]
    assert not session.new and not session.dirty and not session.deleted
    data = response.json()
    assert data["index_generation"] == "9223372036854775807"
    assert data["latest_job"]["job_id"] == str(job.id)
    assert data["latest_job"]["status"] == status
    assert data["snapshot_counts"] == {
        "files": 2,
        "code_units": 12,
        "named_code_units": 6,
    }
    assert data["code_units_by_language"] == [
        {"language": "config", "count": 1},
        {"language": "other", "count": 9},
        {"language": "python", "count": 2},
    ]
    assert data["code_units_by_kind"] == [
        {"kind": "document", "count": 6},
        {"kind": "function", "count": 6},
    ]
    assert "SECRET" not in response.text
    assert client.get(f"/repositories/{owner.id}/overview").json() == data


def test_all_supported_categories(api: tuple[TestClient, Session]) -> None:
    client, session = api
    owner = repository(session)
    languages = ["python", "javascript", "typescript", "documentation", "config"]
    units(session, owner, languages)
    file_id = session.query(File.id).filter_by(repository_id=owner.id).scalar()
    for kind in ["class", "config", "import"]:
        session.add(
            CodeUnit(
                file_id=file_id,
                kind=kind,
                language="python",
                content="x",
                start_line=1,
                end_line=1,
            )
        )
    session.flush()
    data = client.get(f"/repositories/{owner.id}/overview").json()
    assert {item["language"] for item in data["code_units_by_language"]} == set(
        languages
    )
    assert {item["kind"] for item in data["code_units_by_kind"]} == {
        "class",
        "config",
        "document",
        "function",
        "import",
    }
    assert sum(item["count"] for item in data["code_units_by_language"]) == 8
    assert sum(item["count"] for item in data["code_units_by_kind"]) == 8


@pytest.mark.parametrize(
    "error",
    [PoolTimeoutError("SECRET"), OperationalError("SECRET", {}, OSError("SECRET"))],
)
def test_database_errors_are_sanitized(error: Exception) -> None:
    from unittest.mock import Mock

    session = Mock(spec=Session)
    session.execute.side_effect = error

    def override() -> Iterator[Session]:
        yield session

    app.dependency_overrides[get_db_session] = override
    try:
        with TestClient(app) as client:
            response = client.get(f"/repositories/{uuid4()}/overview")
        assert response.status_code == 503
        assert "SECRET" not in response.text
        session.execute.assert_called_once()
        session.commit.assert_not_called()
    finally:
        app.dependency_overrides.pop(get_db_session, None)
