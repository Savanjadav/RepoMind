import os
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import ask_api, search_api
from app.database import create_database_engine, get_db_session
from app.main import app
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.relationship import Relationship
from app.models.repository import Repository


@pytest.fixture
def database() -> Iterator[tuple[Session, Repository, list[File], list[CodeUnit]]]:
    url = os.getenv("DATABASE_URL")
    if url is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(url)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:
                repo = Repository(
                    name="impact", source=f"https://example.com/{uuid4()}"
                )
                session.add(repo)
                session.flush()
                files = [
                    File(repository_id=repo.id, path=p)
                    for p in ("api.py", "service.py", "helper.py")
                ]
                session.add_all(files)
                session.flush()
                units = [
                    CodeUnit(
                        file_id=f.id,
                        kind="function",
                        symbol_name=s,
                        language="python",
                        start_line=10 * n + 1,
                        end_line=10 * n + 5,
                        content="PRIVATE SOURCE",
                    )
                    for n, (f, s) in enumerate(
                        zip(
                            files,
                            ("route_handler", "service_function", "helper"),
                            strict=True,
                        )
                    )
                ]
                session.add_all(units)
                session.flush()
                for index in range(2):
                    session.add(
                        Relationship(
                            repository_id=repo.id,
                            source_file_id=files[index].id,
                            target_file_id=files[index + 1].id,
                            relationship_type="imports",
                        )
                    )
                    session.add(
                        CodeRelationship(
                            repository_id=repo.id,
                            source_file_id=files[index].id,
                            target_file_id=files[index + 1].id,
                            source_code_unit_id=units[index].id,
                            target_code_unit_id=units[index + 1].id,
                            relationship_type="calls",
                        )
                    )
                session.flush()
                yield session, repo, files, units
        finally:
            transaction.rollback()
    engine.dispose()


@pytest.fixture
def client(
    database: tuple[Session, Repository, list[File], list[CodeUnit]],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    def get_session() -> Iterator[Session]:
        yield database[0]

    def no_provider(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Impact must never acquire a model/provider")

    monkeypatch.setattr(search_api, "get_embedding_provider", no_provider)
    monkeypatch.setattr(ask_api, "get_embedding_provider", no_provider)
    monkeypatch.setattr(ask_api, "get_reranking_provider", no_provider)
    monkeypatch.setattr(ask_api, "get_llm_provider", no_provider)
    app.dependency_overrides[get_db_session] = get_session
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("symbol_mode", [False, True])
def test_real_endpoint_returns_only_direct_dependents(
    client: TestClient,
    database: tuple[Session, Repository, list[File], list[CodeUnit]],
    symbol_mode: bool,
) -> None:
    _, repo, files, units = database
    params: dict[str, str] = {"path": "helper.py" if symbol_mode else "service.py"}
    if symbol_mode:
        params["symbol"] = "helper"
    response = client.get(f"/repositories/{repo.id}/impact", params=params)
    assert response.status_code == 200
    body = response.json()
    index = 1 if symbol_mode else 0
    target_index = index + 1

    def metadata(i: int) -> dict[str, Any]:
        return {
            "file_id": str(files[i].id),
            "path": files[i].path,
            "code_unit_id": str(units[i].id) if symbol_mode else None,
            "symbol_name": units[i].symbol_name if symbol_mode else None,
            "kind": "function" if symbol_mode else None,
            "start_line": units[i].start_line if symbol_mode else None,
            "end_line": units[i].end_line if symbol_mode else None,
        }

    assert body == {
        "repository_id": str(repo.id),
        "mode": "symbol" if symbol_mode else "file",
        "target": metadata(target_index),
        "items": [
            {
                **metadata(index),
                "relationship_type": "calls" if symbol_mode else "imports",
            }
        ],
        "limit": 20,
        "truncated": False,
        "analysis": "static_hint",
    }
    assert "PRIVATE SOURCE" not in response.text


@pytest.mark.parametrize("symbol", [None, "route_handler"])
def test_valid_target_without_incoming_hints(
    client: TestClient,
    database: tuple[Session, Repository, list[File], list[CodeUnit]],
    symbol: str | None,
) -> None:
    params = {"path": "api.py"}
    if symbol:
        params["symbol"] = symbol
    response = client.get(f"/repositories/{database[1].id}/impact", params=params)
    assert response.status_code == 200
    assert response.json()["items"] == [] and response.json()["truncated"] is False


@pytest.mark.parametrize("missing", ["repository", "file", "symbol"])
def test_missing_targets_are_404(
    client: TestClient,
    database: tuple[Session, Repository, list[File], list[CodeUnit]],
    missing: str,
) -> None:
    repo_id = uuid4() if missing == "repository" else database[1].id
    params = {"path": "absent.py" if missing == "file" else "api.py"}
    if missing == "symbol":
        params["symbol"] = "absent"
    assert (
        client.get(f"/repositories/{repo_id}/impact", params=params).status_code == 404
    )


def test_ambiguous_exact_symbol_is_409(
    client: TestClient, database: tuple[Session, Repository, list[File], list[CodeUnit]]
) -> None:
    session, repo, files, _ = database
    session.add(
        CodeUnit(
            file_id=files[0].id,
            kind="class",
            symbol_name="route_handler",
            language="python",
            start_line=99,
            end_line=101,
            content="SECRET",
        )
    )
    session.flush()
    response = client.get(
        f"/repositories/{repo.id}/impact",
        params={"path": "api.py", "symbol": "route_handler"},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "Symbol is ambiguous in the selected file"}


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"path": ""},
        {"path": "/api.py"},
        {"path": "a\\b"},
        {"path": "a//b"},
        {"path": "./a"},
        {"path": "../a"},
        {"path": "a\n.py"},
        {"path": "a\x7f.py"},
        {"path": "x" * 2001},
        {"path": "api.py", "symbol": " "},
        {"path": "api.py", "symbol": ""},
        {"path": "api.py", "symbol": "a\t"},
        {"path": "api.py", "symbol": "x" * 2001},
        *[{"path": "api.py", "limit": value} for value in ("0", "101", "true", "1.5")],
    ],
)
def test_input_validation(
    client: TestClient,
    database: tuple[Session, Repository, list[File], list[CodeUnit]],
    params: dict[str, str],
) -> None:
    assert (
        client.get(f"/repositories/{database[1].id}/impact", params=params).status_code
        == 422
    )


def test_invalid_repository_uuid(client: TestClient) -> None:
    assert (
        client.get(
            "/repositories/not-a-uuid/impact", params={"path": "api.py"}
        ).status_code
        == 422
    )


@pytest.mark.parametrize("limit", [1, 100])
def test_limit_boundaries(
    client: TestClient,
    database: tuple[Session, Repository, list[File], list[CodeUnit]],
    limit: int,
) -> None:
    response = client.get(
        f"/repositories/{database[1].id}/impact",
        params={"path": "service.py", "limit": limit},
    )
    assert response.status_code == 200 and response.json()["limit"] == limit


def test_unexpected_database_error_is_sanitized(
    client: TestClient,
    database: tuple[Session, Repository, list[File], list[CodeUnit]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, repo, _, _ = database
    repo_id = repo.id

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("secret SQL connection/password/source")

    monkeypatch.setattr(session, "scalar", fail)
    response = client.get(f"/repositories/{repo_id}/impact", params={"path": "api.py"})
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
