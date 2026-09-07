import os
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.database import create_database_engine, get_db_session
from app.main import app
from app.models import CodeUnit, File, Repository

DATABASE_URL = os.getenv("DATABASE_URL")


@dataclass(frozen=True, slots=True)
class ApiTestContext:
    client: TestClient
    session: Session


@pytest.fixture
def api_context() -> Iterator[ApiTestContext]:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")

    engine = create_database_engine(DATABASE_URL)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:

                def override_db_session() -> Iterator[Session]:
                    yield session

                app.dependency_overrides[get_db_session] = override_db_session
                with TestClient(app) as client:
                    yield ApiTestContext(client=client, session=session)
        finally:
            app.dependency_overrides.clear()
            transaction.rollback()
    engine.dispose()


def _repository(session: Session, name: str = "Repository") -> Repository:
    repository = Repository(
        name=name,
        source=f"https://example.com/{uuid4()}.git",
    )
    session.add(repository)
    session.flush()
    return repository


def _file(session: Session, repository: Repository, path: str) -> File:
    file = File(repository_id=repository.id, path=path)
    session.add(file)
    session.flush()
    return file


def _code_unit(
    session: Session,
    file: File,
    *,
    kind: CodeUnitKind = CodeUnitKind.FUNCTION,
    content: str = "untrusted source content",
    language: str = "python",
    start_line: int = 1,
    end_line: int = 1,
    symbol_name: str | None = "main",
) -> CodeUnit:
    code_unit = CodeUnit(
        file_id=file.id,
        kind=kind.value,
        content=content,
        language=language,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
    )
    session.add(code_unit)
    session.flush()
    return code_unit


def _endpoint(repository_id: UUID) -> str:
    return f"/repositories/{repository_id}/code-units"


def test_lists_one_code_unit_as_bounded_metadata_without_mutation(
    api_context: ApiTestContext,
) -> None:
    repository = _repository(api_context.session)
    file = _file(api_context.session, repository, "missing/文档.py")
    code_unit = _code_unit(
        api_context.session,
        file,
        content='dangerous = "✓"',
        symbol_name=None,
    )
    count_before = api_context.session.scalar(
        select(func.count()).select_from(CodeUnit)
    )

    response = api_context.client.get(_endpoint(repository.id))

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "id": str(code_unit.id),
                "file_id": str(file.id),
                "path": "missing/文档.py",
                "kind": "function",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "symbol_name": None,
            }
        ],
        "limit": 100,
        "offset": 0,
    }
    assert set(response.json()["items"][0]) == {
        "id",
        "file_id",
        "path",
        "kind",
        "language",
        "start_line",
        "end_line",
        "symbol_name",
    }
    assert "content" not in response.json()["items"][0]
    assert "total" not in response.json()
    assert (
        api_context.session.scalar(select(func.count()).select_from(CodeUnit))
        == count_before
    )


def test_existing_empty_repository_and_missing_repository_are_distinct(
    api_context: ApiTestContext,
) -> None:
    repository = _repository(api_context.session)

    empty_response = api_context.client.get(_endpoint(repository.id))
    missing_response = api_context.client.get(_endpoint(uuid4()))

    assert empty_response.status_code == 200
    assert empty_response.json() == {"items": [], "limit": 100, "offset": 0}
    assert missing_response.status_code == 404
    assert missing_response.json() == {"detail": "Repository not found"}


def test_orders_across_files_and_excludes_other_repositories(
    api_context: ApiTestContext,
) -> None:
    repository = _repository(api_context.session)
    a_file = _file(api_context.session, repository, "a.py")
    b_file = _file(api_context.session, repository, "b.py")
    other_repository = _repository(api_context.session, "Other repository")
    other_file = _file(api_context.session, other_repository, "00-private.py")

    _code_unit(
        api_context.session,
        a_file,
        kind=CodeUnitKind.IMPORT,
        start_line=5,
        end_line=5,
        symbol_name=None,
    )
    _code_unit(
        api_context.session,
        a_file,
        kind=CodeUnitKind.CLASS,
        start_line=1,
        end_line=20,
        symbol_name="Outer",
    )
    _code_unit(
        api_context.session,
        a_file,
        kind=CodeUnitKind.FUNCTION,
        start_line=1,
        end_line=10,
        symbol_name="function_tie",
    )
    _code_unit(
        api_context.session,
        a_file,
        kind=CodeUnitKind.IMPORT,
        start_line=1,
        end_line=10,
        symbol_name=None,
    )
    _code_unit(
        api_context.session,
        a_file,
        kind=CodeUnitKind.CLASS,
        start_line=1,
        end_line=10,
        symbol_name="class_tie",
    )
    _code_unit(
        api_context.session,
        b_file,
        kind=CodeUnitKind.FUNCTION,
        symbol_name="b_function",
    )
    _code_unit(
        api_context.session,
        other_file,
        kind=CodeUnitKind.CLASS,
        symbol_name="Private",
    )

    first_response = api_context.client.get(_endpoint(repository.id))
    second_response = api_context.client.get(_endpoint(repository.id))

    assert first_response.status_code == 200
    assert first_response.json() == second_response.json()
    items = first_response.json()["items"]
    assert [
        (item["path"], item["start_line"], item["end_line"], item["kind"])
        for item in items
    ] == [
        ("a.py", 1, 20, "class"),
        ("a.py", 1, 10, "class"),
        ("a.py", 1, 10, "function"),
        ("a.py", 1, 10, "import"),
        ("a.py", 5, 5, "import"),
        ("b.py", 1, 1, "function"),
    ]
    assert all(item["symbol_name"] != "Private" for item in items)


def test_pagination_is_applied_after_deterministic_ordering(
    api_context: ApiTestContext,
) -> None:
    repository = _repository(api_context.session)
    file = _file(api_context.session, repository, "src/main.py")
    for line in [4, 1, 3, 2]:
        _code_unit(
            api_context.session,
            file,
            start_line=line,
            end_line=line,
            symbol_name=f"line_{line}",
        )

    full = api_context.client.get(_endpoint(repository.id)).json()["items"]
    page = api_context.client.get(
        _endpoint(repository.id),
        params={"limit": 2, "offset": 1},
    )

    assert page.status_code == 200
    assert page.json() == {"items": full[1:3], "limit": 2, "offset": 1}


@pytest.mark.parametrize("kind", list(CodeUnitKind))
def test_kind_filter_returns_only_requested_kind_and_repository(
    api_context: ApiTestContext,
    kind: CodeUnitKind,
) -> None:
    repository = _repository(api_context.session)
    file = _file(api_context.session, repository, "src/main.py")
    other_repository = _repository(api_context.session, "Other repository")
    other_file = _file(api_context.session, other_repository, "src/main.py")
    for unit_kind in CodeUnitKind:
        _code_unit(
            api_context.session,
            file,
            kind=unit_kind,
            symbol_name=unit_kind.value,
        )
    _code_unit(
        api_context.session,
        other_file,
        kind=kind,
        symbol_name="other repository",
    )

    response = api_context.client.get(
        _endpoint(repository.id),
        params={"kind": kind.value},
    )

    assert response.status_code == 200
    assert [item["kind"] for item in response.json()["items"]] == [kind.value]
    assert [item["symbol_name"] for item in response.json()["items"]] == [kind.value]


def test_valid_kind_with_no_matches_returns_empty_items(
    api_context: ApiTestContext,
) -> None:
    repository = _repository(api_context.session)
    file = _file(api_context.session, repository, "src/main.py")
    _code_unit(api_context.session, file, kind=CodeUnitKind.CLASS)

    response = api_context.client.get(
        _endpoint(repository.id),
        params={"kind": "config"},
    )

    assert response.status_code == 200
    assert response.json()["items"] == []


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/repositories/not-a-uuid/code-units", {}),
        (None, {"kind": "method"}),
        (None, {"limit": 0}),
        (None, {"limit": 501}),
        (None, {"offset": -1}),
    ],
)
def test_invalid_request_values_return_422(
    api_context: ApiTestContext,
    path: str | None,
    params: dict[str, str | int],
) -> None:
    repository = _repository(api_context.session)

    response = api_context.client.get(path or _endpoint(repository.id), params=params)

    assert response.status_code == 422


def test_pagination_boundary_values_are_accepted(
    api_context: ApiTestContext,
) -> None:
    repository = _repository(api_context.session)

    minimum = api_context.client.get(
        _endpoint(repository.id),
        params={"limit": 1},
    )
    maximum = api_context.client.get(
        _endpoint(repository.id),
        params={"limit": 500},
    )

    assert minimum.status_code == 200
    assert minimum.json()["limit"] == 1
    assert maximum.status_code == 200
    assert maximum.json()["limit"] == 500
