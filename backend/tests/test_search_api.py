import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import sqrt
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.database import create_database_engine, get_db_session
from app.embedding_provider import EmbeddingProvider
from app.main import app
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.repository import Repository
from app.search_api import get_embedding_provider

DATABASE_URL = os.getenv("DATABASE_URL")


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        dimension: int = EMBEDDING_DIMENSION,
        output: list[list[float]] | None = None,
        failure: RuntimeError | None = None,
    ) -> None:
        self._dimension = dimension
        self.output = output if output is not None else [_vector(1.0, 0.0)]
        self.failure = failure
        self.calls: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        if self.failure is not None:
            raise self.failure
        return self.output


def _use_provider(provider: EmbeddingProvider) -> EmbeddingProvider:
    return provider


@dataclass(frozen=True, slots=True)
class ApiTestContext:
    client: TestClient
    session: Session
    provider: FakeEmbeddingProvider


@pytest.fixture
def api_context() -> Iterator[ApiTestContext]:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")

    engine = create_database_engine(DATABASE_URL)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:
                provider = FakeEmbeddingProvider()

                def override_db_session() -> Iterator[Session]:
                    yield session

                def override_embedding_provider() -> EmbeddingProvider:
                    return provider

                app.dependency_overrides[get_db_session] = override_db_session
                app.dependency_overrides[get_embedding_provider] = (
                    override_embedding_provider
                )
                with TestClient(app, raise_server_exceptions=False) as client:
                    yield ApiTestContext(client, session, provider)
        finally:
            app.dependency_overrides.clear()
            transaction.rollback()
    engine.dispose()


def _vector(first: float = 0.0, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * (EMBEDDING_DIMENSION - 2))]


def _repository(session: Session, name: str = "Search API repository") -> Repository:
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
    embedding: list[float],
    content: str = "def authenticate():\n    return 'verified'\n",
    start_line: int = 1,
    end_line: int = 2,
    symbol_name: str = "authenticate",
) -> CodeUnit:
    unit = CodeUnit(
        file_id=file.id,
        kind=CodeUnitKind.FUNCTION.value,
        content=content,
        language="python",
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
        embedding=embedding,
    )
    session.add(unit)
    session.flush()
    return unit


def _search(
    context: ApiTestContext,
    repository_id: UUID,
    *,
    q: str = "how does authentication work?",
    limit: int | None = None,
):
    params: dict[str, str | int] = {"repository_id": str(repository_id), "q": q}
    if limit is not None:
        params["limit"] = limit
    return context.client.get("/search", params=params)


def test_search_composes_exact_query_with_real_semantic_retrieval(
    api_context: ApiTestContext,
) -> None:
    repository = _repository(api_context.session)
    file = _file(api_context.session, repository, "src/认证.py")
    content = "def café_auth():\n    return '✓'\n"
    unit = _code_unit(
        api_context.session,
        file,
        embedding=_vector(1.0, 0.0),
        content=content,
        start_line=12,
        end_line=13,
        symbol_name="café_auth",
    )
    count_before = api_context.session.scalar(
        select(func.count()).select_from(CodeUnit)
    )
    query = "  how does café authentication work?\n"

    response = _search(api_context, repository.id, q=query)

    assert response.status_code == 200
    assert api_context.provider.calls == [(query,)]
    assert response.json() == {
        "items": [
            {
                "code_unit_id": str(unit.id),
                "file_id": str(file.id),
                "path": "src/认证.py",
                "kind": "function",
                "content": content,
                "language": "python",
                "start_line": 12,
                "end_line": 13,
                "symbol_name": "café_auth",
                "cosine_distance": pytest.approx(0.0),
            }
        ],
        "limit": 10,
    }
    assert (
        api_context.session.scalar(select(func.count()).select_from(CodeUnit))
        == count_before
    )


def test_search_preserves_semantic_order_and_repository_scope(
    api_context: ApiTestContext,
) -> None:
    repository = _repository(api_context.session)
    file = _file(api_context.session, repository, "src/order.py")
    vectors = [
        ("opposite", _vector(-1.0, 0.0)),
        ("orthogonal", _vector(0.0, 1.0)),
        ("close", _vector(1.0, 1.0)),
        ("exact", _vector(1.0, 0.0)),
    ]
    units = {
        name: _code_unit(
            api_context.session,
            file,
            embedding=vector,
            symbol_name=name,
        )
        for name, vector in vectors
    }
    other_repository = _repository(api_context.session, "Other repository")
    other_file = _file(api_context.session, other_repository, "src/private.py")
    _code_unit(
        api_context.session,
        other_file,
        embedding=_vector(1.0, 0.0),
        symbol_name="private",
    )

    response = _search(api_context, repository.id)

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["code_unit_id"] for item in items] == [
        str(units[name].id) for name in ["exact", "close", "orthogonal", "opposite"]
    ]
    assert [item["cosine_distance"] for item in items] == pytest.approx(
        [0.0, 1.0 - 1.0 / sqrt(2.0), 1.0, 2.0]
    )
    assert all(item["symbol_name"] != "private" for item in items)


def test_default_and_explicit_limits_are_applied(api_context: ApiTestContext) -> None:
    repository = _repository(api_context.session)
    file = _file(api_context.session, repository, "src/limit.py")
    for index in range(11):
        _code_unit(
            api_context.session,
            file,
            embedding=_vector(1.0, float(index + 1)),
            start_line=index + 1,
            end_line=index + 1,
            symbol_name=f"unit_{index}",
        )

    default_response = _search(api_context, repository.id)
    explicit_response = _search(api_context, repository.id, limit=1)

    assert default_response.status_code == 200
    assert default_response.json()["limit"] == 10
    assert len(default_response.json()["items"]) == 10
    assert explicit_response.status_code == 200
    assert explicit_response.json()["limit"] == 1
    assert [item["symbol_name"] for item in explicit_response.json()["items"]] == [
        "unit_0"
    ]


@pytest.mark.parametrize("limit", [1, 100])
def test_limit_boundaries_are_accepted(
    api_context: ApiTestContext,
    limit: int,
) -> None:
    repository = _repository(api_context.session)

    response = _search(api_context, repository.id, limit=limit)

    assert response.status_code == 200
    assert response.json() == {"items": [], "limit": limit}


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/search", {"repository_id": "not-a-uuid", "q": "valid"}),
        ("/search", {"repository_id": str(uuid4())}),
        ("/search", {"repository_id": str(uuid4()), "q": ""}),
        ("/search", {"repository_id": str(uuid4()), "q": " \t\n "}),
        ("/search", {"repository_id": str(uuid4()), "q": "x" * 2001}),
        ("/search", {"repository_id": str(uuid4()), "q": "valid", "limit": 0}),
        ("/search", {"repository_id": str(uuid4()), "q": "valid", "limit": -1}),
        ("/search", {"repository_id": str(uuid4()), "q": "valid", "limit": 101}),
    ],
)
def test_invalid_requests_return_422_without_embedding(
    api_context: ApiTestContext,
    path: str,
    params: dict[str, str | int],
) -> None:
    response = api_context.client.get(path, params=params)

    assert response.status_code == 422
    assert api_context.provider.calls == []


def test_missing_repository_is_404(api_context: ApiTestContext) -> None:
    response = _search(api_context, uuid4())

    assert response.status_code == 404
    assert response.json() == {"detail": "Repository not found"}


def test_existing_repository_without_results_is_200(
    api_context: ApiTestContext,
) -> None:
    repository = _repository(api_context.session)

    response = _search(api_context, repository.id)

    assert response.status_code == 200
    assert response.json() == {"items": [], "limit": 10}


def test_provider_dimension_mismatch_is_server_error_without_embedding(
    api_context: ApiTestContext,
) -> None:
    api_context.provider._dimension = EMBEDDING_DIMENSION + 1
    repository = _repository(api_context.session)

    response = _search(api_context, repository.id)

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert api_context.provider.calls == []


@pytest.mark.parametrize(
    "output",
    [[], [_vector(1.0, 0.0), _vector(0.0, 1.0)]],
)
def test_malformed_provider_output_is_server_error(
    api_context: ApiTestContext,
    output: list[list[float]],
) -> None:
    api_context.provider.output = output
    repository = _repository(api_context.session)

    response = _search(api_context, repository.id)

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert len(api_context.provider.calls) == 1


def test_provider_runtime_failure_is_server_error(
    api_context: ApiTestContext,
) -> None:
    api_context.provider.failure = RuntimeError("embedding failed")
    repository = _repository(api_context.session)

    response = _search(api_context, repository.id)

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert len(api_context.provider.calls) == 1


def test_fake_provider_structurally_satisfies_contract() -> None:
    assert _use_provider(FakeEmbeddingProvider()).dimension == EMBEDDING_DIMENSION
