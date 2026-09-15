import os
from collections.abc import Iterator, Sequence
from typing import cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

import app.reranked_search as reranked_search_module
from app.code_parser import CodeUnitKind
from app.database import create_database_engine
from app.hybrid_search import HybridSearchResult
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.repository import Repository
from app.reranked_search import (
    MAX_RERANK_CANDIDATES,
    RERANK_CANDIDATE_MULTIPLIER,
    search_code_units_reranked,
)
from app.reranking_provider import RerankerUnavailableError
from app.retrieval_filters import RetrievalFilters

DATABASE_URL = os.getenv("DATABASE_URL")


class FakeReranker:
    def __init__(
        self,
        scores: Sequence[object] = (),
        error: Exception | None = None,
    ) -> None:
        self.scores = list(scores)
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_name(self) -> str:
        return "fake-reranker"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self.error is not None:
            raise self.error
        return cast(list[float], self.scores)


def _candidate(
    number: int,
    *,
    path: str | None = None,
    symbol_name: str | None = None,
    content: str | None = None,
) -> HybridSearchResult:
    return HybridSearchResult(
        code_unit_id=UUID(int=number),
        file_id=UUID(int=100 + number),
        path=path or f"src/candidate_{number}.py",
        kind=CodeUnitKind.FUNCTION,
        content=content or f"def candidate_{number}():\n    return {number}\n",
        language="python",
        start_line=number,
        end_line=number + 1,
        symbol_name=symbol_name,
        hybrid_score=1.0 / (60 + number),
        semantic_rank=number,
        lexical_rank=None,
        cosine_distance=float(number) / 10,
        lexical_score=None,
    )


def test_reranks_expanded_hybrid_candidates_and_preserves_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _candidate(1, symbol_name="first"),
        _candidate(2, symbol_name=None),
        _candidate(3, symbol_name="third"),
        _candidate(4, symbol_name="fourth"),
    ]
    hybrid_calls: list[tuple[str, Sequence[float], int, RetrievalFilters | None]] = []

    def fake_hybrid(
        session: Session,
        *,
        repository_id: UUID,
        query: str,
        query_vector: Sequence[float],
        limit: int,
        filters: RetrievalFilters | None,
    ) -> list[HybridSearchResult]:
        hybrid_calls.append((query, query_vector, limit, filters))
        return candidates

    monkeypatch.setattr(
        reranked_search_module,
        "search_code_units_hybrid",
        fake_hybrid,
    )
    filters = RetrievalFilters(language="python")
    reranker = FakeReranker([0.1, 0.2, 0.9, 0.8])
    query_vector = [1.0]

    results = search_code_units_reranked(
        Mock(spec=Session),
        repository_id=uuid4(),
        query="  original query\n",
        query_vector=query_vector,
        reranker=reranker,
        limit=2,
        filters=filters,
    )

    assert hybrid_calls == [
        ("  original query\n", query_vector, 2 * RERANK_CANDIDATE_MULTIPLIER, filters)
    ]
    assert reranker.calls == [
        (
            "  original query\n",
            [
                "Path: src/candidate_1.py\nSymbol: first\n\nContent:\n"
                "def candidate_1():\n    return 1\n",
                "Path: src/candidate_2.py\n\nContent:\n"
                "def candidate_2():\n    return 2\n",
                "Path: src/candidate_3.py\nSymbol: third\n\nContent:\n"
                "def candidate_3():\n    return 3\n",
                "Path: src/candidate_4.py\nSymbol: fourth\n\nContent:\n"
                "def candidate_4():\n    return 4\n",
            ],
        )
    ]
    assert [result.code_unit_id for result in results] == [UUID(int=3), UUID(int=4)]
    assert [result.rerank_score for result in results] == [0.9, 0.8]
    assert [result.original_hybrid_rank for result in results] == [3, 4]
    assert results[0].hybrid_score == candidates[2].hybrid_score
    assert results[0].semantic_rank == candidates[2].semantic_rank
    assert results[0].cosine_distance == candidates[2].cosine_distance


def test_equal_scores_preserve_original_hybrid_order_and_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    monkeypatch.setattr(
        reranked_search_module,
        "search_code_units_hybrid",
        lambda *args, **kwargs: candidates,
    )
    reranker = FakeReranker([0.5, 0.5, 0.4])

    first = search_code_units_reranked(
        Mock(spec=Session),
        repository_id=uuid4(),
        query="query",
        query_vector=[1.0],
        reranker=reranker,
    )
    second = search_code_units_reranked(
        Mock(spec=Session),
        repository_id=uuid4(),
        query="query",
        query_vector=[1.0],
        reranker=reranker,
    )

    assert [result.code_unit_id for result in first] == [
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
    ]
    assert first == second


def test_empty_hybrid_results_do_not_call_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reranked_search_module,
        "search_code_units_hybrid",
        lambda *args, **kwargs: [],
    )
    reranker = FakeReranker([1.0])

    results = search_code_units_reranked(
        Mock(spec=Session),
        repository_id=uuid4(),
        query="query",
        query_vector=[1.0],
        reranker=reranker,
    )

    assert results == []
    assert reranker.calls == []


def test_candidate_limit_is_capped_at_one_hundred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits: list[int] = []

    def fake_hybrid(*args: object, **kwargs: object) -> list[HybridSearchResult]:
        limits.append(cast(int, kwargs["limit"]))
        return []

    monkeypatch.setattr(
        reranked_search_module,
        "search_code_units_hybrid",
        fake_hybrid,
    )

    search_code_units_reranked(
        Mock(spec=Session),
        repository_id=uuid4(),
        query="query",
        query_vector=[1.0],
        reranker=FakeReranker(),
        limit=100,
    )

    assert limits == [MAX_RERANK_CANDIDATES]


@pytest.mark.parametrize("limit", [True, False, 0, -1, 101, "10", 10.0, None])
def test_invalid_limit_fails_before_hybrid_search(
    monkeypatch: pytest.MonkeyPatch,
    limit: object,
) -> None:
    hybrid = Mock()
    monkeypatch.setattr(reranked_search_module, "search_code_units_hybrid", hybrid)

    with pytest.raises(ValueError, match="between 1 and 100"):
        search_code_units_reranked(
            Mock(spec=Session),
            repository_id=uuid4(),
            query="query",
            query_vector=[1.0],
            reranker=FakeReranker(),
            limit=cast(int, limit),
        )
    hybrid.assert_not_called()


@pytest.mark.parametrize(
    "scores",
    [[], [1.0, 2.0], [True], [float("nan")], [float("inf")], [float("-inf")]],
)
def test_malformed_scores_surface(
    monkeypatch: pytest.MonkeyPatch,
    scores: list[object],
) -> None:
    monkeypatch.setattr(
        reranked_search_module,
        "search_code_units_hybrid",
        lambda *args, **kwargs: [_candidate(1)],
    )

    with pytest.raises(RuntimeError):
        search_code_units_reranked(
            Mock(spec=Session),
            repository_id=uuid4(),
            query="query",
            query_vector=[1.0],
            reranker=FakeReranker(scores),
        )


def test_unavailable_reranker_falls_back_without_fabricating_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    monkeypatch.setattr(
        reranked_search_module,
        "search_code_units_hybrid",
        lambda *args, **kwargs: candidates,
    )

    results = search_code_units_reranked(
        Mock(spec=Session),
        repository_id=uuid4(),
        query="query",
        query_vector=[1.0],
        reranker=FakeReranker(error=RerankerUnavailableError("not cached")),
        limit=2,
    )

    assert [result.code_unit_id for result in results] == [UUID(int=1), UUID(int=2)]
    assert [result.original_hybrid_rank for result in results] == [1, 2]
    assert [result.rerank_score for result in results] == [None, None]


@pytest.mark.parametrize(
    "error",
    [ValueError("bad value"), TypeError("bad type"), AssertionError("defect")],
)
def test_unrelated_reranker_exceptions_propagate(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(
        reranked_search_module,
        "search_code_units_hybrid",
        lambda *args, **kwargs: [_candidate(1)],
    )

    with pytest.raises(type(error), match=str(error)):
        search_code_units_reranked(
            Mock(spec=Session),
            repository_id=uuid4(),
            query="query",
            query_vector=[1.0],
            reranker=FakeReranker(error=error),
        )


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


def _vector(first: float, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * (EMBEDDING_DIMENSION - 2))]


def _repository(session: Session, name: str) -> Repository:
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


def _unit(
    session: Session,
    file: File,
    *,
    language: str,
    symbol_name: str,
) -> CodeUnit:
    unit = CodeUnit(
        file_id=file.id,
        kind=CodeUnitKind.FUNCTION.value,
        content=f"def {symbol_name}():\n    return True\n",
        language=language,
        start_line=1,
        end_line=2,
        symbol_name=symbol_name,
        embedding=_vector(1.0),
    )
    session.add(unit)
    session.flush()
    return unit


def test_real_hybrid_candidates_preserve_filters_and_repository_isolation(
    database_session: Session,
) -> None:
    target_repository = _repository(database_session, "Target")
    target_file = _file(database_session, target_repository, "src/target.py")
    eligible = _unit(
        database_session,
        target_file,
        language="python",
        symbol_name="needle",
    )
    excluded_by_filter = _unit(
        database_session,
        target_file,
        language="typescript",
        symbol_name="needle_ts",
    )
    other_repository = _repository(database_session, "Other")
    other_file = _file(database_session, other_repository, "src/other.py")
    other = _unit(
        database_session,
        other_file,
        language="python",
        symbol_name="needle",
    )

    results = search_code_units_reranked(
        database_session,
        repository_id=target_repository.id,
        query="needle",
        query_vector=_vector(1.0),
        reranker=FakeReranker([1.0]),
        filters=RetrievalFilters(language="python"),
    )

    assert [result.code_unit_id for result in results] == [eligible.id]
    assert excluded_by_filter.id not in {result.code_unit_id for result in results}
    assert other.id not in {result.code_unit_id for result in results}
