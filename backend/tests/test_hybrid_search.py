import os
from collections.abc import Iterator, Sequence
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

import app.hybrid_search as hybrid_search_module
from app.code_parser import CodeUnitKind
from app.database import create_database_engine
from app.hybrid_search import (
    CANDIDATE_MULTIPLIER,
    RRF_K,
    HybridSearchResult,
    search_code_units_hybrid,
)
from app.lexical_search import LexicalSearchResult, search_code_units_lexically
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.repository import Repository
from app.semantic_search import SemanticSearchResult, search_code_units_semantically

DATABASE_URL = os.getenv("DATABASE_URL")


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


def _vector(first: float = 0.0, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * (EMBEDDING_DIMENSION - 2))]


def _repository(session: Session, name: str = "Hybrid repository") -> Repository:
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
    embedding: list[float] | None,
    code_unit_id: UUID | None = None,
    kind: CodeUnitKind = CodeUnitKind.FUNCTION,
    content: str = "unrelated conceptual content",
    language: str = "python",
    start_line: int = 1,
    end_line: int | None = None,
    symbol_name: str | None = None,
) -> CodeUnit:
    unit = CodeUnit(
        id=code_unit_id or uuid4(),
        file_id=file.id,
        kind=kind.value,
        content=content,
        language=language,
        start_line=start_line,
        end_line=end_line if end_line is not None else start_line,
        symbol_name=symbol_name,
        embedding=embedding,
    )
    session.add(unit)
    session.flush()
    return unit


def test_hybrid_preserves_semantic_lexical_and_both_signal_provenance(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    semantic_file = _file(database_session, repository, "a_semantic.py")
    lexical_file = _file(database_session, repository, "b_lexical.py")
    both_file = _file(database_session, repository, "c_both.py")
    semantic_only = _code_unit(
        database_session,
        semantic_file,
        embedding=_vector(1.0, 0.0),
        content="credentials are loaded by the security subsystem",
        symbol_name="load_credentials",
    )
    lexical_only = _code_unit(
        database_session,
        lexical_file,
        embedding=None,
        content="exact environment setting",
        symbol_name="JWT_SECRET",
    )
    both_content = "def JWT_SECRET_handler():\n    return '✓'\n"
    both = _code_unit(
        database_session,
        both_file,
        embedding=_vector(1.0, 0.2),
        content=both_content,
        language="python",
        start_line=10,
        end_line=11,
        symbol_name="JWT_SECRET_handler",
    )
    other_repository = _repository(database_session, "Other")
    other_file = _file(database_session, other_repository, "other.py")
    other = _code_unit(
        database_session,
        other_file,
        embedding=_vector(1.0, 0.0),
        symbol_name="JWT_SECRET",
    )

    semantic_results = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(1.0, 0.0),
        limit=30,
    )
    lexical_results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="JWT_SECRET",
        limit=30,
    )
    assert [result.code_unit_id for result in semantic_results] == [
        semantic_only.id,
        both.id,
    ]
    assert [result.code_unit_id for result in lexical_results] == [
        lexical_only.id,
        both.id,
    ]

    results = search_code_units_hybrid(
        database_session,
        repository_id=repository.id,
        query="JWT_SECRET",
        query_vector=_vector(1.0, 0.0),
    )

    assert [result.code_unit_id for result in results] == [
        both.id,
        semantic_only.id,
        lexical_only.id,
    ]
    assert len([result for result in results if result.code_unit_id == both.id]) == 1
    by_id = {result.code_unit_id: result for result in results}
    assert by_id[semantic_only.id].semantic_rank == 1
    assert by_id[semantic_only.id].lexical_rank is None
    assert by_id[semantic_only.id].cosine_distance == pytest.approx(0.0)
    assert by_id[semantic_only.id].lexical_score is None
    assert by_id[lexical_only.id].semantic_rank is None
    assert by_id[lexical_only.id].lexical_rank == 1
    assert by_id[lexical_only.id].cosine_distance is None
    assert by_id[lexical_only.id].lexical_score == 9
    assert by_id[both.id] == HybridSearchResult(
        code_unit_id=both.id,
        file_id=both_file.id,
        path="c_both.py",
        kind=CodeUnitKind.FUNCTION,
        content=both_content,
        language="python",
        start_line=10,
        end_line=11,
        symbol_name="JWT_SECRET_handler",
        hybrid_score=by_id[both.id].hybrid_score,
        semantic_rank=2,
        lexical_rank=2,
        cosine_distance=by_id[both.id].cosine_distance,
        lexical_score=7,
    )
    assert by_id[both.id].hybrid_score == pytest.approx(2.0 / (RRF_K + 2))
    assert by_id[both.id].cosine_distance == pytest.approx(
        semantic_results[1].cosine_distance
    )
    assert by_id[both.id].hybrid_score > by_id[semantic_only.id].hybrid_score
    assert by_id[both.id].hybrid_score > by_id[lexical_only.id].hybrid_score
    assert other.id not in by_id


def test_candidate_depth_recovers_shared_rank_three_before_final_limit(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "candidates.py")
    semantic_first = _code_unit(
        database_session,
        file,
        embedding=_vector(1.0, 0.0),
        start_line=1,
        symbol_name="concept_a",
    )
    semantic_second = _code_unit(
        database_session,
        file,
        embedding=_vector(1.0, 0.1),
        start_line=2,
        symbol_name="concept_b",
    )
    shared = _code_unit(
        database_session,
        file,
        embedding=_vector(1.0, 0.2),
        content="needle is referenced here",
        start_line=3,
        symbol_name="shared_concept",
    )
    lexical_first = _code_unit(
        database_session,
        file,
        embedding=None,
        start_line=4,
        symbol_name="needle",
    )
    lexical_second = _code_unit(
        database_session,
        file,
        embedding=None,
        start_line=5,
        symbol_name="needle_helper",
    )

    semantic_top_two = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(1.0, 0.0),
        limit=2,
    )
    lexical_top_two = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="needle",
        limit=2,
    )
    semantic_six = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(1.0, 0.0),
        limit=6,
    )
    lexical_six = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="needle",
        limit=6,
    )
    assert [result.code_unit_id for result in semantic_top_two] == [
        semantic_first.id,
        semantic_second.id,
    ]
    assert [result.code_unit_id for result in lexical_top_two] == [
        lexical_first.id,
        lexical_second.id,
    ]
    assert [result.code_unit_id for result in semantic_six].index(shared.id) + 1 == 3
    assert [result.code_unit_id for result in lexical_six].index(shared.id) + 1 == 3

    results = search_code_units_hybrid(
        database_session,
        repository_id=repository.id,
        query="needle",
        query_vector=_vector(1.0, 0.0),
        limit=2,
    )

    assert results[0].code_unit_id == shared.id
    assert results[0].semantic_rank == 3
    assert results[0].lexical_rank == 3
    assert results[0].hybrid_score == pytest.approx(2.0 / (RRF_K + 3))
    assert len(results) == 2


def _semantic_result(
    code_unit_id: UUID,
    *,
    path: str,
    start_line: int,
    end_line: int,
    kind: CodeUnitKind,
) -> SemanticSearchResult:
    return SemanticSearchResult(
        code_unit_id=code_unit_id,
        file_id=UUID(int=1000),
        path=path,
        kind=kind,
        content="tie",
        language="python",
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
        cosine_distance=0.5,
    )


def _lexical_result(result: SemanticSearchResult) -> LexicalSearchResult:
    return LexicalSearchResult(
        code_unit_id=result.code_unit_id,
        file_id=result.file_id,
        path=result.path,
        kind=result.kind,
        content=result.content,
        language=result.language,
        start_line=result.start_line,
        end_line=result.end_line,
        symbol_name=result.symbol_name,
        lexical_score=3,
    )


def test_equal_rrf_scores_use_only_deterministic_metadata_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specifications = [
        (UUID(int=1), "z.py", 1, 1, CodeUnitKind.FUNCTION),
        (UUID(int=2), "a.py", 1, 1, CodeUnitKind.FUNCTION),
        (UUID(int=3), "c.py", 2, 2, CodeUnitKind.FUNCTION),
        (UUID(int=4), "c.py", 1, 1, CodeUnitKind.FUNCTION),
        (UUID(int=5), "d.py", 1, 5, CodeUnitKind.FUNCTION),
        (UUID(int=6), "d.py", 1, 10, CodeUnitKind.FUNCTION),
        (UUID(int=7), "e.py", 1, 1, CodeUnitKind.FUNCTION),
        (UUID(int=8), "e.py", 1, 1, CodeUnitKind.CLASS),
        (UUID(int=9), "f.py", 1, 1, CodeUnitKind.FUNCTION),
        (UUID(int=10), "f.py", 1, 1, CodeUnitKind.FUNCTION),
    ]
    semantic_results = [
        _semantic_result(
            code_unit_id,
            path=path,
            start_line=start_line,
            end_line=end_line,
            kind=kind,
        )
        for code_unit_id, path, start_line, end_line, kind in specifications
    ]
    lexical_results = [
        _lexical_result(semantic_results[index])
        for index in [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]
    ]

    def fake_semantic_search(
        session: Session,
        *,
        repository_id: UUID,
        query_vector: Sequence[float],
        limit: int = 10,
    ) -> list[SemanticSearchResult]:
        return semantic_results

    def fake_lexical_search(
        session: Session,
        *,
        repository_id: UUID,
        query: str,
        limit: int = 10,
    ) -> list[LexicalSearchResult]:
        return lexical_results

    monkeypatch.setattr(
        hybrid_search_module,
        "search_code_units_semantically",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        hybrid_search_module,
        "search_code_units_lexically",
        fake_lexical_search,
    )
    expected = [UUID(int=value) for value in [2, 1, 4, 3, 6, 5, 8, 7, 9, 10]]

    first = search_code_units_hybrid(
        cast(Session, object()),
        repository_id=uuid4(),
        query="tie",
        query_vector=[1.0],
    )
    second = search_code_units_hybrid(
        cast(Session, object()),
        repository_id=uuid4(),
        query="tie",
        query_vector=[1.0],
    )

    assert [result.code_unit_id for result in first] == expected
    assert [result.code_unit_id for result in second] == expected


def test_candidate_limits_are_bounded_and_both_retrievers_called_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_semantic_search(
        session: Session,
        *,
        repository_id: UUID,
        query_vector: Sequence[float],
        limit: int = 10,
    ) -> list[SemanticSearchResult]:
        calls.append(("semantic", limit))
        return []

    def fake_lexical_search(
        session: Session,
        *,
        repository_id: UUID,
        query: str,
        limit: int = 10,
    ) -> list[LexicalSearchResult]:
        calls.append(("lexical", limit))
        return []

    monkeypatch.setattr(
        hybrid_search_module,
        "search_code_units_semantically",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        hybrid_search_module,
        "search_code_units_lexically",
        fake_lexical_search,
    )
    session = cast(Session, object())
    repository_id = uuid4()

    search_code_units_hybrid(
        session,
        repository_id=repository_id,
        query="query",
        query_vector=[1.0],
    )
    search_code_units_hybrid(
        session,
        repository_id=repository_id,
        query="query",
        query_vector=[1.0],
        limit=1,
    )
    search_code_units_hybrid(
        session,
        repository_id=repository_id,
        query="query",
        query_vector=[1.0],
        limit=34,
    )
    search_code_units_hybrid(
        session,
        repository_id=repository_id,
        query="query",
        query_vector=[1.0],
        limit=100,
    )

    assert CANDIDATE_MULTIPLIER == 3
    assert calls == [
        ("semantic", 30),
        ("lexical", 30),
        ("semantic", 3),
        ("lexical", 3),
        ("semantic", 100),
        ("lexical", 100),
        ("semantic", 100),
        ("lexical", 100),
    ]


@pytest.mark.parametrize(
    "limit",
    [
        cast(int, True),
        cast(int, False),
        0,
        -1,
        101,
        cast(int, "10"),
        cast(int, 10.0),
        cast(int, None),
    ],
)
def test_invalid_limit_fails_before_retrievers(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
    def unexpected_call(*args: object, **kwargs: object) -> list[object]:
        raise AssertionError("retriever must not be called")

    monkeypatch.setattr(
        hybrid_search_module,
        "search_code_units_semantically",
        unexpected_call,
    )
    monkeypatch.setattr(
        hybrid_search_module,
        "search_code_units_lexically",
        unexpected_call,
    )

    with pytest.raises(ValueError, match="Limit must be an integer between 1 and 100"):
        search_code_units_hybrid(
            cast(Session, object()),
            repository_id=uuid4(),
            query="query",
            query_vector=[1.0],
            limit=limit,
        )


@pytest.mark.parametrize(
    "query_vector",
    [[1.0], [0.0] * EMBEDDING_DIMENSION, [float("nan"), *([0.0] * 383)]],
)
def test_invalid_vector_error_propagates(
    database_session: Session,
    query_vector: Sequence[float],
) -> None:
    repository = _repository(database_session)

    with pytest.raises(ValueError):
        search_code_units_hybrid(
            database_session,
            repository_id=repository.id,
            query="valid",
            query_vector=query_vector,
        )


def test_invalid_lexical_query_error_propagates(database_session: Session) -> None:
    repository = _repository(database_session)

    with pytest.raises(ValueError, match="non-whitespace"):
        search_code_units_hybrid(
            database_session,
            repository_id=repository.id,
            query=" \t\n ",
            query_vector=_vector(1.0, 0.0),
        )


def test_missing_repository_error_propagates(database_session: Session) -> None:
    with pytest.raises(ValueError, match="Repository does not exist"):
        search_code_units_hybrid(
            database_session,
            repository_id=uuid4(),
            query="valid",
            query_vector=_vector(1.0, 0.0),
        )


def test_default_final_limit_is_ten(database_session: Session) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "many.py")
    for index in range(11):
        _code_unit(
            database_session,
            file,
            embedding=_vector(1.0, float(index + 1)),
            start_line=index + 1,
            symbol_name=f"concept_{index}",
        )

    results = search_code_units_hybrid(
        database_session,
        repository_id=repository.id,
        query="lexically absent",
        query_vector=_vector(1.0, 0.0),
    )

    assert len(results) == 10
