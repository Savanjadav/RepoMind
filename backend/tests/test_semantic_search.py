import os
from collections.abc import Iterator, Sequence
from math import isfinite, sqrt
from typing import cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Select, event
from sqlalchemy.orm import ORMExecuteState, Session

from app.code_parser import CodeUnitKind
from app.database import create_database_engine
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


class FloatConvertible:
    def __float__(self) -> float:
        return 1.0


def _vector(first: float = 0.0, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * (EMBEDDING_DIMENSION - 2))]


def _repository(session: Session, name: str = "Search repository") -> Repository:
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
    content: str = "def target():\n    return 'exact'\n",
    language: str = "python",
    start_line: int = 1,
    end_line: int = 2,
    symbol_name: str | None = "target",
) -> CodeUnit:
    unit = CodeUnit(
        id=code_unit_id or uuid4(),
        file_id=file.id,
        kind=kind.value,
        content=content,
        language=language,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
        embedding=embedding,
    )
    session.add(unit)
    session.flush()
    return unit


def test_orders_by_cosine_distance_and_returns_exact_evidence(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/meaning.py")
    opposite = _code_unit(
        database_session,
        file,
        embedding=_vector(-1.0, 0.0),
        content="opposite",
        start_line=7,
        end_line=7,
        symbol_name="opposite",
    )
    orthogonal = _code_unit(
        database_session,
        file,
        embedding=_vector(0.0, 1.0),
        content="orthogonal",
        start_line=5,
        end_line=5,
        symbol_name="orthogonal",
    )
    close = _code_unit(
        database_session,
        file,
        embedding=_vector(1.0, 1.0),
        content="close",
        start_line=3,
        end_line=3,
        symbol_name="close",
    )
    exact_content = "def café():\n    return '✓'\n"
    exact = _code_unit(
        database_session,
        file,
        embedding=_vector(1.0, 0.0),
        content=exact_content,
        symbol_name="café",
    )

    semantic_select_count = 0

    def count_selects(orm_execute_state: ORMExecuteState) -> None:
        nonlocal semantic_select_count
        statement = orm_execute_state.statement
        if not orm_execute_state.is_select or not isinstance(statement, Select):
            return
        entities = {
            description.get("entity") for description in statement.column_descriptions
        }
        if CodeUnit in entities and File in entities:
            semantic_select_count += 1

    event.listen(database_session, "do_orm_execute", count_selects)
    try:
        results = search_code_units_semantically(
            database_session,
            repository_id=repository.id,
            query_vector=_vector(1.0, 0.0),
        )
    finally:
        event.remove(database_session, "do_orm_execute", count_selects)

    assert [result.code_unit_id for result in results] == [
        exact.id,
        close.id,
        orthogonal.id,
        opposite.id,
    ]
    assert [result.cosine_distance for result in results] == pytest.approx(
        [0.0, 1.0 - 1.0 / sqrt(2.0), 1.0, 2.0]
    )
    assert results[0] == SemanticSearchResult(
        code_unit_id=exact.id,
        file_id=file.id,
        path="src/meaning.py",
        kind=CodeUnitKind.FUNCTION,
        content=exact_content,
        language="python",
        start_line=1,
        end_line=2,
        symbol_name="café",
        cosine_distance=results[0].cosine_distance,
    )
    assert results[0].cosine_distance == pytest.approx(0.0)
    assert semantic_select_count == 1


def test_scopes_results_to_repository(database_session: Session) -> None:
    target_repository = _repository(database_session, "Target")
    target_file = _file(database_session, target_repository, "src/target.py")
    target = _code_unit(
        database_session,
        target_file,
        embedding=_vector(1.0, 1.0),
    )
    other_repository = _repository(database_session, "Other")
    other_file = _file(database_session, other_repository, "src/other.py")
    _code_unit(database_session, other_file, embedding=_vector(1.0, 0.0))

    results = search_code_units_semantically(
        database_session,
        repository_id=target_repository.id,
        query_vector=_vector(1.0, 0.0),
    )

    assert [result.code_unit_id for result in results] == [target.id]


def test_null_and_stored_zero_embeddings_do_not_enter_cosine_results(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/vectors.py")
    valid = _code_unit(database_session, file, embedding=_vector(1.0, 0.0))
    null_unit = _code_unit(
        database_session,
        file,
        embedding=None,
        symbol_name="null",
    )
    zero_unit = _code_unit(
        database_session,
        file,
        embedding=[0.0] * EMBEDDING_DIMENSION,
        symbol_name="zero",
    )

    results = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(1.0, 0.0),
    )

    assert [result.code_unit_id for result in results] == [valid.id]
    assert null_unit.id not in {result.code_unit_id for result in results}
    assert zero_unit.id not in {result.code_unit_id for result in results}
    assert all(isfinite(result.cosine_distance) for result in results)


def test_limit_is_applied_after_distance_ordering(database_session: Session) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/limit.py")
    for index in range(11):
        _code_unit(
            database_session,
            file,
            embedding=_vector(1.0, float(index + 1)),
            start_line=index + 1,
            end_line=index + 1,
            symbol_name=f"unit_{index}",
        )

    default_results = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(1.0, 0.0),
    )
    one_result = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(1.0, 0.0),
        limit=1,
    )

    assert len(default_results) == 10
    assert [result.symbol_name for result in default_results] == [
        f"unit_{index}" for index in range(10)
    ]
    assert [result.symbol_name for result in one_result] == ["unit_0"]


@pytest.mark.parametrize("limit", [1, 100])
def test_limit_boundaries_are_accepted(
    database_session: Session,
    limit: int,
) -> None:
    repository = _repository(database_session)

    assert (
        search_code_units_semantically(
            database_session,
            repository_id=repository.id,
            query_vector=_vector(1.0, 0.0),
            limit=limit,
        )
        == []
    )


@pytest.mark.parametrize("limit", [0, -1, 101, cast(int, True)])
def test_invalid_limit_fails_before_database_access(limit: int) -> None:
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)

    with pytest.raises(ValueError, match="Limit must be an integer between 1 and 100"):
        search_code_units_semantically(
            session,
            repository_id=uuid4(),
            query_vector=_vector(1.0, 0.0),
            limit=limit,
        )

    assert session_mock.method_calls == []


@pytest.mark.parametrize(
    ("query_vector", "message"),
    [
        ([1.0], "exactly 384"),
        ([cast(float, True), *([0.0] * (EMBEDDING_DIMENSION - 1))], "real"),
        ([cast(float, "1.0"), *([0.0] * (EMBEDDING_DIMENSION - 1))], "real"),
        (
            [cast(float, FloatConvertible()), *([0.0] * (EMBEDDING_DIMENSION - 1))],
            "real",
        ),
        ([float("nan"), *([0.0] * (EMBEDDING_DIMENSION - 1))], "finite"),
        ([float("inf"), *([0.0] * (EMBEDDING_DIMENSION - 1))], "finite"),
        ([float("-inf"), *([0.0] * (EMBEDDING_DIMENSION - 1))], "finite"),
        ([0.0] * EMBEDDING_DIMENSION, "must not be all zero"),
    ],
)
def test_invalid_query_vector_fails_before_database_access(
    query_vector: Sequence[float],
    message: str,
) -> None:
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)

    with pytest.raises(ValueError, match=message):
        search_code_units_semantically(
            session,
            repository_id=uuid4(),
            query_vector=query_vector,
        )

    assert session_mock.method_calls == []


def test_nonzero_query_with_zero_components_is_accepted(
    database_session: Session,
) -> None:
    repository = _repository(database_session)

    assert (
        search_code_units_semantically(
            database_session,
            repository_id=repository.id,
            query_vector=_vector(1.0, 0.0),
        )
        == []
    )


def test_missing_repository_is_distinct_from_empty_results(
    database_session: Session,
) -> None:
    with pytest.raises(ValueError, match="Repository does not exist"):
        search_code_units_semantically(
            database_session,
            repository_id=uuid4(),
            query_vector=_vector(1.0, 0.0),
        )

    repository = _repository(database_session)
    assert (
        search_code_units_semantically(
            database_session,
            repository_id=repository.id,
            query_vector=_vector(1.0, 0.0),
        )
        == []
    )


def test_equal_distances_use_deterministic_metadata_tie_breakers(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    a_file = _file(database_session, repository, "a.py")
    b_file = _file(database_session, repository, "b.py")
    high_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    low_id = UUID("00000000-0000-0000-0000-000000000001")
    b_path = _code_unit(
        database_session,
        b_file,
        embedding=_vector(1.0, 0.0),
        symbol_name="b_path",
    )
    later_line = _code_unit(
        database_session,
        a_file,
        embedding=_vector(1.0, 0.0),
        start_line=2,
        end_line=2,
        symbol_name="later_line",
    )
    shorter = _code_unit(
        database_session,
        a_file,
        embedding=_vector(1.0, 0.0),
        start_line=1,
        end_line=5,
        symbol_name="shorter",
    )
    function = _code_unit(
        database_session,
        a_file,
        embedding=_vector(1.0, 0.0),
        code_unit_id=high_id,
        kind=CodeUnitKind.FUNCTION,
        start_line=1,
        end_line=10,
        symbol_name="function_high_id",
    )
    class_unit = _code_unit(
        database_session,
        a_file,
        embedding=_vector(1.0, 0.0),
        kind=CodeUnitKind.CLASS,
        start_line=1,
        end_line=10,
        symbol_name="class",
    )
    lower_id_function = _code_unit(
        database_session,
        a_file,
        embedding=_vector(1.0, 0.0),
        code_unit_id=low_id,
        kind=CodeUnitKind.FUNCTION,
        start_line=1,
        end_line=10,
        symbol_name="function_low_id",
    )

    results = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(1.0, 0.0),
    )

    assert [result.code_unit_id for result in results] == [
        class_unit.id,
        lower_id_function.id,
        function.id,
        shorter.id,
        later_line.id,
        b_path.id,
    ]
