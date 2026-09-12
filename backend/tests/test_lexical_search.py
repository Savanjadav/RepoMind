import os
from collections.abc import Iterator
from typing import cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Select, event
from sqlalchemy.orm import ORMExecuteState, Session

from app.code_parser import CodeUnitKind
from app.database import create_database_engine
from app.lexical_search import LexicalSearchResult, search_code_units_lexically
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.repository import Repository

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


def _repository(session: Session, name: str = "Lexical repository") -> Repository:
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
    code_unit_id: UUID | None = None,
    kind: CodeUnitKind = CodeUnitKind.FUNCTION,
    content: str = "unrelated content",
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
        embedding=None,
    )
    session.add(unit)
    session.flush()
    return unit


def test_exact_identifier_tiers_and_returned_evidence(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/auth.py")
    exact_content = "def authenticate_user():\n    return '✓'\n"
    exact = _code_unit(
        database_session,
        file,
        content=exact_content,
        language="python",
        start_line=10,
        end_line=11,
        symbol_name="authenticate_user",
    )
    substring = _code_unit(
        database_session,
        file,
        start_line=20,
        symbol_name="authenticate_user_token",
    )
    content_only = _code_unit(
        database_session,
        file,
        content="result = authenticate_user(request)",
        start_line=30,
    )
    unrelated = _code_unit(
        database_session,
        file,
        content="def logout(): pass",
        start_line=40,
        symbol_name="logout",
    )
    retrieval_select_count = 0

    def count_retrieval_selects(orm_execute_state: ORMExecuteState) -> None:
        nonlocal retrieval_select_count
        statement = orm_execute_state.statement
        if not orm_execute_state.is_select or not isinstance(statement, Select):
            return
        entities = {
            description.get("entity") for description in statement.column_descriptions
        }
        if CodeUnit in entities and File in entities:
            retrieval_select_count += 1

    event.listen(database_session, "do_orm_execute", count_retrieval_selects)
    try:
        results = search_code_units_lexically(
            database_session,
            repository_id=repository.id,
            query="authenticate_user",
        )
    finally:
        event.remove(database_session, "do_orm_execute", count_retrieval_selects)

    assert [result.code_unit_id for result in results] == [
        exact.id,
        substring.id,
        content_only.id,
    ]
    assert [result.lexical_score for result in results] == [9, 7, 3]
    assert results[0] == LexicalSearchResult(
        code_unit_id=exact.id,
        file_id=file.id,
        path="src/auth.py",
        kind=CodeUnitKind.FUNCTION,
        content=exact_content,
        language="python",
        start_line=10,
        end_line=11,
        symbol_name="authenticate_user",
        lexical_score=9,
    )
    assert unrelated.id not in {result.code_unit_id for result in results}
    assert retrieval_select_count == 1


def test_case_sensitive_exact_symbol_precedes_case_insensitive_exact(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "config/secrets.py")
    exact = _code_unit(database_session, file, symbol_name="JWT_SECRET")
    different_case = _code_unit(
        database_session,
        file,
        start_line=2,
        symbol_name="jwt_secret",
    )

    results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="JWT_SECRET",
    )

    assert [result.code_unit_id for result in results] == [exact.id, different_case.id]
    assert [result.lexical_score for result in results] == [9, 8]
    assert [result.symbol_name for result in results] == ["JWT_SECRET", "jwt_secret"]


def test_case_insensitive_substring_tiers(database_session: Session) -> None:
    repository = _repository(database_session)
    symbol_file = _file(database_session, repository, "src/symbol.py")
    path_file = _file(database_session, repository, "src/Needle/service.py")
    content_file = _file(database_session, repository, "src/content.py")
    symbol = _code_unit(
        database_session,
        symbol_file,
        symbol_name="PrefixNeedleSuffix",
    )
    path = _code_unit(database_session, path_file)
    content = _code_unit(
        database_session,
        content_file,
        content="Use Needle Here",
    )

    results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="needle",
    )

    assert [result.code_unit_id for result in results] == [
        symbol.id,
        path.id,
        content.id,
    ]
    assert [result.lexical_score for result in results] == [6, 4, 2]


@pytest.mark.parametrize("identifier", ["get_user_by_id", "getUserById"])
def test_snake_and_camel_case_identifiers_are_found_exactly(
    database_session: Session,
    identifier: str,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/users.py")
    unit = _code_unit(database_session, file, symbol_name=identifier)

    results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query=identifier,
    )

    assert [(result.code_unit_id, result.lexical_score) for result in results] == [
        (unit.id, 9)
    ]


def test_null_symbol_embedding_null_environment_and_api_content_are_searchable(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "config/runtime.py")
    environment = _code_unit(
        database_session,
        file,
        kind=CodeUnitKind.CONFIG,
        content="database_url = os.getenv('DATABASE_URL')",
        start_line=1,
        symbol_name=None,
    )
    endpoint = _code_unit(
        database_session,
        file,
        content="fetch('/api/login')",
        start_line=2,
        symbol_name=None,
    )

    environment_results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="DATABASE_URL",
    )
    exact_api_results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="/api/login",
    )
    partial_api_results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="api",
    )

    assert [
        (result.code_unit_id, result.lexical_score) for result in environment_results
    ] == [(environment.id, 3)]
    assert [
        (result.code_unit_id, result.lexical_score) for result in exact_api_results
    ] == [(endpoint.id, 3)]
    assert [
        (result.code_unit_id, result.lexical_score) for result in partial_api_results
    ] == [(endpoint.id, 3)]
    assert all(
        result.symbol_name is None
        for result in [
            *environment_results,
            *exact_api_results,
            *partial_api_results,
        ]
    )
    assert environment.embedding is None
    assert endpoint.embedding is None


def test_path_only_match_uses_literal_substring(database_session: Session) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/auth/token_service.py")
    unit = _code_unit(database_session, file)

    results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="token_service",
    )

    assert [(result.code_unit_id, result.lexical_score) for result in results] == [
        (unit.id, 5)
    ]


def test_simple_fts_matches_noncontiguous_query_terms(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/session.py")
    unit = _code_unit(
        database_session,
        file,
        content="token values are safe to rotate",
    )

    results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="rotate token",
    )

    assert [(result.code_unit_id, result.lexical_score) for result in results] == [
        (unit.id, 1)
    ]


def test_repository_scope_excludes_stronger_other_repository_match(
    database_session: Session,
) -> None:
    target_repository = _repository(database_session, "Target")
    target_file = _file(database_session, target_repository, "src/target.py")
    target = _code_unit(
        database_session,
        target_file,
        content="call target_term here",
    )
    other_repository = _repository(database_session, "Other")
    other_file = _file(database_session, other_repository, "src/other.py")
    other = _code_unit(database_session, other_file, symbol_name="target_term")

    results = search_code_units_lexically(
        database_session,
        repository_id=target_repository.id,
        query="target_term",
    )

    assert [result.code_unit_id for result in results] == [target.id]
    assert other.id not in {result.code_unit_id for result in results}


def test_equal_scores_use_deterministic_metadata_tie_breakers(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    a_file = _file(database_session, repository, "a.py")
    b_file = _file(database_session, repository, "b.py")
    low_id = UUID("00000000-0000-0000-0000-000000000001")
    high_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    b_path = _code_unit(database_session, b_file, content="needle")
    later_line = _code_unit(
        database_session,
        a_file,
        content="needle",
        start_line=2,
        end_line=2,
    )
    shorter = _code_unit(
        database_session,
        a_file,
        content="needle",
        start_line=1,
        end_line=5,
    )
    high_id_function = _code_unit(
        database_session,
        a_file,
        code_unit_id=high_id,
        kind=CodeUnitKind.FUNCTION,
        content="needle",
        start_line=1,
        end_line=10,
    )
    class_unit = _code_unit(
        database_session,
        a_file,
        kind=CodeUnitKind.CLASS,
        content="needle",
        start_line=1,
        end_line=10,
    )
    low_id_function = _code_unit(
        database_session,
        a_file,
        code_unit_id=low_id,
        kind=CodeUnitKind.FUNCTION,
        content="needle",
        start_line=1,
        end_line=10,
    )

    first = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="needle",
    )
    second = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="needle",
    )

    expected = [
        class_unit.id,
        low_id_function.id,
        high_id_function.id,
        shorter.id,
        later_line.id,
        b_path.id,
    ]
    assert [result.code_unit_id for result in first] == expected
    assert [result.code_unit_id for result in second] == expected
    assert {result.lexical_score for result in first} == {3}


def test_default_and_explicit_limits_apply_after_ranking(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/limit.py")
    for index in range(11):
        _code_unit(
            database_session,
            file,
            start_line=index + 1,
            symbol_name="target",
        )

    default_results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="target",
    )
    one_result = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="target",
        limit=1,
    )

    assert len(default_results) == 10
    assert [result.start_line for result in default_results] == list(range(1, 11))
    assert [result.start_line for result in one_result] == [1]


@pytest.mark.parametrize("limit", [1, 100])
def test_limit_boundaries_are_accepted(
    database_session: Session,
    limit: int,
) -> None:
    repository = _repository(database_session)

    assert (
        search_code_units_lexically(
            database_session,
            repository_id=repository.id,
            query="missing",
            limit=limit,
        )
        == []
    )


@pytest.mark.parametrize(
    "limit",
    [
        cast(int, True),
        cast(int, False),
        0,
        -1,
        101,
        cast(int, 1.5),
        cast(int, "10"),
    ],
)
def test_invalid_limit_fails_before_database_access(limit: int) -> None:
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)

    with pytest.raises(ValueError, match="Limit must be an integer between 1 and 100"):
        search_code_units_lexically(
            session,
            repository_id=uuid4(),
            query="valid",
            limit=limit,
        )

    assert session_mock.method_calls == []


@pytest.mark.parametrize(
    "query",
    [cast(str, 123), "", " \t\n ", "x" * 2001],
)
def test_invalid_query_fails_before_database_access(query: str) -> None:
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)

    with pytest.raises(ValueError):
        search_code_units_lexically(
            session,
            repository_id=uuid4(),
            query=query,
        )

    assert session_mock.method_calls == []


def test_missing_repository_and_empty_results_are_distinct(
    database_session: Session,
) -> None:
    with pytest.raises(ValueError, match="Repository does not exist"):
        search_code_units_lexically(
            database_session,
            repository_id=uuid4(),
            query="target",
        )

    repository = _repository(database_session)
    assert (
        search_code_units_lexically(
            database_session,
            repository_id=repository.id,
            query="target",
        )
        == []
    )
