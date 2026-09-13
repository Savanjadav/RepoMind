import os
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.database import create_database_engine
from app.hybrid_search import search_code_units_hybrid
from app.lexical_search import search_code_units_lexically
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.repository import Repository
from app.retrieval_filters import (
    RetrievalFilters,
    build_retrieval_filter_predicates,
)
from app.semantic_search import search_code_units_semantically

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


def _vector(first: float = 1.0, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * (EMBEDDING_DIMENSION - 2))]


def _repository(session: Session, name: str = "Filter repository") -> Repository:
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
    symbol_name: str | None,
    language: str = "python",
    content: str = "unrelated content",
    embedding: list[float] | None = None,
    start_line: int = 1,
) -> CodeUnit:
    unit = CodeUnit(
        file_id=file.id,
        kind=CodeUnitKind.FUNCTION.value,
        content=content,
        language=language,
        start_line=start_line,
        end_line=start_line,
        symbol_name=symbol_name,
        embedding=embedding,
    )
    session.add(unit)
    session.flush()
    return unit


def test_retrieval_filters_are_frozen_slotted_and_default_to_no_predicates() -> None:
    filters = RetrievalFilters()

    assert filters == RetrievalFilters(path=None, language=None, symbol=None)
    assert build_retrieval_filter_predicates(None) == ()
    assert build_retrieval_filter_predicates(filters) == ()
    assert not hasattr(filters, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(filters, "path", "src")


def test_retrieval_filters_preserve_valid_values() -> None:
    filters = RetrievalFilters(
        path=" src/auth ",
        language="python",
        symbol="login_user",
    )

    assert filters.path == " src/auth "
    assert filters.language == "python"
    assert filters.symbol == "login_user"
    assert len(build_retrieval_filter_predicates(filters)) == 3


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RetrievalFilters(path=""),
        lambda: RetrievalFilters(path=" \t "),
        lambda: RetrievalFilters(path=cast(str, 1)),
        lambda: RetrievalFilters(path="p" * 2_001),
        lambda: RetrievalFilters(language=""),
        lambda: RetrievalFilters(language=" \n "),
        lambda: RetrievalFilters(language=cast(str, False)),
        lambda: RetrievalFilters(language="l" * 65),
        lambda: RetrievalFilters(symbol=""),
        lambda: RetrievalFilters(symbol="   "),
        lambda: RetrievalFilters(symbol=cast(str, object())),
        lambda: RetrievalFilters(symbol="s" * 2_001),
    ],
)
def test_invalid_filter_values_are_rejected(
    factory: Callable[[], RetrievalFilters],
) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "filters",
    [
        RetrievalFilters(path="p" * 2_000),
        RetrievalFilters(language="l" * 64),
        RetrievalFilters(symbol="s" * 2_000),
    ],
)
def test_filter_length_boundaries_are_accepted(filters: RetrievalFilters) -> None:
    assert any(
        value is not None for value in (filters.path, filters.language, filters.symbol)
    )


def test_filters_constrain_path_language_symbol_and_combinations(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    auth_python_file = _file(database_session, repository, "src/auth/login.py")
    auth_typescript_file = _file(database_session, repository, "src/auth/login.ts")
    users_file = _file(database_session, repository, "src/users/login.py")
    token_file = _file(database_session, repository, "src/auth/token.py")
    target = _code_unit(
        database_session,
        auth_python_file,
        symbol_name="login_user",
        embedding=_vector(1.0, 0.1),
        start_line=1,
    )
    null_symbol = _code_unit(
        database_session,
        auth_python_file,
        symbol_name=None,
        content="login appears only in content",
        embedding=_vector(1.0, 0.2),
        start_line=2,
    )
    typescript = _code_unit(
        database_session,
        auth_typescript_file,
        symbol_name="login_admin",
        language="typescript",
        embedding=_vector(1.0, 0.3),
    )
    users = _code_unit(
        database_session,
        users_file,
        symbol_name="login_profile",
        embedding=_vector(1.0, 0.4),
    )
    other_symbol = _code_unit(
        database_session,
        token_file,
        symbol_name="authenticate_user",
        content="login appears only in content",
        embedding=_vector(1.0, 0.5),
    )
    other_repository = _repository(database_session, "Other repository")
    other_file = _file(database_session, other_repository, "src/auth/login.py")
    other_repository_match = _code_unit(
        database_session,
        other_file,
        symbol_name="login_user",
        embedding=_vector(1.0, 0.0),
    )

    def search(filters: RetrievalFilters) -> list[UUID]:
        return [
            result.code_unit_id
            for result in search_code_units_semantically(
                database_session,
                repository_id=repository.id,
                query_vector=_vector(),
                limit=100,
                filters=filters,
            )
        ]

    assert set(search(RetrievalFilters(path="src/auth"))) == {
        target.id,
        null_symbol.id,
        typescript.id,
        other_symbol.id,
    }
    assert set(search(RetrievalFilters(language="python"))) == {
        target.id,
        null_symbol.id,
        users.id,
        other_symbol.id,
    }
    assert search(RetrievalFilters(language="Python")) == []
    assert set(search(RetrievalFilters(symbol="login"))) == {
        target.id,
        typescript.id,
        users.id,
    }
    assert set(search(RetrievalFilters(path="src/auth", language="python"))) == {
        target.id,
        null_symbol.id,
        other_symbol.id,
    }
    assert set(search(RetrievalFilters(path="src/auth", symbol="login"))) == {
        target.id,
        typescript.id,
    }
    assert set(search(RetrievalFilters(language="python", symbol="login"))) == {
        target.id,
        users.id,
    }
    assert search(
        RetrievalFilters(path="src/auth", language="python", symbol="login")
    ) == [target.id]
    assert other_repository_match.id not in search(RetrievalFilters())


def test_percent_and_underscore_are_literal_filter_characters(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    literal_file = _file(database_session, repository, "src/rate%value.py")
    wildcard_file = _file(database_session, repository, "src/rateXvalue.py")
    literal_percent = _code_unit(
        database_session,
        literal_file,
        symbol_name="token_value",
        embedding=_vector(1.0, 0.1),
    )
    _code_unit(
        database_session,
        wildcard_file,
        symbol_name="tokenXvalue",
        embedding=_vector(1.0, 0.2),
    )

    percent_results = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(),
        filters=RetrievalFilters(path="rate%value"),
    )
    underscore_results = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(),
        filters=RetrievalFilters(symbol="token_value"),
    )

    assert [result.code_unit_id for result in percent_results] == [literal_percent.id]
    assert [result.code_unit_id for result in underscore_results] == [
        literal_percent.id
    ]


def test_semantic_filters_apply_before_ranking_and_limit(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/search.py")
    for index in range(3):
        _code_unit(
            database_session,
            file,
            symbol_name=f"excluded_{index}",
            language="javascript",
            embedding=_vector(1.0, float(index) / 100.0),
            start_line=index + 1,
        )
    eligible = _code_unit(
        database_session,
        file,
        symbol_name="eligible",
        language="python",
        embedding=_vector(0.0, 1.0),
        start_line=10,
    )

    results = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(),
        limit=1,
        filters=RetrievalFilters(language="python"),
    )

    assert [result.code_unit_id for result in results] == [eligible.id]
    assert results[0].cosine_distance == pytest.approx(1.0)


def test_lexical_filters_apply_before_scoring_and_limit(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    source_file = _file(database_session, repository, "src/settings.py")
    config_file = _file(database_session, repository, "config/settings.py")
    for index in range(3):
        _code_unit(
            database_session,
            source_file,
            symbol_name="needle",
            start_line=index + 1,
        )
    eligible = _code_unit(
        database_session,
        config_file,
        symbol_name="load_settings",
        content="the needle is read here",
    )

    results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="needle",
        limit=1,
        filters=RetrievalFilters(path="config/"),
    )

    assert [result.code_unit_id for result in results] == [eligible.id]
    assert results[0].lexical_score == 3


def test_hybrid_passes_filters_before_source_ranking_and_fusion(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    excluded_file = _file(database_session, repository, "src/excluded.py")
    included_file = _file(database_session, repository, "config/included.py")
    for index in range(4):
        _code_unit(
            database_session,
            excluded_file,
            symbol_name="needle",
            embedding=_vector(1.0, float(index) / 100.0),
            start_line=index + 1,
        )
    eligible = _code_unit(
        database_session,
        included_file,
        symbol_name="load_configuration",
        content="needle configuration",
        embedding=_vector(1.0, 1.0),
    )
    filters = RetrievalFilters(path="config/")

    semantic_results = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(),
        limit=3,
        filters=filters,
    )
    lexical_results = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="needle",
        limit=3,
        filters=filters,
    )
    hybrid_results = search_code_units_hybrid(
        database_session,
        repository_id=repository.id,
        query="needle",
        query_vector=_vector(),
        limit=1,
        filters=filters,
    )

    assert [result.code_unit_id for result in semantic_results] == [eligible.id]
    assert [result.code_unit_id for result in lexical_results] == [eligible.id]
    assert [result.code_unit_id for result in hybrid_results] == [eligible.id]
    assert hybrid_results[0].semantic_rank == 1
    assert hybrid_results[0].lexical_rank == 1


def test_none_and_all_none_filters_preserve_existing_retrieval(
    database_session: Session,
) -> None:
    repository = _repository(database_session)
    file = _file(database_session, repository, "src/example.py")
    _code_unit(
        database_session,
        file,
        symbol_name="needle",
        content="needle",
        embedding=_vector(1.0, 0.0),
        start_line=1,
    )
    _code_unit(
        database_session,
        file,
        symbol_name="needle_helper",
        content="needle helper",
        embedding=_vector(1.0, 0.5),
        start_line=2,
    )

    semantic_without = search_code_units_semantically(
        database_session,
        repository_id=repository.id,
        query_vector=_vector(),
    )
    lexical_without = search_code_units_lexically(
        database_session,
        repository_id=repository.id,
        query="needle",
    )
    hybrid_without = search_code_units_hybrid(
        database_session,
        repository_id=repository.id,
        query="needle",
        query_vector=_vector(),
    )

    for filters in (None, RetrievalFilters()):
        assert (
            search_code_units_semantically(
                database_session,
                repository_id=repository.id,
                query_vector=_vector(),
                filters=filters,
            )
            == semantic_without
        )
        assert (
            search_code_units_lexically(
                database_session,
                repository_id=repository.id,
                query="needle",
                filters=filters,
            )
            == lexical_without
        )
        assert (
            search_code_units_hybrid(
                database_session,
                repository_id=repository.id,
                query="needle",
                query_vector=_vector(),
                filters=filters,
            )
            == hybrid_without
        )
