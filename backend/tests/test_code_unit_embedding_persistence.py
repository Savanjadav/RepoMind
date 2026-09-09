import os
from collections.abc import Iterator, Sequence
from typing import cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import DataError
from sqlalchemy.orm import ORMExecuteState, Session

from app.code_unit_embedding_persistence import (
    CodeUnitEmbedding,
    persist_code_unit_embeddings,
)
from app.database import create_database_engine
from app.models import CodeUnit, File, Repository
from app.models.code_unit import EMBEDDING_DIMENSION

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


def _file(session: Session) -> File:
    repository = Repository(
        name="Embedding persistence repository",
        source=f"https://example.com/{uuid4()}.git",
    )
    file = File(path="src/main.py")
    repository.files.append(file)
    session.add(repository)
    session.flush()
    return file


def _code_unit(session: Session, file: File, symbol_name: str) -> CodeUnit:
    unit = CodeUnit(
        file_id=file.id,
        kind="function",
        content=f"def {symbol_name}(): pass",
        language="python",
        start_line=1,
        end_line=1,
        symbol_name=symbol_name,
    )
    session.add(unit)
    session.flush()
    return unit


def _vector(seed: float = 0.0) -> list[float]:
    return [seed + index / 1000 for index in range(EMBEDDING_DIMENSION)]


def _request(unit: CodeUnit, vector: Sequence[float]) -> CodeUnitEmbedding:
    return CodeUnitEmbedding(code_unit_id=unit.id, vector=vector)


def test_empty_input_is_a_true_no_op() -> None:
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)

    assert persist_code_unit_embeddings(session, []) == []
    assert session_mock.method_calls == []


def test_persists_multiple_vectors_in_input_order_with_one_lookup(
    database_session: Session,
) -> None:
    file = _file(database_session)
    first = _code_unit(database_session, file, "first")
    second = _code_unit(database_session, file, "second")
    assert first.embedding is None
    assert second.embedding is None

    select_count = 0

    def count_selects(orm_execute_state: ORMExecuteState) -> None:
        nonlocal select_count
        if orm_execute_state.is_select:
            select_count += 1

    event.listen(database_session, "do_orm_execute", count_selects)
    try:
        returned = persist_code_unit_embeddings(
            database_session,
            [
                _request(second, _vector(2.0)),
                _request(first, _vector(1.0)),
            ],
        )
    finally:
        event.remove(database_session, "do_orm_execute", count_selects)

    assert [unit.id for unit in returned] == [second.id, first.id]
    assert select_count == 1
    assert second.embedding == pytest.approx(_vector(2.0))
    assert first.embedding == pytest.approx(_vector(1.0))


def test_vector_survives_expire_and_reload_with_plain_list_readback(
    database_session: Session,
) -> None:
    file = _file(database_session)
    unit = _code_unit(database_session, file, "reload")
    expected = _vector(0.123456789)

    persist_code_unit_embeddings(database_session, [_request(unit, expected)])
    unit_id = unit.id
    database_session.expire_all()
    reloaded = database_session.get(CodeUnit, unit_id)

    assert reloaded is not None
    assert type(reloaded.embedding) is list
    assert reloaded.embedding == pytest.approx(expected, abs=1e-6)


def test_existing_embedding_is_replaced_without_creating_rows(
    database_session: Session,
) -> None:
    file = _file(database_session)
    updated = _code_unit(database_session, file, "updated")
    unrelated = _code_unit(database_session, file, "unrelated")
    first_vector = _vector(1.0)
    replacement = _vector(2.0)
    unrelated_vector = _vector(3.0)
    persist_code_unit_embeddings(
        database_session,
        [_request(updated, first_vector), _request(unrelated, unrelated_vector)],
    )
    count_before = database_session.scalar(select(func.count()).select_from(CodeUnit))

    returned = persist_code_unit_embeddings(
        database_session,
        [_request(updated, replacement)],
    )

    assert returned == [updated]
    assert updated.embedding == pytest.approx(replacement)
    assert unrelated.embedding == pytest.approx(unrelated_vector)
    assert (
        database_session.scalar(select(func.count()).select_from(CodeUnit))
        == count_before
    )


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([0.0] * (EMBEDDING_DIMENSION - 1), "exactly 384"),
        ([cast(float, True)] + [0.0] * (EMBEDDING_DIMENSION - 1), "real numbers"),
        ([cast(float, "1.0")] + [0.0] * (EMBEDDING_DIMENSION - 1), "real numbers"),
        ([float("nan")] + [0.0] * (EMBEDDING_DIMENSION - 1), "finite"),
        ([float("inf")] + [0.0] * (EMBEDDING_DIMENSION - 1), "finite"),
        ([float("-inf")] + [0.0] * (EMBEDDING_DIMENSION - 1), "finite"),
    ],
)
def test_invalid_vector_is_rejected_before_lookup_or_mutation(
    database_session: Session,
    vector: Sequence[float],
    message: str,
) -> None:
    file = _file(database_session)
    unit = _code_unit(database_session, file, "invalid")
    select_count = 0

    def count_selects(orm_execute_state: ORMExecuteState) -> None:
        nonlocal select_count
        if orm_execute_state.is_select:
            select_count += 1

    event.listen(database_session, "do_orm_execute", count_selects)
    try:
        with pytest.raises(ValueError, match=message):
            persist_code_unit_embeddings(database_session, [_request(unit, vector)])
    finally:
        event.remove(database_session, "do_orm_execute", count_selects)

    assert select_count == 0
    assert unit.embedding is None


def test_zero_vector_is_persisted(database_session: Session) -> None:
    file = _file(database_session)
    unit = _code_unit(database_session, file, "zero")
    zero_vector = [0.0] * EMBEDDING_DIMENSION

    persist_code_unit_embeddings(database_session, [_request(unit, zero_vector)])

    assert unit.embedding == pytest.approx(zero_vector)


def test_duplicate_id_is_rejected_before_database_work() -> None:
    code_unit_id = uuid4()
    request = CodeUnitEmbedding(code_unit_id, _vector())
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)

    with pytest.raises(ValueError, match="Duplicate CodeUnit ID"):
        persist_code_unit_embeddings(session, [request, request])

    assert session_mock.method_calls == []


def test_unknown_id_rejects_mixed_request_without_partial_assignment(
    database_session: Session,
) -> None:
    file = _file(database_session)
    first = _code_unit(database_session, file, "first")
    second = _code_unit(database_session, file, "second")

    with pytest.raises(ValueError, match="CodeUnit does not exist"):
        persist_code_unit_embeddings(
            database_session,
            [
                _request(first, _vector(1.0)),
                CodeUnitEmbedding(uuid4(), _vector(2.0)),
                _request(second, _vector(3.0)),
            ],
        )

    assert first.embedding is None
    assert second.embedding is None


def test_invalid_vector_rejects_mixed_request_without_partial_assignment(
    database_session: Session,
) -> None:
    file = _file(database_session)
    first = _code_unit(database_session, file, "first")
    second = _code_unit(database_session, file, "second")

    with pytest.raises(ValueError, match="exactly 384"):
        persist_code_unit_embeddings(
            database_session,
            [
                _request(first, _vector(1.0)),
                _request(second, [0.0]),
            ],
        )

    assert first.embedding is None
    assert second.embedding is None


def test_database_rejects_wrong_vector_dimension(
    database_session: Session,
) -> None:
    file = _file(database_session)
    unit = _code_unit(database_session, file, "database_dimension")

    with pytest.raises(DataError):
        with database_session.begin_nested():
            database_session.execute(
                text(
                    """
                    UPDATE code_units
                    SET embedding = '[0]'::vector
                    WHERE id = :code_unit_id
                    """
                ),
                {"code_unit_id": unit.id},
            )


def test_schema_has_vector_extension_column_and_hnsw_cosine_index(
    database_session: Session,
) -> None:
    extension_version = database_session.scalar(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    )
    column = database_session.execute(
        text(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod),
                   attribute.attnotnull
            FROM pg_attribute AS attribute
            JOIN pg_class AS table_class ON table_class.oid = attribute.attrelid
            WHERE table_class.relname = 'code_units'
              AND attribute.attname = 'embedding'
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
            """
        )
    ).one()
    index = database_session.execute(
        text(
            """
            SELECT access_method.amname, operator_class.opcname
            FROM pg_class AS index_class
            JOIN pg_index AS index_metadata
              ON index_metadata.indexrelid = index_class.oid
            JOIN pg_am AS access_method
              ON access_method.oid = index_class.relam
            JOIN LATERAL unnest(index_metadata.indclass)
              AS indexed_operator_class(oid) ON TRUE
            JOIN pg_opclass AS operator_class
              ON operator_class.oid = indexed_operator_class.oid
            WHERE index_class.relname = 'ix_code_units_embedding_hnsw'
            """
        )
    ).one()

    assert extension_version is not None
    assert tuple(column) == ("vector(384)", False)
    assert tuple(index) == ("hnsw", "vector_cosine_ops")


def test_service_leaves_transaction_control_to_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")

    engine = create_database_engine(DATABASE_URL)
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection) as session:
            file = _file(session)
            unit = _code_unit(session, file, "transaction")

            def unexpected_transaction_method() -> None:
                pytest.fail("Persistence service must not own the transaction")

            monkeypatch.setattr(session, "commit", unexpected_transaction_method)
            monkeypatch.setattr(session, "rollback", unexpected_transaction_method)
            persist_code_unit_embeddings(session, [_request(unit, _vector())])
            unit_id = unit.id

        transaction.rollback()

    with Session(engine) as verification_session:
        assert verification_session.get(CodeUnit, unit_id) is None
    engine.dispose()
