import os
from collections.abc import Iterator
from typing import cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind, ParsedCodeUnit
from app.code_unit_persistence import persist_code_units
from app.database import create_database_engine
from app.models import CodeUnit, File, Repository

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


def _file(session: Session, path: str = "src/main.py") -> File:
    repository = Repository(
        name="Code unit test repository",
        source=f"https://example.com/{uuid4()}.git",
    )
    file = File(path=path)
    repository.files.append(file)
    session.add(repository)
    session.flush()
    return file


def _unit(
    *,
    kind: CodeUnitKind = CodeUnitKind.FUNCTION,
    content: str = "def main():\n    return True",
    relative_path: str = "src/main.py",
    language: str = "python",
    start_line: int = 1,
    end_line: int = 2,
    symbol_name: str | None = "main",
) -> ParsedCodeUnit:
    return ParsedCodeUnit(
        kind=kind,
        content=content,
        relative_path=relative_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        symbol_name=symbol_name,
    )


def _direct_code_unit(
    file_id: UUID,
    *,
    kind: str = "function",
    start_line: int = 1,
    end_line: int = 1,
) -> CodeUnit:
    return CodeUnit(
        file_id=file_id,
        kind=kind,
        content="source",
        language="python",
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
    )


def test_persists_all_supported_kinds_with_exact_fields_and_relationships(
    database_session: Session,
) -> None:
    file = _file(database_session)
    parsed_units = [
        _unit(
            kind=CodeUnitKind.CLASS,
            content="class Café:\n    pass",
            language="python",
            start_line=1,
            end_line=2,
            symbol_name="Café",
        ),
        _unit(
            kind=CodeUnitKind.FUNCTION,
            content="function run() { return '✓'; }",
            language="javascript",
            start_line=3,
            end_line=3,
            symbol_name="run",
        ),
        _unit(
            kind=CodeUnitKind.IMPORT,
            content='import type { User } from "./types";',
            language="typescript",
            start_line=4,
            end_line=4,
            symbol_name=None,
        ),
        _unit(
            kind=CodeUnitKind.DOCUMENT,
            content="# Authentication\nUse OAuth.",
            language="documentation",
            start_line=5,
            end_line=6,
            symbol_name="Authentication",
        ),
        _unit(
            kind=CodeUnitKind.CONFIG,
            content='greeting = "✓ Ready"',
            language="config",
            start_line=7,
            end_line=7,
            symbol_name=None,
        ),
    ]

    rows = persist_code_units(database_session, file.id, parsed_units)

    assert len(rows) == len(parsed_units)
    assert all(isinstance(row.id, UUID) for row in rows)
    assert [row.file_id for row in rows] == [file.id] * len(parsed_units)
    assert [row.kind for row in rows] == [unit.kind.value for unit in parsed_units]
    assert [row.content for row in rows] == [unit.content for unit in parsed_units]
    assert [row.language for row in rows] == [unit.language for unit in parsed_units]
    assert [row.start_line for row in rows] == [
        unit.start_line for unit in parsed_units
    ]
    assert [row.end_line for row in rows] == [unit.end_line for unit in parsed_units]
    assert [row.symbol_name for row in rows] == [
        unit.symbol_name for unit in parsed_units
    ]

    database_session.expunge_all()
    stored = database_session.get(CodeUnit, rows[0].id)
    assert stored is not None
    assert stored.file.id == file.id
    assert stored.file.repository.id == file.repository_id
    assert stored in stored.file.code_units
    assert not hasattr(stored, "repository_id")
    assert not hasattr(stored, "relative_path")
    assert not hasattr(stored, "ordinal")


def test_preserves_input_order_and_identical_units_are_not_deduplicated(
    database_session: Session,
) -> None:
    file = _file(database_session)
    later = _unit(content="later", start_line=10, end_line=10, symbol_name="later")
    earlier = _unit(content="earlier", start_line=1, end_line=1, symbol_name="earlier")

    rows = persist_code_units(
        database_session,
        file.id,
        [later, earlier, earlier],
    )

    assert [row.content for row in rows] == ["later", "earlier", "earlier"]
    assert len({row.id for row in rows}) == 3
    assert (
        database_session.scalar(
            select(func.count())
            .select_from(CodeUnit)
            .where(CodeUnit.file_id == file.id)
        )
        == 3
    )


def test_persistence_is_additive_and_empty_input_deletes_nothing(
    database_session: Session,
) -> None:
    file = _file(database_session)
    unit = _unit()

    first = persist_code_units(database_session, file.id, [unit])
    second = persist_code_units(database_session, file.id, [unit])
    empty = persist_code_units(database_session, file.id, [])

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].id != second[0].id
    assert empty == []
    assert (
        database_session.scalar(
            select(func.count())
            .select_from(CodeUnit)
            .where(CodeUnit.file_id == file.id)
        )
        == 2
    )


def test_empty_input_is_a_true_no_op_without_file_validation() -> None:
    session_mock = Mock(spec=Session)
    session = cast(Session, session_mock)

    assert persist_code_units(session, uuid4(), []) == []
    assert session_mock.method_calls == []


def test_nonexistent_file_is_rejected(database_session: Session) -> None:
    with pytest.raises(ValueError, match="File does not exist"):
        persist_code_units(database_session, uuid4(), [_unit()])

    assert not any(isinstance(item, CodeUnit) for item in database_session.new)


def test_path_mismatch_rejects_all_units_before_orm_mutation(
    database_session: Session,
) -> None:
    file = _file(database_session, "src/auth.py")
    units = [
        _unit(relative_path="src/auth.py"),
        _unit(relative_path="src/other.py"),
    ]
    pending_before = set(database_session.new)

    with pytest.raises(ValueError, match="does not match target file"):
        persist_code_units(database_session, file.id, units)

    assert set(database_session.new) == pending_before
    assert (
        database_session.scalar(
            select(func.count())
            .select_from(CodeUnit)
            .where(CodeUnit.file_id == file.id)
        )
        == 0
    )


@pytest.mark.parametrize(
    "invalid_row",
    [
        {"kind": "method"},
        {"start_line": 0},
        {"start_line": 3, "end_line": 2},
    ],
)
def test_database_check_constraints_reject_invalid_rows(
    database_session: Session,
    invalid_row: dict[str, object],
) -> None:
    file = _file(database_session)
    row = _direct_code_unit(
        file.id,
        kind=cast(str, invalid_row.get("kind", "function")),
        start_line=cast(int, invalid_row.get("start_line", 1)),
        end_line=cast(int, invalid_row.get("end_line", 1)),
    )

    with pytest.raises(IntegrityError):
        with database_session.begin_nested():
            database_session.add(row)
            database_session.flush()


@pytest.mark.parametrize("invalid_file_id", [None, uuid4()])
def test_database_requires_existing_file_foreign_key(
    database_session: Session,
    invalid_file_id: UUID | None,
) -> None:
    row = _direct_code_unit(cast(UUID, invalid_file_id))

    with pytest.raises(IntegrityError):
        with database_session.begin_nested():
            database_session.add(row)
            database_session.flush()


def test_overlapping_source_ranges_are_allowed(database_session: Session) -> None:
    file = _file(database_session)
    rows = persist_code_units(
        database_session,
        file.id,
        [
            _unit(
                kind=CodeUnitKind.CLASS,
                content="class Service: ...",
                start_line=1,
                end_line=20,
                symbol_name="Service",
            ),
            _unit(
                content="def run(self): ...",
                start_line=5,
                end_line=10,
                symbol_name="run",
            ),
        ],
    )

    assert [(row.start_line, row.end_line) for row in rows] == [(1, 20), (5, 10)]


def test_file_deletion_uses_database_cascade(database_session: Session) -> None:
    file = _file(database_session)
    rows = persist_code_units(database_session, file.id, [_unit(), _unit()])
    file_id = file.id
    row_ids = [row.id for row in rows]
    database_session.expunge_all()

    file_to_delete = database_session.get(File, file_id)
    assert file_to_delete is not None
    database_session.delete(file_to_delete)
    database_session.flush()

    assert all(database_session.get(CodeUnit, row_id) is None for row_id in row_ids)


def test_repository_deletion_cascades_through_files(
    database_session: Session,
) -> None:
    file = _file(database_session)
    repository_id = file.repository_id
    row = persist_code_units(database_session, file.id, [_unit()])[0]
    row_id = row.id
    database_session.expunge_all()

    repository = database_session.get(Repository, repository_id)
    assert repository is not None
    database_session.delete(repository)
    database_session.flush()

    assert database_session.get(CodeUnit, row_id) is None


def test_migration_schema_matches_model(database_session: Session) -> None:
    inspector = inspect(database_session.get_bind())
    columns = {column["name"]: column for column in inspector.get_columns("code_units")}

    assert set(columns) == {
        "id",
        "file_id",
        "kind",
        "content",
        "language",
        "start_line",
        "end_line",
        "symbol_name",
        "embedding",
    }
    assert columns["symbol_name"]["nullable"] is True
    assert columns["embedding"]["nullable"] is True
    assert str(columns["embedding"]["type"]).lower() == "vector(384)"
    assert all(
        columns[name]["nullable"] is False
        for name in set(columns) - {"symbol_name", "embedding"}
    )

    foreign_keys = inspector.get_foreign_keys("code_units")
    assert len(foreign_keys) == 1
    assert foreign_keys[0]["name"] == "fk_code_units_file_id_files"
    assert foreign_keys[0]["referred_table"] == "files"
    assert foreign_keys[0]["constrained_columns"] == ["file_id"]
    assert foreign_keys[0]["options"]["ondelete"] == "CASCADE"

    assert {
        constraint["name"]
        for constraint in inspector.get_check_constraints("code_units")
    } == {
        "ck_code_units_end_line_not_before_start",
        "ck_code_units_kind",
        "ck_code_units_start_line_positive",
    }
    assert {index["name"] for index in inspector.get_indexes("code_units")} == {
        "ix_code_units_embedding_hnsw",
        "ix_code_units_file_id",
    }


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

            def unexpected_transaction_method() -> None:
                pytest.fail("Persistence service must not own the transaction")

            monkeypatch.setattr(session, "commit", unexpected_transaction_method)
            monkeypatch.setattr(session, "rollback", unexpected_transaction_method)
            row = persist_code_units(session, file.id, [_unit()])[0]
            row_id = row.id

        transaction.rollback()

    with Session(engine) as verification_session:
        assert verification_session.get(CodeUnit, row_id) is None
    engine.dispose()


def test_caller_can_commit_persisted_rows() -> None:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")

    engine = create_database_engine(DATABASE_URL)
    repository_id: UUID | None = None
    row_id: UUID | None = None
    try:
        with Session(engine) as session:
            file = _file(session)
            repository_id = file.repository_id
            row = persist_code_units(session, file.id, [_unit()])[0]
            row_id = row.id
            session.commit()

        with Session(engine) as session:
            stored = session.get(CodeUnit, row_id)
            assert stored is not None
            assert stored.content == _unit().content
    finally:
        if repository_id is not None:
            with Session(engine) as cleanup_session:
                repository = cleanup_session.get(Repository, repository_id)
                if repository is not None:
                    cleanup_session.delete(repository)
                    cleanup_session.commit()

        if row_id is not None:
            with Session(engine) as verification_session:
                assert verification_session.get(CodeUnit, row_id) is None
        engine.dispose()
