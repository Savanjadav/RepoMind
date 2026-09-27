import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import create_database_engine
from app.impact_analysis import (
    AmbiguousImpactTargetError,
    ImpactNotFoundError,
    ImpactValidationError,
    analyze_impact,
)
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.relationship import Relationship
from app.models.repository import Repository


@pytest.fixture
def session() -> Iterator[Session]:
    url = os.getenv("DATABASE_URL")
    if url is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(url)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:
                yield session
        finally:
            transaction.rollback()
    engine.dispose()


def _repository(session: Session) -> Repository:
    repository = Repository(name="impact", source=f"https://example.com/{uuid4()}")
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
    symbol: str = "helper",
    start: int = 2,
    end: int = 8,
    kind: str = "function",
) -> CodeUnit:
    unit = CodeUnit(
        file_id=file.id,
        symbol_name=symbol,
        start_line=start,
        end_line=end,
        kind=kind,
        language="python",
        content="sensitive source not part of impact",
    )
    session.add(unit)
    session.flush()
    return unit


def _import(
    session: Session, repository: Repository, source: File, target: File
) -> None:
    session.add(
        Relationship(
            repository_id=repository.id,
            source_file_id=source.id,
            target_file_id=target.id,
            relationship_type="imports",
        )
    )
    session.flush()


def _call(
    session: Session, repository: Repository, source: CodeUnit, target: CodeUnit
) -> None:
    session.add(
        CodeRelationship(
            repository_id=repository.id,
            source_file_id=source.file_id,
            source_code_unit_id=source.id,
            target_file_id=target.file_id,
            target_code_unit_id=target.id,
            relationship_type="calls",
        )
    )
    session.flush()


@pytest.mark.parametrize(
    "path",
    [
        None,
        3,
        "",
        "/a.py",
        "a\\b.py",
        "a//b",
        "a/",
        "./a",
        "a/../b",
        "a/./b",
        "x" * 2001,
    ]
    + [f"a{chr(code)}.py" for code in [*range(32), 127]],
)
def test_invalid_path_fails_before_database(path: Any) -> None:
    session = Mock(spec=Session)
    with pytest.raises(ImpactValidationError):
        analyze_impact(session, uuid4(), path=path)
    assert session.mock_calls == []


@pytest.mark.parametrize(
    "symbol", [1, "", " \t ", "x" * 2001] + [f"a{chr(c)}" for c in [*range(32), 127]]
)
def test_invalid_symbol_fails_before_database(symbol: Any) -> None:
    session = Mock(spec=Session)
    with pytest.raises(ImpactValidationError):
        analyze_impact(session, uuid4(), path="a.py", symbol=symbol)
    assert session.mock_calls == []


@pytest.mark.parametrize("limit", [True, False, 0, -1, 101, 1.0, "1", None])
def test_invalid_limit_fails_before_database(limit: Any) -> None:
    session = Mock(spec=Session)
    with pytest.raises(ImpactValidationError):
        analyze_impact(session, uuid4(), path="a.py", limit=limit)
    assert session.mock_calls == []


def test_file_impact_is_incoming_one_hop_and_repository_scoped(
    session: Session,
) -> None:
    repository = _repository(session)
    api, service, helper = [
        _file(session, repository, p) for p in ("api.py", "service.py", "helper.py")
    ]
    _import(session, repository, api, service)
    _import(session, repository, service, helper)
    foreign = _repository(session)
    foreign_api = _file(session, foreign, "api.py")
    foreign_service = _file(session, foreign, "service.py")
    _import(session, foreign, foreign_api, foreign_service)
    result = analyze_impact(session, repository.id, path="service.py")
    assert result.mode == "file" and result.analysis == "static_hint"
    assert result.target.file_id == service.id
    assert result.limit == 20 and not result.truncated
    assert [item.file_id for item in result.items] == [api.id]
    item = result.items[0]
    assert item.path == "api.py" and item.relationship_type == "imports"
    assert (
        item.code_unit_id,
        item.symbol_name,
        item.kind,
        item.start_line,
        item.end_line,
    ) == (None,) * 5
    assert analyze_impact(session, repository.id, path="api.py").items == ()
    assert [
        i.file_id
        for i in analyze_impact(session, repository.id, path="helper.py").items
    ] == [service.id]


def test_symbol_impact_is_incoming_one_hop_and_repository_scoped(
    session: Session,
) -> None:
    repository = _repository(session)
    units = [
        _unit(session, _file(session, repository, p), s)
        for p, s in (
            ("api.py", "route_handler"),
            ("service.py", "service_function"),
            ("helper.py", "helper"),
        )
    ]
    route, service, helper = units
    _call(session, repository, route, service)
    _call(session, repository, service, helper)
    _call(session, repository, helper, helper)
    foreign = _repository(session)
    foreign_helper = _unit(session, _file(session, foreign, "helper.py"))
    foreign_service = _unit(
        session, _file(session, foreign, "service.py"), "service_function"
    )
    _call(session, foreign, foreign_service, foreign_helper)
    result = analyze_impact(session, repository.id, path="helper.py", symbol="helper")
    assert result.mode == "symbol" and result.analysis == "static_hint"
    assert result.target.code_unit_id == helper.id
    assert [i.code_unit_id for i in result.items] == [service.id]
    item = result.items[0]
    assert (
        item.file_id,
        item.path,
        item.symbol_name,
        item.kind,
        item.start_line,
        item.end_line,
        item.relationship_type,
    ) == (service.file_id, "service.py", "service_function", "function", 2, 8, "calls")
    assert not result.truncated
    assert (
        analyze_impact(
            session, repository.id, path="api.py", symbol="route_handler"
        ).items
        == ()
    )


@pytest.mark.parametrize(
    "limit,expected,truncated", [(1, 1, True), (3, 3, False), (100, 3, False)]
)
def test_file_order_and_limit(
    session: Session, limit: int, expected: int, truncated: bool
) -> None:
    repo = _repository(session)
    target = _file(session, repo, "target.py")
    for path in ("c.py", "a.py", "b.py"):
        _import(session, repo, _file(session, repo, path), target)
    result = analyze_impact(session, repo.id, path=target.path, limit=limit)
    assert [i.path for i in result.items] == ["a.py", "b.py", "c.py"][:expected]
    assert result.truncated is truncated


@pytest.mark.parametrize("limit", [1, 5, 6, 100])
def test_symbol_order_same_file_and_identical_metadata(
    session: Session, limit: int
) -> None:
    repo = _repository(session)
    file = _file(session, repo, "a.py")
    target = _unit(session, file, "target", 50, 60)
    callers = [
        _unit(session, file, "same", 2, 5),
        _unit(session, file, "same", 2, 5),
        _unit(session, file, "long", 2, 9),
        _unit(session, file, "class_caller", 2, 5, "class"),
        _unit(session, file, "later", 10, 11),
        _unit(session, _file(session, repo, "z.py"), "early", 1, 1),
    ]
    for unit in reversed(callers):
        _call(session, repo, unit, target)
    _call(session, repo, target, target)
    expected = [
        callers[2],
        callers[3],
        *sorted(callers[:2], key=lambda u: u.id),
        callers[4],
        callers[5],
    ]
    result = analyze_impact(
        session, repo.id, path=file.path, symbol="target", limit=limit
    )
    assert [i.code_unit_id for i in result.items] == [u.id for u in expected[:limit]]
    assert result.truncated is (limit < len(expected))
    assert len({i.code_unit_id for i in result.items}) == len(result.items)


@pytest.mark.parametrize("mode", ["file", "symbol"])
def test_duplicate_edges_cannot_create_duplicate_results(
    session: Session, mode: str
) -> None:
    repo = _repository(session)
    source, target = _file(session, repo, "a.py"), _file(session, repo, "b.py")
    if mode == "file":
        _import(session, repo, source, target)
        with pytest.raises(IntegrityError), session.begin_nested():
            _import(session, repo, source, target)
        result = analyze_impact(session, repo.id, path="b.py")
    else:
        caller, callee = _unit(session, source), _unit(session, target)
        _call(session, repo, caller, callee)
        with pytest.raises(IntegrityError), session.begin_nested():
            _call(session, repo, caller, callee)
        result = analyze_impact(session, repo.id, path="b.py", symbol="helper")
    assert len(result.items) == 1


@pytest.mark.parametrize("missing", ["repository", "file", "symbol"])
def test_missing_targets(session: Session, missing: str) -> None:
    repo = _repository(session)
    _file(session, repo, "a.py")
    with pytest.raises(ImpactNotFoundError):
        analyze_impact(
            session,
            uuid4() if missing == "repository" else repo.id,
            path="missing.py" if missing == "file" else "a.py",
            symbol="absent" if missing == "symbol" else None,
        )


def test_ambiguity_is_not_resolved_using_kind_or_edges(session: Session) -> None:
    repo = _repository(session)
    file = _file(session, repo, "a.py")
    first = _unit(session, file, "duplicate", 1, 2)
    _unit(session, file, "duplicate", 3, 4, "class")
    caller = _unit(session, file, "caller", 10, 11)
    _call(session, repo, caller, first)
    with pytest.raises(AmbiguousImpactTargetError):
        analyze_impact(session, repo.id, path=file.path, symbol="duplicate")


@pytest.mark.parametrize(
    "path,symbol", [(" 日本語.py ", " café "), ("x" * 2000, "y" * 2000)]
)
def test_exact_metadata_preserved_and_named_noncallable_can_be_empty(
    session: Session, path: str, symbol: str
) -> None:
    repo = _repository(session)
    file = _file(session, repo, path)
    unit = _unit(session, file, symbol, 4, 12, "class")
    result = analyze_impact(session, repo.id, path=path, symbol=symbol)
    assert result.target.path == path and result.target.symbol_name == symbol
    assert result.target.code_unit_id == unit.id
    assert result.items == ()
    with pytest.raises(ImpactNotFoundError):
        analyze_impact(session, repo.id, path="Z" + path[1:])
    with pytest.raises(ImpactNotFoundError):
        analyze_impact(session, repo.id, path=path, symbol="OTHER")


@pytest.mark.parametrize("symbol_mode,expected_queries", [(False, 3), (True, 4)])
def test_bounded_metadata_queries_and_no_transaction_ownership(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
    symbol_mode: bool,
    expected_queries: int,
) -> None:
    repo = _repository(session)
    file = _file(session, repo, "target.py")
    target = _unit(session, file, "target")
    for index in range(8):
        source = _file(session, repo, f"caller{index}.py")
        _import(session, repo, source, file)
        _call(session, repo, _unit(session, source), target)
    repo_id = repo.id
    pending = Repository(name="pending", source=f"https://example.com/{uuid4()}")
    session.add(pending)
    session.autoflush = True
    statements: list[tuple[str, Any]] = []

    def capture(
        conn: Connection,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        statements.append((statement, parameters))

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Impact service must not own writes or transactions")

    for name in ("add", "add_all", "flush", "commit", "rollback"):
        monkeypatch.setattr(session, name, forbidden)
    connection = session.connection()
    transaction = session.get_transaction()
    event.listen(connection, "before_cursor_execute", capture)
    try:
        result = analyze_impact(
            session,
            repo_id,
            path="target.py",
            symbol="target" if symbol_mode else None,
            limit=2,
        )
    finally:
        event.remove(connection, "before_cursor_execute", capture)
    assert len(result.items) == 2 and result.truncated
    assert len(statements) == expected_queries
    assert all(sql.lstrip().startswith("SELECT") for sql, _ in statements)
    assert all(
        ".content" not in sql and ".embedding" not in sql for sql, _ in statements
    )
    assert "LIMIT" in statements[-1][0] and 3 in statements[-1][1].values()
    if symbol_mode:
        assert "LIMIT" in statements[2][0] and 2 in statements[2][1].values()
    assert pending in session.new and pending.id is None
    assert (
        session.get_transaction() is transaction
        and transaction is not None
        and transaction.is_active
    )


def test_lookup_values_are_not_sql(session: Session) -> None:
    repo = _repository(session)
    path = "x'; SELECT secret; --.py"
    symbol = "name' OR '1'='1"
    file = _file(session, repo, path)
    unit = _unit(session, file, symbol)
    assert (
        analyze_impact(session, repo.id, path=path, symbol=symbol).target.code_unit_id
        == unit.id
    )
