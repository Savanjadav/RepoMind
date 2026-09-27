import os
from collections.abc import Iterator
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from app import dependency_expansion as module
from app.code_parser import CodeUnitKind
from app.database import create_database_engine
from app.dependency_expansion import (
    RetrievalEvidence,
    search_code_units_with_dependencies,
)
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.repository import Repository
from app.retrieval_filters import RetrievalFilters


def _evidence(number: int) -> RetrievalEvidence:
    return RetrievalEvidence(
        UUID(int=number),
        UUID(int=number + 1000),
        f"{number}.py",
        CodeUnitKind.FUNCTION,
        "source",
        "python",
        1,
        2,
        f"f{number}",
    )


@pytest.mark.parametrize("limit", [1, 2, 3, 10, 100])
@pytest.mark.parametrize("available", [0, 1, 2, 3, 4, 6])
def test_actual_additions_reserve_only_used_slots(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
    available: int,
) -> None:
    direct = [_evidence(i) for i in range(1, limit + 1)]
    additions = [_evidence(200 + i) for i in range(available)]
    retrieval = Mock(return_value=direct)
    neighbors = Mock(return_value=additions)
    monkeypatch.setattr(module, "search_code_units_reranked", retrieval)
    monkeypatch.setattr(module, "_related_evidence", neighbors)
    repository_id = uuid4()
    reranker = Mock()
    vector = [1.0]
    filters = RetrievalFilters(language="python")
    with Session() as session:
        result = search_code_units_with_dependencies(
            session,
            repository_id=repository_id,
            query=" original ",
            query_vector=vector,
            reranker=reranker,
            limit=limit,
            filters=filters,
        )
    count = min(available, 4, limit // 3)
    assert result == direct[: limit - count] + additions[:count]
    assert len(result) == limit
    retrieval.assert_called_once_with(
        session,
        repository_id=repository_id,
        query=" original ",
        query_vector=vector,
        reranker=reranker,
        limit=limit,
        filters=filters,
    )
    reranker.score.assert_not_called()
    if limit < 3:
        neighbors.assert_not_called()
    else:
        assert (
            neighbors.call_args.kwargs["seeds"]
            == direct[: min(3, limit - min(4, limit // 3))]
        )
        assert neighbors.call_args.kwargs["excluded_ids"] == {
            r.code_unit_id for r in direct
        }
        assert neighbors.call_args.kwargs["filters"] is filters


def test_deduplicates_entire_direct_list_and_shared_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = [_evidence(i) for i in range(1, 11)]
    monkeypatch.setattr(
        module, "search_code_units_reranked", Mock(return_value=direct + direct[:1])
    )
    monkeypatch.setattr(
        module,
        "_related_evidence",
        Mock(return_value=[direct[-1], _evidence(20), _evidence(20), _evidence(21)]),
    )
    with Session() as session:
        result = search_code_units_with_dependencies(
            session,
            repository_id=uuid4(),
            query="q",
            query_vector=[1.0],
            reranker=Mock(),
        )
    assert result == direct[:8] + [_evidence(20), _evidence(21)]
    assert not hasattr(result[-1], "hybrid_score")
    assert not hasattr(result[-1], "original_hybrid_rank")


def test_empty_and_short_direct_results(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieval = Mock(return_value=[])
    neighbors = Mock(return_value=[_evidence(2)])
    monkeypatch.setattr(module, "search_code_units_reranked", retrieval)
    monkeypatch.setattr(module, "_related_evidence", neighbors)
    with Session() as session:
        assert (
            search_code_units_with_dependencies(
                session,
                repository_id=uuid4(),
                query="q",
                query_vector=[1.0],
                reranker=Mock(),
            )
            == []
        )
        neighbors.assert_not_called()
        retrieval.return_value = [_evidence(1)]
        assert search_code_units_with_dependencies(
            session,
            repository_id=uuid4(),
            query="q",
            query_vector=[1.0],
            reranker=Mock(),
        ) == [_evidence(1), _evidence(2)]


@pytest.mark.parametrize("limit", [True, False, 0, -1, 101, 1.5, "3", None])
def test_invalid_limit_before_retrieval(
    monkeypatch: pytest.MonkeyPatch, limit: object
) -> None:
    retrieval = Mock()
    monkeypatch.setattr(module, "search_code_units_reranked", retrieval)
    with Session() as session, pytest.raises(ValueError):
        search_code_units_with_dependencies(
            session,
            repository_id=uuid4(),
            query="q",
            query_vector=[1.0],
            reranker=Mock(),
            limit=limit,  # type: ignore[arg-type]
        )
    retrieval.assert_not_called()


def test_unexpected_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = RuntimeError("unexpected")
    monkeypatch.setattr(module, "search_code_units_reranked", Mock(side_effect=failure))
    with Session() as session, pytest.raises(RuntimeError) as caught:
        search_code_units_with_dependencies(
            session,
            repository_id=uuid4(),
            query="q",
            query_vector=[1.0],
            reranker=Mock(),
        )
    assert caught.value is failure


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


def _repo(session: Session) -> Repository:
    repo = Repository(name="flow", source=f"https://example.com/{uuid4()}")
    session.add(repo)
    session.flush()
    return repo


def _unit(session: Session, repo: Repository, path: str) -> CodeUnit:
    file = File(repository_id=repo.id, path=path)
    session.add(file)
    session.flush()
    unit = CodeUnit(
        file_id=file.id,
        kind="function",
        content="source",
        language="python",
        start_line=1,
        end_line=3,
        symbol_name=path,
        embedding=None,
    )
    session.add(unit)
    session.flush()
    return unit


def _edge(
    session: Session, repo: Repository, source: CodeUnit, target: CodeUnit
) -> None:
    session.add(
        CodeRelationship(
            repository_id=repo.id,
            source_file_id=source.file_id,
            source_code_unit_id=source.id,
            target_file_id=target.file_id,
            target_code_unit_id=target.id,
            relationship_type="calls",
        )
    )
    session.flush()


def _dto(unit: CodeUnit, path: str) -> RetrievalEvidence:
    return RetrievalEvidence(
        unit.id,
        unit.file_id,
        path,
        CodeUnitKind.FUNCTION,
        unit.content,
        unit.language,
        unit.start_line,
        unit.end_line,
        unit.symbol_name,
    )


def test_sql_caps_and_seed_first_order(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(session)
    seeds = [_unit(session, repo, f"seed{i}.py") for i in range(4)]
    expected: list[RetrievalEvidence] = []
    # Reverse insertion order and lexical order across seeds: seed rank wins.
    for index, seed in enumerate(seeds):
        for suffix in ["c", "b", "a"]:
            path = f"{9 - index}-{suffix}.py"
            unit = _unit(session, repo, path)
            _edge(session, repo, seed, unit)
            if index < 3 and suffix != "c":
                expected.append(_dto(unit, path))
    direct = [_dto(unit, f"seed{i}.py") for i, unit in enumerate(seeds)]
    expected = [expected[i] for i in [1, 0, 3, 2, 5, 4]]
    actual = module._related_evidence(
        session,
        repository_id=repo.id,
        seeds=direct[:3],
        excluded_ids={unit.id for unit in seeds},
        filters=None,
    )
    assert actual == expected
    assert len(actual) == 6
    monkeypatch.setattr(module, "search_code_units_reranked", Mock(return_value=direct))
    result = search_code_units_with_dependencies(
        session,
        repository_id=repo.id,
        query="q",
        query_vector=[1.0],
        reranker=Mock(),
        limit=100,
    )
    assert result == direct + expected[:4]


def test_neighbor_filters_are_applied_before_cap(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(session)
    seed = _unit(session, repo, "seed.py")
    for path in ["a.py", "b.py", "z-match.py"]:
        target = _unit(session, repo, path)
        _edge(session, repo, seed, target)
    monkeypatch.setattr(
        module, "search_code_units_reranked", Mock(return_value=[_dto(seed, "seed.py")])
    )
    result = search_code_units_with_dependencies(
        session,
        repository_id=repo.id,
        query="q",
        query_vector=[1.0],
        reranker=Mock(),
        filters=RetrievalFilters(path="match", symbol="match", language="python"),
    )
    assert [item.path for item in result] == ["seed.py", "z-match.py"]


@pytest.mark.parametrize(
    "filters",
    [
        None,
        RetrievalFilters(path="neighbor"),
        RetrievalFilters(language="typescript"),
        RetrievalFilters(symbol="missing"),
    ],
)
def test_scoped_bounded_one_hop_sql(
    session: Session, monkeypatch: pytest.MonkeyPatch, filters: RetrievalFilters | None
) -> None:
    repo = _repo(session)
    seeds = [_unit(session, repo, f"seed{i}.py") for i in range(4)]
    neighbors = [_unit(session, repo, f"neighbor{i}.py") for i in range(5)]
    incoming = _unit(session, repo, "incoming.py")
    second_hop = _unit(session, repo, "second-hop.py")
    fourth_only = _unit(session, repo, "fourth-only.py")
    for seed in seeds[:3]:
        for neighbor in neighbors:
            _edge(session, repo, seed, neighbor)
    _edge(session, repo, seeds[0], seeds[0])
    _edge(session, repo, seeds[0], seeds[-1])
    _edge(session, repo, incoming, seeds[0])
    _edge(session, repo, neighbors[0], second_hop)
    _edge(session, repo, seeds[3], fourth_only)
    other = _repo(session)
    other_seed = _unit(session, other, "seed0.py")
    other_neighbor = _unit(session, other, "neighbor0.py")
    _edge(session, other, other_seed, other_neighbor)
    direct = [_dto(unit, f"seed{i}.py") for i, unit in enumerate(seeds)]
    monkeypatch.setattr(module, "search_code_units_reranked", Mock(return_value=direct))
    statements: list[str] = []

    def capture(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        statements.append(statement)

    connection = session.connection()
    event.listen(connection, "before_cursor_execute", capture)
    pending = Repository(name="must not flush", source=f"pending:{uuid4()}")
    session.add(pending)
    try:
        first = search_code_units_with_dependencies(
            session,
            repository_id=repo.id,
            query="q",
            query_vector=[1.0],
            reranker=Mock(),
            filters=filters,
        )
        second = search_code_units_with_dependencies(
            session,
            repository_id=repo.id,
            query="q",
            query_vector=[1.0],
            reranker=Mock(),
            filters=filters,
        )
    finally:
        event.remove(connection, "before_cursor_execute", capture)
    assert first == second
    expected = (
        []
        if filters and (filters.language or filters.symbol)
        else [_dto(unit, f"neighbor{i}.py") for i, unit in enumerate(neighbors[:2])]
    )
    assert first == direct + expected
    assert len(statements) == 2
    assert all(sql.lstrip().startswith("SELECT") for sql in statements)
    assert pending in session.new and pending.id is None
    assert session.in_transaction()
    # Even a caller supplying a foreign seed cannot cross the ownership boundary.
    assert (
        module._related_evidence(
            session,
            repository_id=repo.id,
            seeds=[_dto(other_seed, "seed0.py")],
            excluded_ids={other_seed.id},
            filters=None,
        )
        == []
    )
