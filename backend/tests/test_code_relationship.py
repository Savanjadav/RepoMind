import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import create_database_engine
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import CodeUnit
from app.models.file import File
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


def _endpoints(session: Session) -> tuple[Repository, CodeUnit, CodeUnit]:
    repo = Repository(name="calls", source=f"https://example.com/{uuid4()}")
    session.add(repo)
    session.flush()
    units = []
    for name in ("source", "target"):
        file = File(repository_id=repo.id, path=f"{name}.py")
        session.add(file)
        session.flush()
        unit = CodeUnit(
            file_id=file.id,
            kind="function",
            symbol_name=name,
            content=f"def {name}(): pass",
            language="python",
            start_line=1,
            end_line=1,
        )
        session.add(unit)
        session.flush()
        units.append(unit)
    return repo, units[0], units[1]


def _edge(repo: Repository, source: CodeUnit, target: CodeUnit) -> CodeRelationship:
    return CodeRelationship(
        repository_id=repo.id,
        source_file_id=source.file_id,
        source_code_unit_id=source.id,
        target_file_id=target.file_id,
        target_code_unit_id=target.id,
        relationship_type="calls",
    )


@pytest.mark.parametrize("recursive", [False, True])
def test_valid_edge_and_caller_savepoint_ownership(
    session: Session, recursive: bool
) -> None:
    repo, source, target = _endpoints(session)
    savepoint = session.begin_nested()
    edge = _edge(repo, source, source if recursive else target)
    session.add(edge)
    session.flush()
    query = select(CodeRelationship.id).where(CodeRelationship.repository_id == repo.id)
    assert session.scalar(query) == edge.id
    savepoint.rollback()
    assert session.scalar(query) is None


@pytest.mark.parametrize(
    "invalid",
    [
        "type",
        "duplicate",
        "repository",
        "source_file",
        "target_file",
        "source_unit",
        "target_unit",
        "source_mismatch",
        "target_mismatch",
        "foreign_source",
        "foreign_target",
        "foreign_repository",
    ],
)
def test_constraints(session: Session, invalid: str) -> None:
    repo, source, target = _endpoints(session)
    foreign_repo, foreign, _ = _endpoints(session)
    edge = _edge(repo, source, target)
    if invalid == "type":
        edge.relationship_type = "imports"
    elif invalid == "duplicate":
        session.add(_edge(repo, source, target))
        session.flush()
    elif invalid == "repository":
        edge.repository_id = uuid4()
    elif invalid == "foreign_repository":
        edge.repository_id = foreign_repo.id
    elif invalid.endswith("mismatch"):
        if invalid.startswith("source"):
            edge.source_code_unit_id = target.id
        else:
            edge.target_code_unit_id = source.id
    elif invalid.startswith("foreign"):
        if invalid == "foreign_source":
            edge.source_file_id, edge.source_code_unit_id = foreign.file_id, foreign.id
        else:
            edge.target_file_id, edge.target_code_unit_id = foreign.file_id, foreign.id
    else:
        attribute = invalid.replace("unit", "code_unit") + "_id"
        setattr(edge, attribute, uuid4())
    with pytest.raises(IntegrityError), session.begin_nested():
        session.add(edge)
        session.flush()


@pytest.mark.parametrize(
    "deleted",
    ["source_unit", "target_unit", "source_file", "target_file", "repository"],
)
def test_cascades(session: Session, deleted: str) -> None:
    repo, source, target = _endpoints(session)
    edge = _edge(repo, source, target)
    session.add(edge)
    session.flush()
    query = select(CodeRelationship.id).where(CodeRelationship.repository_id == repo.id)
    assert session.scalar(query) == edge.id
    savepoint = session.begin_nested()
    if deleted == "repository":
        session.execute(delete(Repository).where(Repository.id == repo.id))
    elif deleted.endswith("file"):
        session.execute(
            delete(File).where(
                File.id
                == (source.file_id if deleted.startswith("source") else target.file_id)
            )
        )
    else:
        session.execute(
            delete(CodeUnit).where(
                CodeUnit.id
                == (source.id if deleted.startswith("source") else target.id)
            )
        )
    assert session.scalar(query) is None
    savepoint.rollback()
    assert session.scalar(query) == edge.id
