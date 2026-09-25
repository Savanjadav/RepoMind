import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import create_database_engine
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


def _files(session: Session) -> tuple[Repository, File, File]:
    repository = Repository(
        name="relationships", source=f"https://example.com/{uuid4()}"
    )
    session.add(repository)
    session.flush()
    source = File(repository_id=repository.id, path="a.py")
    target = File(repository_id=repository.id, path="b.py")
    session.add_all([source, target])
    session.flush()
    return repository, source, target


@pytest.mark.parametrize(
    "invalid",
    ["type", "self", "missing", "foreign_source", "foreign_target", "duplicate"],
)
def test_constraints(session: Session, invalid: str) -> None:
    repo, source, target = _files(session)
    _, foreign, _ = _files(session)
    kwargs = dict(
        repository_id=repo.id,
        source_file_id=source.id,
        target_file_id=target.id,
        relationship_type="imports",
    )
    if invalid == "duplicate":
        session.add(Relationship(**kwargs))
        session.flush()
    elif invalid == "type":
        kwargs["relationship_type"] = "calls"
    elif invalid == "self":
        kwargs["target_file_id"] = source.id
    elif invalid == "missing":
        kwargs["target_file_id"] = uuid4()
    elif invalid == "foreign_source":
        kwargs["source_file_id"] = foreign.id
    else:
        kwargs["target_file_id"] = foreign.id
    with pytest.raises(IntegrityError), session.begin_nested():
        session.add(Relationship(**kwargs))
        session.flush()


@pytest.mark.parametrize("endpoint", ["source", "target", "repository"])
def test_cascade_and_savepoint_rollback(session: Session, endpoint: str) -> None:
    repo, source, target = _files(session)
    edge = Relationship(
        repository_id=repo.id,
        source_file_id=source.id,
        target_file_id=target.id,
        relationship_type="imports",
    )
    session.add(edge)
    session.flush()
    query = select(Relationship.id).where(Relationship.repository_id == repo.id)
    assert session.scalar(query) == edge.id
    nested = session.begin_nested()
    if endpoint == "repository":
        session.execute(delete(Repository).where(Repository.id == repo.id))
    else:
        session.execute(
            delete(File).where(
                File.id == (source.id if endpoint == "source" else target.id)
            )
        )
    assert session.scalar(query) is None
    nested.rollback()
    assert session.scalar(query) == edge.id
