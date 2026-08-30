import os
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import create_database_engine
from app.models import Repository

DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.mark.skipif(DATABASE_URL is None, reason="DATABASE_URL is not configured")
def test_repository_can_be_persisted_and_queried() -> None:
    assert DATABASE_URL is not None
    engine = create_database_engine(DATABASE_URL)

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:
                repository = Repository(
                    name="RepoMind test repository",
                    source=f"https://example.com/{uuid4()}.git",
                )
                session.add(repository)
                session.flush()

                repository_id = repository.id
                session.expunge_all()

                stored_repository = session.scalar(
                    select(Repository).where(Repository.id == repository_id)
                )

                assert stored_repository is not None
                assert stored_repository.name == repository.name
                assert stored_repository.source == repository.source
                assert stored_repository.created_at is not None
        finally:
            transaction.rollback()

    with Session(engine) as session:
        assert session.get(Repository, repository_id) is None

    engine.dispose()
