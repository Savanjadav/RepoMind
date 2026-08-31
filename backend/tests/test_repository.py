import os
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import create_database_engine
from app.models import File, IndexingJob, Repository

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


@pytest.mark.skipif(DATABASE_URL is None, reason="DATABASE_URL is not configured")
def test_repository_owned_models_and_database_cascade() -> None:
    assert DATABASE_URL is not None
    engine = create_database_engine(DATABASE_URL)

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:
                repository = Repository(
                    name="RepoMind relationship test repository",
                    source=f"https://example.com/{uuid4()}.git",
                )
                file = File(path="src/main.py")
                indexing_job = IndexingJob()
                repository.files.append(file)
                repository.indexing_jobs.append(indexing_job)
                session.add(repository)
                session.flush()

                repository_id = repository.id
                file_id = file.id
                indexing_job_id = indexing_job.id

                assert file.repository_id == repository_id
                assert indexing_job.repository_id == repository_id
                assert file.path == "src/main.py"
                assert indexing_job.status == "pending"
                assert indexing_job.created_at is not None
                assert indexing_job.created_at.utcoffset() is not None
                assert file.repository is repository
                assert indexing_job.repository is repository
                assert repository.files == [file]
                assert repository.indexing_jobs == [indexing_job]

                session.expunge_all()

                stored_file = session.get(File, file_id)
                stored_indexing_job = session.get(IndexingJob, indexing_job_id)

                assert stored_file is not None
                assert stored_indexing_job is not None
                assert stored_file.repository.id == repository_id
                assert stored_indexing_job.repository.id == repository_id
                assert stored_file in stored_file.repository.files
                assert (
                    stored_indexing_job in stored_indexing_job.repository.indexing_jobs
                )

                session.expunge_all()

                repository_to_delete = session.get(Repository, repository_id)
                assert repository_to_delete is not None
                session.delete(repository_to_delete)
                session.flush()

                assert session.get(File, file_id) is None
                assert session.get(IndexingJob, indexing_job_id) is None
        finally:
            transaction.rollback()

    with Session(engine) as session:
        assert session.get(Repository, repository_id) is None
        assert session.get(File, file_id) is None
        assert session.get(IndexingJob, indexing_job_id) is None

    engine.dispose()
