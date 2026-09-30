"""In-process job reservation and execution; no durable delivery or recovery."""

import logging
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.database import get_application_session_factory
from app.embedding_provider import EmbeddingProvider
from app.models.indexing_job import IndexingJob
from app.models.repository import Repository
from app.repository_clone import RepositoryCloneError, _require_github_https_source
from app.repository_indexing import execute_reserved_indexing_job
from app.repository_source import (
    RepositorySourceValidationError,
    validate_repository_source,
)

logger = logging.getLogger(__name__)


class RepositoryNotFoundError(ValueError):
    pass


class IndexingConflictError(ValueError):
    pass


class InvalidIndexingSourceError(ValueError):
    pass


def reserve_indexing_job(session: Session, repository_id: UUID) -> IndexingJob:
    """Reserve under a row lock; the request caller must commit or roll back."""
    try:
        repository = session.scalar(
            select(Repository)
            .where(Repository.id == repository_id)
            .with_for_update(nowait=True)
            .execution_options(populate_existing=True)
        )
    except DBAPIError as error:
        if getattr(error.orig, "sqlstate", None) == "55P03":
            raise IndexingConflictError("Repository is busy") from error
        raise
    if repository is None:
        raise RepositoryNotFoundError("Repository not found")
    try:
        _require_github_https_source(validate_repository_source(repository.source))
    except (RepositorySourceValidationError, RepositoryCloneError) as error:
        raise InvalidIndexingSourceError(
            "Repository source is not supported"
        ) from error
    if (
        session.scalar(
            select(IndexingJob.id)
            .where(
                IndexingJob.repository_id == repository_id,
                IndexingJob.status.in_(("pending", "running")),
            )
            .limit(1)
        )
        is not None
    ):
        raise IndexingConflictError("Repository already has an active indexing job")
    job = IndexingJob(repository_id=repository_id, status="pending")
    session.add(job)
    session.flush()
    return job


def _claim_job(session: Session, job_id: UUID) -> UUID | None:
    repository_id = session.scalar(
        select(IndexingJob.repository_id).where(IndexingJob.id == job_id)
    )
    if repository_id is None:
        return None
    repository = session.scalar(
        select(Repository).where(Repository.id == repository_id).with_for_update()
    )
    job = session.scalar(
        select(IndexingJob)
        .where(IndexingJob.id == job_id, IndexingJob.repository_id == repository_id)
        .execution_options(populate_existing=True)
    )
    if repository is None or job is None or job.status != "pending":
        return None
    job.status = "running"
    session.commit()
    return repository_id


class _SerializedEmbeddingProvider:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._lock = Lock()

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            return self._provider.embed(texts)


_provider: EmbeddingProvider | None = None
_provider_lock = Lock()


def get_indexing_embedding_provider() -> EmbeddingProvider:
    global _provider
    with _provider_lock:
        if _provider is None:
            from app.sentence_transformer_embedding_provider import (
                SentenceTransformerEmbeddingProvider,
            )

            _provider = _SerializedEmbeddingProvider(
                SentenceTransformerEmbeddingProvider()
            )
        return _provider


def _mark_failed(job_id: UUID, repository_id: UUID) -> None:
    # A fresh transaction also covers an unusable execution Session. Never
    # overwrite terminal state, including a commit that succeeded remotely.
    try:
        with get_application_session_factory()() as session:
            session.execute(
                update(IndexingJob)
                .where(
                    IndexingJob.id == job_id,
                    IndexingJob.repository_id == repository_id,
                    IndexingJob.status == "running",
                )
                .values(status="failed")
            )
            session.commit()
    except Exception as error:
        logger.error(
            "Indexing failure status unavailable job=%s category=%s",
            job_id,
            type(error).__name__,
        )


def run_indexing_job(job_id: UUID) -> None:
    """Own independent Sessions and workspace, executing only a claimed job."""
    repository_id: UUID | None = None
    try:
        factory = get_application_session_factory()
        with factory() as session:
            repository_id = _claim_job(session, job_id)
        if repository_id is None:
            return

        provider = get_indexing_embedding_provider()
        with factory() as session:
            # Workspace cleanup must succeed BEFORE the snapshot commit too.
            with TemporaryDirectory(prefix="repomind-indexing-") as workspace:
                execute_reserved_indexing_job(
                    session,
                    repository_id=repository_id,
                    job_id=job_id,
                    embedding_provider=provider,
                    workspace_root=Path(workspace),
                )
            session.commit()
    except Exception as error:
        # Session contexts roll back before a separate failure update. This is
        # the background execution boundary, not a retry or error reclassification.
        logger.error("Indexing failed job=%s category=%s", job_id, type(error).__name__)
        if repository_id is not None:
            _mark_failed(job_id, repository_id)
