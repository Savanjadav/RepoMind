import logging
from collections.abc import Iterator
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session

from app.database import get_application_session_factory
from app.indexing_jobs import (
    IndexingConflictError,
    InvalidIndexingSourceError,
    RepositoryNotFoundError,
    reserve_indexing_job,
    run_indexing_job,
)
from app.models.indexing_job import IndexingJob

router = APIRouter()
logger = logging.getLogger(__name__)


class IndexingJobResponse(BaseModel):
    job_id: UUID
    repository_id: UUID
    status: Literal["pending", "running", "completed", "failed"]


def _request_error(error: Exception) -> HTTPException:
    if isinstance(error, RepositoryNotFoundError):
        return HTTPException(404, "Repository not found")
    if isinstance(error, InvalidIndexingSourceError):
        return HTTPException(422, "Repository source is not supported")
    if isinstance(error, IndexingConflictError):
        return HTTPException(409, "Repository already has active indexing or is busy")
    unavailable = isinstance(error, PoolTimeoutError)
    if isinstance(error, DBAPIError):
        code = getattr(error.orig, "sqlstate", None)
        unavailable = (
            error.connection_invalidated
            or (
                isinstance(code, str)
                and (code.startswith("08") or code in {"57P01", "57P02", "57P03"})
            )
            or (isinstance(error, OperationalError) and code is None)
        )
    logger.error("Indexing request failed category=%s", type(error).__name__)
    return HTTPException(
        503 if unavailable else 500,
        "Database unavailable" if unavailable else "Indexing request failed",
    )


def get_indexing_session() -> Iterator[Session]:
    try:
        session = get_application_session_factory()()
    except Exception as error:
        raise _request_error(error) from None
    try:
        yield session
    finally:
        session.close()


@router.post(
    "/repositories/{repository_id}/index",
    response_model=IndexingJobResponse,
    status_code=202,
)
def start_indexing(
    repository_id: UUID,
    session: Annotated[Session, Depends(get_indexing_session)],
) -> JSONResponse:
    try:
        job = reserve_indexing_job(session, repository_id)
        body = IndexingJobResponse(
            job_id=job.id, repository_id=repository_id, status="pending"
        )
        tasks = BackgroundTasks()
        tasks.add_task(run_indexing_job, job.id)
        # This response owns its tasks explicitly. Neither tasks nor response
        # are returned on commit failure; no dependency-injected task can leak.
        response = JSONResponse(
            status_code=202,
            content=body.model_dump(mode="json"),
            headers={
                "Location": f"/repositories/{repository_id}/indexing-jobs/{job.id}"
            },
            background=tasks,
        )
        session.commit()
        return response
    except Exception as error:
        try:
            session.rollback()
        except Exception:
            logger.error("Indexing reservation rollback unavailable")
        raise _request_error(error) from None


@router.get(
    "/repositories/{repository_id}/indexing-jobs/{job_id}",
    response_model=IndexingJobResponse,
)
def get_indexing_job(
    repository_id: UUID,
    job_id: UUID,
    session: Annotated[Session, Depends(get_indexing_session)],
) -> IndexingJobResponse:
    try:
        job = session.scalar(
            select(IndexingJob).where(
                IndexingJob.id == job_id, IndexingJob.repository_id == repository_id
            )
        )
        if job is None:
            raise HTTPException(404, "Indexing job not found")
        return IndexingJobResponse.model_validate(
            {"job_id": job.id, "repository_id": job.repository_id, "status": job.status}
        )
    except HTTPException:
        raise
    except Exception as error:
        raise _request_error(error) from None
