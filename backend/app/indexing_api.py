import logging
from collections.abc import Iterator
from contextlib import nullcontext
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select, true
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
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.indexing_job import IndexingJob
from app.models.repository import Repository
from app.redis_cache import RedisCache, get_redis_cache

router = APIRouter()
logger = logging.getLogger(__name__)


class IndexingJobResponse(BaseModel):
    job_id: UUID
    repository_id: UUID
    status: Literal["pending", "running", "completed", "failed"]


class LatestIndexingJob(BaseModel):
    job_id: UUID
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime


class SnapshotCounts(BaseModel):
    files: int = Field(ge=0)
    code_units: int = Field(ge=0)


class IndexingSummary(BaseModel):
    repository_id: UUID
    latest_job: LatestIndexingJob | None
    snapshot_counts: SnapshotCounts | None


class IndexingSummaryResponse(BaseModel):
    items: list[IndexingSummary]


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


@router.get("/repositories/indexing-summary", response_model=IndexingSummaryResponse)
def get_indexing_summaries(
    repository_id: Annotated[list[UUID], Query(min_length=1, max_length=100)],
    session: Annotated[Session, Depends(get_indexing_session)],
) -> IndexingSummaryResponse:
    ids = sorted(set(repository_id))
    latest = (
        select(IndexingJob.id, IndexingJob.status, IndexingJob.created_at)
        .where(IndexingJob.repository_id == Repository.id)
        .order_by(IndexingJob.created_at.desc(), IndexingJob.id.desc())
        .limit(1)
        .correlate(Repository)
        .lateral("latest_job")
    )
    files = (
        select(func.count(File.id))
        .where(File.repository_id == Repository.id)
        .correlate(Repository)
        .scalar_subquery()
    )
    units = (
        select(func.count(CodeUnit.id))
        .join(File, CodeUnit.file_id == File.id)
        .where(File.repository_id == Repository.id)
        .correlate(Repository)
        .scalar_subquery()
    )
    # One statement observes job state and counts in the same committed snapshot.
    # Separate aggregates prevent File x CodeUnit count multiplication. No ORM
    # entities/content/vectors are loaded, and active runs do not expose old counts.
    statement = (
        select(
            Repository.id.label("repository_id"),
            latest.c.id.label("job_id"),
            latest.c.status,
            latest.c.created_at,
            case((latest.c.status == "completed", files)).label("files"),
            case((latest.c.status == "completed", units)).label("code_units"),
        )
        .outerjoin(latest, true())
        .where(Repository.id.in_(ids))
        .order_by(Repository.id)
    )
    try:
        rows = session.execute(statement).mappings().all()
    except (DBAPIError, PoolTimeoutError) as error:
        raise _request_error(error) from None
    if len(rows) != len(ids):
        raise HTTPException(404, "Repository not found")
    return IndexingSummaryResponse(
        items=[
            IndexingSummary(
                repository_id=row["repository_id"],
                latest_job=(
                    LatestIndexingJob(
                        job_id=row["job_id"],
                        status=row["status"],
                        created_at=row["created_at"],
                    )
                    if row["job_id"] is not None
                    else None
                ),
                snapshot_counts=(
                    SnapshotCounts(files=row["files"], code_units=row["code_units"])
                    if row["status"] == "completed"
                    else None
                ),
            )
            for row in rows
        ]
    )


@router.post(
    "/repositories/{repository_id}/index",
    response_model=IndexingJobResponse,
    status_code=202,
)
def start_indexing(
    repository_id: UUID,
    session: Annotated[Session, Depends(get_indexing_session)],
    cache: Annotated[RedisCache | None, Depends(get_redis_cache)],
) -> JSONResponse:
    # Only the short reservation transaction gets an advisory lease. PostgreSQL
    # remains authoritative after expiry, eviction, or a Redis outage.
    with (
        cache.reservation(repository_id)
        if cache is not None
        else nullcontext(True) as allowed
    ):
        if not allowed:
            raise HTTPException(
                409, "Repository already has active indexing or is busy"
            )
        return _reserve_response(repository_id, session)


def _reserve_response(repository_id: UUID, session: Session) -> JSONResponse:
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
