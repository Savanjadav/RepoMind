"""Bounded metadata aggregates from one committed PostgreSQL observation."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session

from app.database import get_db_session
from app.indexing_api import LatestIndexingJob
from app.repository_api import _database_error

router = APIRouter()


class OverviewCounts(BaseModel):
    files: int = Field(ge=0)
    code_units: int = Field(ge=0)
    named_code_units: int = Field(ge=0)


class LanguageCount(BaseModel):
    language: Literal[
        "python", "javascript", "typescript", "documentation", "config", "other"
    ]
    count: int = Field(gt=0)


class KindCount(BaseModel):
    kind: Literal["class", "config", "document", "function", "import"]
    count: int = Field(gt=0)


class RepositoryOverview(BaseModel):
    repository_id: UUID
    index_generation: str = Field(pattern=r"^(0|[1-9][0-9]*)$")
    latest_job: LatestIndexingJob | None
    snapshot_counts: OverviewCounts
    code_units_by_language: list[LanguageCount]
    code_units_by_kind: list[KindCount]


# One parameterized statement, including existence/job/generation. PostgreSQL
# READ COMMITTED gives all subqueries the same statement snapshot; no row locks.
# Aggregate only metadata, never source content or vectors. Unknown languages
# are bucketed BEFORE grouping, so there can be only one "other" entry.
_OVERVIEW = text("""
WITH repository_files AS (
    SELECT id FROM files WHERE repository_id = :repository_id
), units AS (
    SELECT u.kind, u.symbol_name IS NOT NULL AS named,
           CASE WHEN u.language IN
               ('python', 'javascript', 'typescript', 'documentation', 'config')
                THEN u.language ELSE 'other' END AS language
    FROM code_units u JOIN repository_files f ON f.id = u.file_id
), languages AS (
    SELECT language, count(*) AS count FROM units GROUP BY language
), kinds AS (
    SELECT kind, count(*) AS count FROM units GROUP BY kind
)
SELECT r.id AS repository_id, r.index_generation::text AS index_generation,
       (SELECT jsonb_build_object('job_id', j.id, 'status', j.status,
                                 'created_at', j.created_at)
        FROM indexing_jobs j WHERE j.repository_id = r.id
        ORDER BY j.created_at DESC, j.id DESC LIMIT 1) AS latest_job,
       jsonb_build_object(
           'files', (SELECT count(*) FROM repository_files),
           'code_units', (SELECT count(*) FROM units),
           'named_code_units', (SELECT count(*) FROM units WHERE named)
       ) AS snapshot_counts,
       COALESCE((SELECT jsonb_agg(jsonb_build_object(
           'language', language, 'count', count) ORDER BY language)
           FROM languages), '[]'::jsonb) AS code_units_by_language,
       COALESCE((SELECT jsonb_agg(jsonb_build_object(
           'kind', kind, 'count', count) ORDER BY kind)
           FROM kinds), '[]'::jsonb) AS code_units_by_kind
FROM repositories r WHERE r.id = :repository_id
""")


@router.get("/repositories/{repository_id}/overview", response_model=RepositoryOverview)
def get_repository_overview(
    repository_id: UUID,
    session: Annotated[Session, Depends(get_db_session)],
) -> RepositoryOverview:
    try:
        row = (
            session.execute(_OVERVIEW, {"repository_id": repository_id})
            .mappings()
            .one_or_none()
        )
    except (DBAPIError, PoolTimeoutError) as error:
        raise _database_error(error) from None
    if row is None:
        raise HTTPException(404, "Repository not found")
    return RepositoryOverview.model_validate(row)
