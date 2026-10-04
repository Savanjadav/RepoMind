"""Metadata-only repository registration and bounded listing."""

import re
from datetime import datetime
from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session

from app.database import get_db_session
from app.models.repository import Repository
from app.repository_source import (
    RepositorySourceKind,
    RepositorySourceValidationError,
    validate_repository_source,
)

router = APIRouter()


class RepositoryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str = Field(min_length=1, max_length=2048)


class RepositoryResponse(BaseModel):
    id: UUID
    name: str
    source: str
    created_at: datetime


class RepositoryListResponse(BaseModel):
    items: list[RepositoryResponse]
    limit: int
    offset: int


def _canonical_source(source: str) -> tuple[str, str]:
    """Reject encoded/ambiguous paths rather than decoding a different identity."""
    validated = validate_repository_source(source)
    if validated.kind is not RepositorySourceKind.HTTPS:
        raise RepositorySourceValidationError("Unsupported registration source")
    parsed = urlsplit(validated.value)
    if parsed.hostname != "github.com" or parsed.port not in (None, 443):
        raise RepositorySourceValidationError("Unsupported registration source")
    match = re.fullmatch(r"/([A-Za-z0-9-]+)/([A-Za-z0-9_.-]+)/?", parsed.path)
    if match is None:
        raise RepositorySourceValidationError("Invalid repository path")
    owner, name = (part.lower() for part in match.groups())
    name = name.removesuffix(".git")
    if (
        not 1 <= len(owner) <= 39
        or owner.startswith("-")
        or owner.endswith("-")
        or "--" in owner
        or not 1 <= len(name) <= 100
        or name in {".", ".."}
    ):
        raise RepositorySourceValidationError("Invalid repository identity")
    return f"https://github.com/{owner}/{name}", name


def _database_error(error: DBAPIError | PoolTimeoutError) -> HTTPException:
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
    return HTTPException(
        503 if unavailable else 500,
        "Repository service unavailable"
        if unavailable
        else "Repository request failed",
    )


@router.post("/repositories", response_model=RepositoryResponse, status_code=201)
def register_repository(
    request: RepositoryCreate,
    session: Annotated[Session, Depends(get_db_session)],
) -> RepositoryResponse:
    try:
        source, name = _canonical_source(request.source)
    except RepositorySourceValidationError:
        raise HTTPException(422, "Use a valid GitHub HTTPS repository URL") from None
    try:
        repository = Repository(name=name, source=source)
        session.add(repository)
        session.flush()
        response = RepositoryResponse(
            id=repository.id,
            name=repository.name,
            source=repository.source,
            created_at=repository.created_at,
        )
        session.commit()
        return response
    except (DBAPIError, PoolTimeoutError) as error:
        session.rollback()
        if (
            isinstance(error, IntegrityError)
            and getattr(error.orig, "sqlstate", None) == "23505"
            and getattr(getattr(error.orig, "diag", None), "constraint_name", None)
            == "uq_repositories_source"
        ):
            raise HTTPException(409, "This repository is already registered") from None
        raise _database_error(error) from None


@router.get("/repositories", response_model=RepositoryListResponse)
def list_repositories(
    session: Annotated[Session, Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RepositoryListResponse:
    try:
        rows = session.execute(
            select(
                Repository.id, Repository.name, Repository.source, Repository.created_at
            )
            .order_by(Repository.created_at.desc(), Repository.id.desc())
            .limit(limit)
            .offset(offset)
        ).mappings()
        return RepositoryListResponse(
            items=[RepositoryResponse.model_validate(row) for row in rows],
            limit=limit,
            offset=offset,
        )
    except (DBAPIError, PoolTimeoutError) as error:
        raise _database_error(error) from None
