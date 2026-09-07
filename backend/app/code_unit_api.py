from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.database import get_db_session
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.repository import Repository

router = APIRouter()


class CodeUnitListItem(BaseModel):
    id: UUID
    file_id: UUID
    path: str
    kind: CodeUnitKind
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None


class CodeUnitListResponse(BaseModel):
    items: list[CodeUnitListItem]
    limit: int
    offset: int


@router.get(
    "/repositories/{repository_id}/code-units",
    response_model=CodeUnitListResponse,
)
def list_repository_code_units(
    repository_id: UUID,
    session: Annotated[Session, Depends(get_db_session)],
    kind: CodeUnitKind | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CodeUnitListResponse:
    if session.get(Repository, repository_id) is None:
        raise HTTPException(status_code=404, detail="Repository not found")

    statement = (
        select(
            CodeUnit.id,
            CodeUnit.file_id,
            File.path,
            CodeUnit.kind,
            CodeUnit.language,
            CodeUnit.start_line,
            CodeUnit.end_line,
            CodeUnit.symbol_name,
        )
        .join(File, CodeUnit.file_id == File.id)
        .where(File.repository_id == repository_id)
    )
    if kind is not None:
        statement = statement.where(CodeUnit.kind == kind.value)

    statement = (
        statement.order_by(
            File.path.asc(),
            CodeUnit.start_line.asc(),
            CodeUnit.end_line.desc(),
            CodeUnit.kind.asc(),
            CodeUnit.id.asc(),
        )
        .offset(offset)
        .limit(limit)
    )

    items = [
        CodeUnitListItem(
            id=unit_id,
            file_id=file_id,
            path=path,
            kind=CodeUnitKind(stored_kind),
            language=language,
            start_line=start_line,
            end_line=end_line,
            symbol_name=symbol_name,
        )
        for (
            unit_id,
            file_id,
            path,
            stored_kind,
            language,
            start_line,
            end_line,
            symbol_name,
        ) in session.execute(statement)
    ]
    return CodeUnitListResponse(items=items, limit=limit, offset=offset)
