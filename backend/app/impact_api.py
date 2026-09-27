"""Read-only API for persisted, one-hop impact hints."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.database import get_db_session
from app.impact_analysis import (
    DEFAULT_IMPACT_LIMIT,
    MAX_IMPACT_RESULTS,
    AmbiguousImpactTargetError,
    ImpactNotFoundError,
    ImpactValidationError,
    analyze_impact,
)

router = APIRouter()


class ImpactTargetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    file_id: UUID
    path: str
    code_unit_id: UUID | None
    symbol_name: str | None
    kind: str | None
    start_line: int | None
    end_line: int | None


class ImpactItemResponse(ImpactTargetResponse):
    relationship_type: Literal["imports", "calls"]


class ImpactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    repository_id: UUID
    mode: Literal["file", "symbol"]
    target: ImpactTargetResponse
    items: list[ImpactItemResponse]
    limit: int
    truncated: bool
    analysis: Literal["static_hint"]


@router.get("/repositories/{repository_id}/impact", response_model=ImpactResponse)
def get_repository_impact(
    repository_id: UUID,
    session: Annotated[Session, Depends(get_db_session)],
    path: Annotated[str, Query(min_length=1, max_length=2000)],
    symbol: Annotated[str | None, Query(min_length=1, max_length=2000)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_IMPACT_RESULTS)] = DEFAULT_IMPACT_LIMIT,
) -> ImpactResponse:
    try:
        result = analyze_impact(
            session, repository_id, path=path, symbol=symbol, limit=limit
        )
    except ImpactValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except ImpactNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except AmbiguousImpactTargetError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return ImpactResponse.model_validate(result)
