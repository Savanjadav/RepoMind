from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.database import get_db_session
from app.embedding_provider import EmbeddingProvider
from app.models.code_unit import EMBEDDING_DIMENSION
from app.semantic_search import search_code_units_semantically

router = APIRouter()


class SemanticSearchItemResponse(BaseModel):
    code_unit_id: UUID
    file_id: UUID
    path: str
    kind: CodeUnitKind
    content: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None
    cosine_distance: float


class SemanticSearchResponse(BaseModel):
    items: list[SemanticSearchItemResponse]
    limit: int


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    from app.sentence_transformer_embedding_provider import (
        SentenceTransformerEmbeddingProvider,
    )

    return SentenceTransformerEmbeddingProvider()


@router.get("/search", response_model=SemanticSearchResponse)
def search(
    repository_id: UUID,
    q: Annotated[str, Query(min_length=1, max_length=2000)],
    session: Annotated[Session, Depends(get_db_session)],
    embedding_provider: Annotated[
        EmbeddingProvider,
        Depends(get_embedding_provider),
    ],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> SemanticSearchResponse:
    if not q.strip():
        raise HTTPException(
            status_code=422,
            detail="Query must contain non-whitespace characters",
        )
    if embedding_provider.dimension != EMBEDDING_DIMENSION:
        raise RuntimeError("Embedding provider dimension does not match storage")

    vectors = embedding_provider.embed([q])
    if len(vectors) != 1:
        raise RuntimeError("Embedding provider returned an unexpected vector count")

    try:
        results = search_code_units_semantically(
            session,
            repository_id=repository_id,
            query_vector=vectors[0],
            limit=limit,
        )
    except ValueError as error:
        if error.args == ("Repository does not exist",):
            raise HTTPException(
                status_code=404, detail="Repository not found"
            ) from error
        raise

    return SemanticSearchResponse(
        items=[
            SemanticSearchItemResponse(
                code_unit_id=result.code_unit_id,
                file_id=result.file_id,
                path=result.path,
                kind=result.kind,
                content=result.content,
                language=result.language,
                start_line=result.start_line,
                end_line=result.end_line,
                symbol_name=result.symbol_name,
                cosine_distance=result.cosine_distance,
            )
            for result in results
        ],
        limit=limit,
    )
