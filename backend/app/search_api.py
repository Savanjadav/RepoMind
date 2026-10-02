import json
from collections.abc import Callable
from functools import lru_cache
from math import isfinite
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.database import get_db_session
from app.embedding_provider import EmbeddingProvider
from app.models.code_unit import EMBEDDING_DIMENSION
from app.models.repository import Repository
from app.redis_cache import RedisCache, get_redis_cache, search_key
from app.semantic_search import search_code_units_semantically

router = APIRouter()


class SemanticSearchItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    model_config = ConfigDict(extra="forbid")
    items: list[SemanticSearchItemResponse]
    limit: int


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    from app.sentence_transformer_embedding_provider import (
        SentenceTransformerEmbeddingProvider,
    )

    return SentenceTransformerEmbeddingProvider()


def get_search_provider_factory() -> Callable[[], EmbeddingProvider]:
    """Resolve the cached provider only on a miss, not in eager dependencies."""
    return get_embedding_provider


def _generation(session: Session, repository_id: UUID) -> int:
    # Scalar SQL deliberately bypasses the ORM identity map under READ COMMITTED.
    generation = session.scalar(
        select(Repository.index_generation).where(Repository.id == repository_id)
    )
    if generation is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    return generation


def _cached_response(
    cache: RedisCache, key: str, limit: int
) -> SemanticSearchResponse | None:
    value = cache.read(key)
    if value is None:
        return None
    try:
        result = SemanticSearchResponse.model_validate_json(
            json.dumps(value), strict=True
        )
    except (ValidationError, ValueError, RecursionError):
        return None
    if (
        result.limit != limit
        or len(result.items) > limit
        or len({item.code_unit_id for item in result.items}) != len(result.items)
        or any(
            not isfinite(item.cosine_distance)
            or item.start_line < 1
            or item.end_line < item.start_line
            for item in result.items
        )
    ):
        return None
    return result


@router.get("/search", response_model=SemanticSearchResponse)
def search(
    repository_id: UUID,
    q: Annotated[str, Query(min_length=1, max_length=2000)],
    session: Annotated[Session, Depends(get_db_session)],
    provider_factory: Annotated[
        Callable[[], EmbeddingProvider],
        Depends(get_search_provider_factory),
    ],
    cache: Annotated[RedisCache | None, Depends(get_redis_cache)],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> SemanticSearchResponse:
    if not q.strip():
        raise HTTPException(
            status_code=422,
            detail="Query must contain non-whitespace characters",
        )
    generation = _generation(session, repository_id)
    key = search_key(repository_id, generation, q, limit)
    if cache is not None:
        cached = _cached_response(cache, key, limit)
        if cached is not None:
            current = _generation(session, repository_id)
            if current == generation:
                return cached
            generation = current
            key = search_key(repository_id, generation, q, limit)

    embedding_provider = provider_factory()
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

    response = SemanticSearchResponse(
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
    if cache is not None and _generation(session, repository_id) == generation:
        cache.write(key, response.model_dump(mode="json"))
    return response
