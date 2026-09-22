from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db_session
from app.grounded_answer import generate_grounded_answer
from app.llm_provider import (
    LLMProvider,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.models.code_unit import EMBEDDING_DIMENSION
from app.models.repository import Repository
from app.rag_context import ContextEvidence, format_context
from app.reranked_search import search_code_units_reranked
from app.reranking_provider import RerankingProvider
from app.search_api import get_embedding_provider

ASK_CONTEXT_MAX_CHARACTERS = 8_000
_DEFAULT_LLM_MODEL = "qwen2.5-coder:3b"

router = APIRouter()


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_id: UUID
    q: Annotated[str, Field(strict=True, min_length=1, max_length=2000)]
    limit: Annotated[int, Field(strict=True, ge=1, le=100)] = 10

    @field_validator("q")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must contain non-whitespace characters")
        if any(
            (ord(character) <= 0x1F and character not in "\t\n\r")
            or ord(character) == 0x7F
            for character in value
        ):
            raise ValueError("Question must not contain prohibited ASCII controls")
        return value


class AskResponse(BaseModel):
    answer: str


@lru_cache(maxsize=1)
def get_reranking_provider() -> RerankingProvider:
    from app.cross_encoder_reranking_provider import CrossEncoderRerankingProvider

    return CrossEncoderRerankingProvider()


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    from app.ollama_llm_provider import OllamaLLMProvider

    return OllamaLLMProvider(model_name=_DEFAULT_LLM_MODEL)


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    session: Annotated[Session, Depends(get_db_session)],
    reranker: Annotated[RerankingProvider, Depends(get_reranking_provider)],
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> AskResponse:
    repository = session.get(Repository, request.repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found")

    embedding_provider = get_embedding_provider()
    if embedding_provider.dimension != EMBEDDING_DIMENSION:
        raise RuntimeError("Embedding provider dimension does not match storage")
    vectors = embedding_provider.embed([request.q])
    if len(vectors) != 1:
        raise RuntimeError("Embedding provider returned an unexpected vector count")

    results = search_code_units_reranked(
        session,
        repository_id=request.repository_id,
        query=request.q,
        query_vector=vectors[0],
        reranker=reranker,
        limit=request.limit,
        filters=None,
    )
    evidence = [
        ContextEvidence(
            repository_name=repository.name,
            path=result.path,
            symbol_name=result.symbol_name,
            start_line=result.start_line,
            end_line=result.end_line,
            content=result.content,
        )
        for result in results
    ]
    context = format_context(evidence, max_characters=ASK_CONTEXT_MAX_CHARACTERS)
    try:
        response = generate_grounded_answer(
            llm_provider, question=request.q, context=context
        )
    except LLMUnavailableError as error:
        raise HTTPException(
            status_code=503, detail="Language model service unavailable"
        ) from error
    except LLMTimeoutError as error:
        raise HTTPException(
            status_code=504, detail="Language model request timed out"
        ) from error
    except LLMResponseError as error:
        raise HTTPException(
            status_code=502, detail="Language model returned an invalid response"
        ) from error
    except LLMProviderError as error:
        raise HTTPException(
            status_code=502, detail="Language model request failed"
        ) from error
    return AskResponse(answer=response.content)
