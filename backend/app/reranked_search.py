from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from uuid import UUID

from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.hybrid_search import HybridSearchResult, search_code_units_hybrid
from app.reranking_provider import RerankerUnavailableError, RerankingProvider
from app.retrieval_filters import RetrievalFilters

RERANK_CANDIDATE_MULTIPLIER = 2
MAX_RERANK_CANDIDATES = 100


@dataclass(frozen=True, slots=True)
class RerankedSearchResult:
    code_unit_id: UUID
    file_id: UUID
    path: str
    kind: CodeUnitKind
    content: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None
    hybrid_score: float
    semantic_rank: int | None
    lexical_rank: int | None
    cosine_distance: float | None
    lexical_score: int | None
    rerank_score: float | None
    original_hybrid_rank: int


def search_code_units_reranked(
    session: Session,
    *,
    repository_id: UUID,
    query: str,
    query_vector: Sequence[float],
    reranker: RerankingProvider,
    limit: int = 10,
    filters: RetrievalFilters | None = None,
) -> list[RerankedSearchResult]:
    _validate_limit(limit)
    candidate_limit = min(
        limit * RERANK_CANDIDATE_MULTIPLIER,
        MAX_RERANK_CANDIDATES,
    )
    candidates = search_code_units_hybrid(
        session,
        repository_id=repository_id,
        query=query,
        query_vector=query_vector,
        limit=candidate_limit,
        filters=filters,
    )
    if not candidates:
        return []

    try:
        scores = reranker.score(
            query,
            [_build_document(candidate) for candidate in candidates],
        )
    except RerankerUnavailableError:
        return [
            _build_result(candidate, rank=rank, rerank_score=None)
            for rank, candidate in enumerate(candidates[:limit], start=1)
        ]

    validated_scores = _validate_scores(scores, expected_count=len(candidates))
    results = [
        _build_result(candidate, rank=rank, rerank_score=score)
        for rank, (candidate, score) in enumerate(
            zip(candidates, validated_scores, strict=True),
            start=1,
        )
    ]
    results.sort(
        key=lambda result: (
            -_required_score(result.rerank_score),
            result.original_hybrid_rank,
        )
    )
    return results[:limit]


def _build_document(candidate: HybridSearchResult) -> str:
    fields = [f"Path: {candidate.path}"]
    if candidate.symbol_name is not None:
        fields.append(f"Symbol: {candidate.symbol_name}")
    return "\n".join(fields) + f"\n\nContent:\n{candidate.content}"


def _validate_scores(scores: Sequence[float], *, expected_count: int) -> list[float]:
    values = list(scores)
    if len(values) != expected_count:
        raise RuntimeError("Reranker returned an unexpected score count")

    validated: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError("Reranker scores must be real numbers")
        converted = float(value)
        if not isfinite(converted):
            raise RuntimeError("Reranker scores must be finite")
        validated.append(converted)
    return validated


def _build_result(
    candidate: HybridSearchResult,
    *,
    rank: int,
    rerank_score: float | None,
) -> RerankedSearchResult:
    return RerankedSearchResult(
        code_unit_id=candidate.code_unit_id,
        file_id=candidate.file_id,
        path=candidate.path,
        kind=candidate.kind,
        content=candidate.content,
        language=candidate.language,
        start_line=candidate.start_line,
        end_line=candidate.end_line,
        symbol_name=candidate.symbol_name,
        hybrid_score=candidate.hybrid_score,
        semantic_rank=candidate.semantic_rank,
        lexical_rank=candidate.lexical_rank,
        cosine_distance=candidate.cosine_distance,
        lexical_score=candidate.lexical_score,
        rerank_score=rerank_score,
        original_hybrid_rank=rank,
    )


def _required_score(score: float | None) -> float:
    if score is None:
        raise RuntimeError("Reranked result is missing a score")
    return score


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Limit must be an integer between 1 and 100")
