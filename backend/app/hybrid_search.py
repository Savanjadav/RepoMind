from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.lexical_search import LexicalSearchResult, search_code_units_lexically
from app.semantic_search import SemanticSearchResult, search_code_units_semantically

RRF_K = 60
CANDIDATE_MULTIPLIER = 3


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
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


def search_code_units_hybrid(
    session: Session,
    *,
    repository_id: UUID,
    query: str,
    query_vector: Sequence[float],
    limit: int = 10,
) -> list[HybridSearchResult]:
    _validate_limit(limit)
    candidate_limit = min(limit * CANDIDATE_MULTIPLIER, 100)

    semantic_results = search_code_units_semantically(
        session,
        repository_id=repository_id,
        query_vector=query_vector,
        limit=candidate_limit,
    )
    lexical_results = search_code_units_lexically(
        session,
        repository_id=repository_id,
        query=query,
        limit=candidate_limit,
    )

    semantic_by_id = {
        result.code_unit_id: (rank, result)
        for rank, result in enumerate(semantic_results, start=1)
    }
    lexical_by_id = {
        result.code_unit_id: (rank, result)
        for rank, result in enumerate(lexical_results, start=1)
    }

    results = [
        _build_result(
            code_unit_id,
            semantic_by_id.get(code_unit_id),
            lexical_by_id.get(code_unit_id),
        )
        for code_unit_id in semantic_by_id.keys() | lexical_by_id.keys()
    ]
    results.sort(
        key=lambda result: (
            -result.hybrid_score,
            result.path,
            result.start_line,
            -result.end_line,
            result.kind.value,
            result.code_unit_id,
        )
    )
    return results[:limit]


def _build_result(
    code_unit_id: UUID,
    semantic_entry: tuple[int, SemanticSearchResult] | None,
    lexical_entry: tuple[int, LexicalSearchResult] | None,
) -> HybridSearchResult:
    semantic_rank, semantic_result = (
        semantic_entry if semantic_entry is not None else (None, None)
    )
    lexical_rank, lexical_result = (
        lexical_entry if lexical_entry is not None else (None, None)
    )
    metadata = semantic_result if semantic_result is not None else lexical_result
    if metadata is None:
        raise RuntimeError("Hybrid candidate has no retrieval metadata")

    hybrid_score = 0.0
    if semantic_rank is not None:
        hybrid_score += 1.0 / (RRF_K + semantic_rank)
    if lexical_rank is not None:
        hybrid_score += 1.0 / (RRF_K + lexical_rank)

    return HybridSearchResult(
        code_unit_id=code_unit_id,
        file_id=metadata.file_id,
        path=metadata.path,
        kind=metadata.kind,
        content=metadata.content,
        language=metadata.language,
        start_line=metadata.start_line,
        end_line=metadata.end_line,
        symbol_name=metadata.symbol_name,
        hybrid_score=hybrid_score,
        semantic_rank=semantic_rank,
        lexical_rank=lexical_rank,
        cosine_distance=(
            semantic_result.cosine_distance if semantic_result is not None else None
        ),
        lexical_score=(
            lexical_result.lexical_score if lexical_result is not None else None
        ),
    )


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Limit must be an integer between 1 and 100")
