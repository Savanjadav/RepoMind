"""Bounded outgoing-call evidence supplements, without graph relevance scores."""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import literal, select, union_all
from sqlalchemy.orm import Session, aliased

from app.code_parser import CodeUnitKind
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.reranked_search import search_code_units_reranked
from app.reranking_provider import RerankingProvider
from app.retrieval_filters import RetrievalFilters, build_retrieval_filter_predicates

MAX_EXPANSION_SEEDS = 3
MAX_RELATED_PER_SEED = 2
MAX_GRAPH_ADDED_RESULTS = 4
MAX_FINAL_RESULTS = 100


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    code_unit_id: UUID
    file_id: UUID
    path: str
    kind: CodeUnitKind
    content: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None


def search_code_units_with_dependencies(
    session: Session,
    *,
    repository_id: UUID,
    query: str,
    query_vector: Sequence[float],
    reranker: RerankingProvider,
    limit: int = 10,
    filters: RetrievalFilters | None = None,
) -> list[RetrievalEvidence]:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_FINAL_RESULTS
    ):
        raise ValueError("Limit must be an integer between 1 and 100")
    with session.no_autoflush:
        ranked = search_code_units_reranked(
            session,
            repository_id=repository_id,
            query=query,
            query_vector=query_vector,
            reranker=reranker,
            limit=limit,
            filters=filters,
        )
        direct: list[RetrievalEvidence] = []
        seen: set[UUID] = set()
        for item in ranked:
            if item.code_unit_id in seen:
                continue
            seen.add(item.code_unit_id)
            direct.append(
                RetrievalEvidence(
                    item.code_unit_id,
                    item.file_id,
                    item.path,
                    item.kind,
                    item.content,
                    item.language,
                    item.start_line,
                    item.end_line,
                    item.symbol_name,
                )
            )
        allowance = min(MAX_GRAPH_ADDED_RESULTS, limit // 3)
        if not direct or not allowance:
            return direct
        seeds = direct[: min(MAX_EXPANSION_SEEDS, limit - allowance)]
        neighbors = _related_evidence(
            session,
            repository_id=repository_id,
            seeds=seeds,
            excluded_ids=set(seen),
            filters=filters,
        )
        additions: list[RetrievalEvidence] = []
        for neighbor in neighbors:
            if neighbor.code_unit_id in seen:
                continue
            seen.add(neighbor.code_unit_id)
            additions.append(neighbor)
            if len(additions) == allowance:
                break
        return direct[: limit - len(additions)] + additions


def _related_evidence(
    session: Session,
    *,
    repository_id: UUID,
    seeds: Sequence[RetrievalEvidence],
    excluded_ids: set[UUID],
    filters: RetrievalFilters | None,
) -> list[RetrievalEvidence]:
    source_file = aliased(File)
    source_unit = aliased(CodeUnit)
    branches = []
    for rank, seed in enumerate(seeds):
        branches.append(
            select(
                literal(rank).label("seed_rank"),
                CodeUnit.id.label("code_unit_id"),
                CodeUnit.file_id,
                File.path,
                CodeUnit.kind,
                CodeUnit.content,
                CodeUnit.language,
                CodeUnit.start_line,
                CodeUnit.end_line,
                CodeUnit.symbol_name,
            )
            .select_from(CodeRelationship)
            .join(source_file, source_file.id == CodeRelationship.source_file_id)
            .join(
                source_unit,
                (source_unit.id == CodeRelationship.source_code_unit_id)
                & (source_unit.file_id == source_file.id),
            )
            .join(File, File.id == CodeRelationship.target_file_id)
            .join(
                CodeUnit,
                (CodeUnit.id == CodeRelationship.target_code_unit_id)
                & (CodeUnit.file_id == File.id),
            )
            .where(
                CodeRelationship.repository_id == repository_id,
                source_file.repository_id == repository_id,
                File.repository_id == repository_id,
                source_unit.id == seed.code_unit_id,
                source_file.id == seed.file_id,
                CodeRelationship.relationship_type == "calls",
                CodeUnit.id != source_unit.id,
                CodeUnit.id.not_in(excluded_ids),
                *build_retrieval_filter_predicates(filters),
            )
            .order_by(
                File.path,
                CodeUnit.start_line,
                CodeUnit.end_line.desc(),
                CodeUnit.kind,
                CodeUnit.id,
            )
            .limit(MAX_RELATED_PER_SEED)
        )
    candidates = union_all(*branches).subquery()
    statement = select(candidates).order_by(
        candidates.c.seed_rank,
        candidates.c.path,
        candidates.c.start_line,
        candidates.c.end_line.desc(),
        candidates.c.kind,
        candidates.c.code_unit_id,
    )
    return [
        RetrievalEvidence(
            row.code_unit_id,
            row.file_id,
            row.path,
            CodeUnitKind(row.kind),
            row.content,
            row.language,
            row.start_line,
            row.end_line,
            row.symbol_name,
        )
        for row in session.execute(statement)
    ]
