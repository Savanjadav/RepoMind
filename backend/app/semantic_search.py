from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.code_parser import CodeUnitKind
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.repository import Repository


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
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


def search_code_units_semantically(
    session: Session,
    *,
    repository_id: UUID,
    query_vector: Sequence[float],
    limit: int = 10,
) -> list[SemanticSearchResult]:
    validated_query = _validate_query_vector(query_vector)
    _validate_limit(limit)

    if session.get(Repository, repository_id) is None:
        raise ValueError("Repository does not exist")

    zero_vector = [0.0] * EMBEDDING_DIMENSION
    cosine_distance = CodeUnit.embedding.cosine_distance(validated_query)
    statement = (
        select(
            CodeUnit.id,
            CodeUnit.file_id,
            File.path,
            CodeUnit.kind,
            CodeUnit.content,
            CodeUnit.language,
            CodeUnit.start_line,
            CodeUnit.end_line,
            CodeUnit.symbol_name,
            cosine_distance,
        )
        .join(File, CodeUnit.file_id == File.id)
        .where(
            File.repository_id == repository_id,
            CodeUnit.embedding.is_not(None),
            CodeUnit.embedding != zero_vector,
        )
        .order_by(
            cosine_distance.asc(),
            File.path.asc(),
            CodeUnit.start_line.asc(),
            CodeUnit.end_line.desc(),
            CodeUnit.kind.asc(),
            CodeUnit.id.asc(),
        )
        .limit(limit)
    )

    return [
        SemanticSearchResult(
            code_unit_id=code_unit_id,
            file_id=file_id,
            path=path,
            kind=CodeUnitKind(kind),
            content=content,
            language=language,
            start_line=start_line,
            end_line=end_line,
            symbol_name=symbol_name,
            cosine_distance=float(distance),
        )
        for (
            code_unit_id,
            file_id,
            path,
            kind,
            content,
            language,
            start_line,
            end_line,
            symbol_name,
            distance,
        ) in session.execute(statement)
    ]


def _validate_query_vector(query_vector: Sequence[float]) -> list[float]:
    values = list(query_vector)
    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Query vector must contain exactly {EMBEDDING_DIMENSION} values"
        )

    validated: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Query vector values must be real numbers")
        converted = float(value)
        if not isfinite(converted):
            raise ValueError("Query vector values must be finite")
        validated.append(converted)

    if all(value == 0.0 for value in validated):
        raise ValueError("Query vector must not be all zero")
    return validated


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Limit must be an integer between 1 and 100")
