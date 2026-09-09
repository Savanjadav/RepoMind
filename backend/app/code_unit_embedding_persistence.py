from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit


@dataclass(frozen=True, slots=True)
class CodeUnitEmbedding:
    code_unit_id: UUID
    vector: Sequence[float]


def persist_code_unit_embeddings(
    session: Session,
    embeddings: Sequence[CodeUnitEmbedding],
) -> list[CodeUnit]:
    requested_embeddings = list(embeddings)
    if not requested_embeddings:
        return []

    requested_ids = [embedding.code_unit_id for embedding in requested_embeddings]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Duplicate CodeUnit ID")

    vectors = [_validate_vector(embedding.vector) for embedding in requested_embeddings]

    stored_units = session.scalars(
        select(CodeUnit).where(CodeUnit.id.in_(requested_ids))
    ).all()
    units_by_id = {unit.id: unit for unit in stored_units}
    if set(units_by_id) != set(requested_ids):
        raise ValueError("CodeUnit does not exist")

    ordered_units = [units_by_id[code_unit_id] for code_unit_id in requested_ids]
    for unit, vector in zip(ordered_units, vectors, strict=True):
        unit.embedding = vector

    session.flush()
    return ordered_units


def _validate_vector(vector: Sequence[float]) -> list[float]:
    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding vector must contain exactly {EMBEDDING_DIMENSION} values"
        )

    validated: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Embedding vector values must be real numbers")
        converted = float(value)
        if not isfinite(converted):
            raise ValueError("Embedding vector values must be finite")
        validated.append(converted)
    return validated
