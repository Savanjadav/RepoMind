"""Bounded incoming dependency hints from persisted graph edges only."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from app.models.code_relationship import CodeRelationship
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.models.relationship import Relationship
from app.models.repository import Repository

DEFAULT_IMPACT_LIMIT = 20
MAX_IMPACT_RESULTS = 100


class ImpactValidationError(ValueError):
    """Invalid impact lookup input."""


class ImpactNotFoundError(LookupError):
    """The requested persisted target does not exist."""


class AmbiguousImpactTargetError(LookupError):
    """Multiple persisted units have the exact requested symbol."""


@dataclass(frozen=True)
class ImpactTarget:
    file_id: UUID
    path: str
    code_unit_id: UUID | None = None
    symbol_name: str | None = None
    kind: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class ImpactItem(ImpactTarget):
    relationship_type: Literal["imports", "calls"] = "imports"


@dataclass(frozen=True)
class ImpactResult:
    repository_id: UUID
    mode: Literal["file", "symbol"]
    target: ImpactTarget
    items: tuple[ImpactItem, ...]
    limit: int
    truncated: bool
    analysis: Literal["static_hint"] = "static_hint"


def _validate(path: str, symbol: str | None, limit: int) -> None:
    if (
        not isinstance(path, str)
        or not path
        or len(path) > 2000
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
    ):
        raise ImpactValidationError("path must be a valid relative POSIX path")
    if symbol is not None and (
        not isinstance(symbol, str)
        or not symbol.strip()
        or len(symbol) > 2000
        or any(ord(char) < 32 or ord(char) == 127 for char in symbol)
    ):
        raise ImpactValidationError("symbol must be a nonempty exact symbol name")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_IMPACT_RESULTS
    ):
        raise ImpactValidationError("limit must be an integer between 1 and 100")


def analyze_impact(
    session: Session,
    repository_id: UUID,
    *,
    path: str,
    symbol: str | None = None,
    limit: int = DEFAULT_IMPACT_LIMIT,
) -> ImpactResult:
    """Return one-hop direct dependents; an empty result is not proof of safety.

    Only metadata is selected. The caller owns the session and transaction,
    including pending changes that must not be autoflushed by this read.
    """
    _validate(path, symbol, limit)
    with session.no_autoflush:
        if (
            session.scalar(select(Repository.id).where(Repository.id == repository_id))
            is None
        ):
            raise ImpactNotFoundError("Repository not found")
        file = session.execute(
            select(File.id, File.path).where(
                File.repository_id == repository_id, File.path == path
            )
        ).one_or_none()
        if file is None:
            raise ImpactNotFoundError("File not found")
        target = ImpactTarget(file_id=file.id, path=file.path)
        if symbol is None:
            items = _incoming_imports(session, repository_id, file.id, limit)
        else:
            units = session.execute(
                select(
                    CodeUnit.id,
                    CodeUnit.symbol_name,
                    CodeUnit.kind,
                    CodeUnit.start_line,
                    CodeUnit.end_line,
                )
                .where(CodeUnit.file_id == file.id, CodeUnit.symbol_name == symbol)
                .limit(2)
            ).all()
            if not units:
                raise ImpactNotFoundError("Symbol not found")
            if len(units) > 1:
                raise AmbiguousImpactTargetError(
                    "Symbol is ambiguous in the selected file"
                )
            unit = units[0]
            target = ImpactTarget(
                file_id=file.id,
                path=file.path,
                code_unit_id=unit.id,
                symbol_name=unit.symbol_name,
                kind=unit.kind,
                start_line=unit.start_line,
                end_line=unit.end_line,
            )
            items = _incoming_calls(session, repository_id, file.id, unit.id, limit)
    return ImpactResult(
        repository_id=repository_id,
        mode="file" if symbol is None else "symbol",
        target=target,
        items=tuple(items[:limit]),
        limit=limit,
        truncated=len(items) > limit,
    )


def _incoming_imports(
    session: Session, repository_id: UUID, target_id: UUID, limit: int
) -> list[ImpactItem]:
    source = aliased(File)
    target = aliased(File)
    rows = session.execute(
        select(source.id, source.path)
        .select_from(Relationship)
        .join(source, source.id == Relationship.source_file_id)
        .join(target, target.id == Relationship.target_file_id)
        .where(
            Relationship.repository_id == repository_id,
            Relationship.relationship_type == "imports",
            source.repository_id == repository_id,
            target.repository_id == repository_id,
            target.id == target_id,
            source.id != target.id,
        )
        .distinct()
        .order_by(source.path.asc(), source.id.asc())
        .limit(limit + 1)
    )
    return [ImpactItem(file_id=row.id, path=row.path) for row in rows]


def _incoming_calls(
    session: Session,
    repository_id: UUID,
    target_file_id: UUID,
    target_id: UUID,
    limit: int,
) -> list[ImpactItem]:
    source_file, target_file = aliased(File), aliased(File)
    source, target = aliased(CodeUnit), aliased(CodeUnit)
    rows = session.execute(
        select(
            source.file_id,
            source_file.path,
            source.id,
            source.symbol_name,
            source.kind,
            source.start_line,
            source.end_line,
        )
        .select_from(CodeRelationship)
        .join(source_file, source_file.id == CodeRelationship.source_file_id)
        .join(target_file, target_file.id == CodeRelationship.target_file_id)
        .join(
            source,
            (source.id == CodeRelationship.source_code_unit_id)
            & (source.file_id == source_file.id),
        )
        .join(
            target,
            (target.id == CodeRelationship.target_code_unit_id)
            & (target.file_id == target_file.id),
        )
        .where(
            CodeRelationship.repository_id == repository_id,
            CodeRelationship.relationship_type == "calls",
            source_file.repository_id == repository_id,
            target_file.repository_id == repository_id,
            target_file.id == target_file_id,
            target.id == target_id,
            source.id != target.id,
        )
        .distinct()
        .order_by(
            source_file.path.asc(),
            source.start_line.asc(),
            source.end_line.desc(),
            source.kind.asc(),
            source.id.asc(),
        )
        .limit(limit + 1)
    )
    return [
        ImpactItem(
            file_id=row.file_id,
            path=row.path,
            code_unit_id=row.id,
            symbol_name=row.symbol_name,
            kind=row.kind,
            start_line=row.start_line,
            end_line=row.end_line,
            relationship_type="calls",
        )
        for row in rows
    ]
