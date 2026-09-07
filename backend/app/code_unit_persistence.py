from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.code_parser import ParsedCodeUnit
from app.models.code_unit import CodeUnit
from app.models.file import File


def persist_code_units(
    session: Session,
    file_id: UUID,
    parsed_units: Sequence[ParsedCodeUnit],
) -> list[CodeUnit]:
    units = list(parsed_units)
    if not units:
        return []

    file = session.get(File, file_id)
    if file is None:
        raise ValueError("File does not exist")

    if any(unit.relative_path != file.path for unit in units):
        raise ValueError("Parsed code unit path does not match target file")

    rows = [
        CodeUnit(
            file_id=file_id,
            kind=unit.kind.value,
            content=unit.content,
            language=unit.language,
            start_line=unit.start_line,
            end_line=unit.end_line,
            symbol_name=unit.symbol_name,
        )
        for unit in units
    ]
    session.add_all(rows)
    session.flush()
    return rows
