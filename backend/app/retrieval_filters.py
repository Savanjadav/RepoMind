from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.sql.elements import ColumnElement

from app.models.code_unit import CodeUnit
from app.models.file import File

_PATH_MAX_LENGTH = 2_000
_LANGUAGE_MAX_LENGTH = 64
_SYMBOL_MAX_LENGTH = 2_000


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    path: str | None = None
    language: str | None = None
    symbol: str | None = None

    def __post_init__(self) -> None:
        _validate_filter_value("Path", self.path, _PATH_MAX_LENGTH)
        _validate_filter_value("Language", self.language, _LANGUAGE_MAX_LENGTH)
        _validate_filter_value("Symbol", self.symbol, _SYMBOL_MAX_LENGTH)


def build_retrieval_filter_predicates(
    filters: RetrievalFilters | None,
) -> tuple[ColumnElement[bool], ...]:
    if filters is None:
        return ()

    predicates: list[ColumnElement[bool]] = []
    if filters.path is not None:
        predicates.append(func.strpos(File.path, filters.path) > 0)
    if filters.language is not None:
        predicates.append(CodeUnit.language == filters.language)
    if filters.symbol is not None:
        predicates.append(func.strpos(CodeUnit.symbol_name, filters.symbol) > 0)
    return tuple(predicates)


def _validate_filter_value(name: str, value: str | None, max_length: int) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{name} filter must be a string or None")
    if not value.strip():
        raise ValueError(f"{name} filter must contain non-whitespace characters")
    if len(value) > max_length:
        raise ValueError(f"{name} filter must contain at most {max_length} characters")
