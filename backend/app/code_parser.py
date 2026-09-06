from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class CodeParsingError(RuntimeError):
    """Raised when source cannot be parsed into reliable code units."""


class CodeUnitKind(StrEnum):
    CLASS = "class"
    CONFIG = "config"
    DOCUMENT = "document"
    FUNCTION = "function"
    IMPORT = "import"


@dataclass(frozen=True, slots=True)
class ParsedCodeUnit:
    kind: CodeUnitKind
    content: str
    relative_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CodeUnitKind):
            raise ValueError("Code unit kind must be a CodeUnitKind")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("Code unit content must be a non-empty string")
        _validate_relative_path(self.relative_path)
        _validate_language(self.language)
        _validate_line_range(self.start_line, self.end_line)
        if self.symbol_name is not None and (
            not isinstance(self.symbol_name, str) or not self.symbol_name.strip()
        ):
            raise ValueError("Symbol name must be a non-empty string or None")


class CodeParser(Protocol):
    def parse(
        self,
        *,
        content: str,
        relative_path: str,
    ) -> list[ParsedCodeUnit]: ...


def _validate_relative_path(relative_path: str) -> None:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Relative path must be a non-empty string")
    if relative_path.startswith("/") or "\\" in relative_path:
        raise ValueError("Relative path must be a relative POSIX path")

    components = relative_path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("Relative path must not contain empty or dot components")


def _validate_language(language: str) -> None:
    if not isinstance(language, str) or not language:
        raise ValueError("Language must be a non-empty string")
    if language != language.strip() or language != language.lower():
        raise ValueError("Language must be lowercase without surrounding whitespace")


def _validate_line_range(start_line: int, end_line: int) -> None:
    if (
        isinstance(start_line, bool)
        or not isinstance(start_line, int)
        or start_line < 1
    ):
        raise ValueError("Start line must be an integer greater than or equal to 1")
    if (
        isinstance(end_line, bool)
        or not isinstance(end_line, int)
        or end_line < start_line
    ):
        raise ValueError(
            "End line must be an integer greater than or equal to start line"
        )
