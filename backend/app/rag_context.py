from collections.abc import Sequence
from dataclasses import dataclass

_BLOCK_SEPARATOR = "\n\n"
_TRUNCATION_MARKER = "[TRUNCATED]"


@dataclass(frozen=True, slots=True)
class ContextEvidence:
    repository_name: str
    path: str
    symbol_name: str | None
    start_line: int
    end_line: int
    content: str

    def __post_init__(self) -> None:
        _validate_metadata("Repository name", self.repository_name)
        _validate_metadata("Path", self.path)
        if self.symbol_name is not None:
            _validate_metadata("Symbol name", self.symbol_name)
        _validate_line_range(self.start_line, self.end_line)
        if not isinstance(self.content, str):
            raise ValueError("Content must be a string")


@dataclass(frozen=True, slots=True)
class FormattedContext:
    text: str
    included_evidence_count: int
    truncated: bool
    used_characters: int
    max_characters: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("Formatted context text must be a string")
        if (
            isinstance(self.included_evidence_count, bool)
            or not isinstance(self.included_evidence_count, int)
            or self.included_evidence_count < 0
        ):
            raise ValueError("Included evidence count must be a non-negative integer")
        if not isinstance(self.truncated, bool):
            raise ValueError("Truncated must be a bool")
        if (
            isinstance(self.used_characters, bool)
            or not isinstance(self.used_characters, int)
            or self.used_characters != len(self.text)
        ):
            raise ValueError("Used characters must equal the formatted text length")
        _validate_max_characters(self.max_characters)
        if self.used_characters > self.max_characters:
            raise ValueError("Formatted context exceeds its character budget")


def format_context(
    evidence: Sequence[ContextEvidence],
    *,
    max_characters: int,
) -> FormattedContext:
    _validate_max_characters(max_characters)
    items = tuple(evidence)
    if any(not isinstance(item, ContextEvidence) for item in items):
        raise ValueError("Every evidence item must be a ContextEvidence")

    blocks: list[str] = []
    used_characters = 0
    truncated = False

    for index, item in enumerate(items, start=1):
        separator_length = len(_BLOCK_SEPARATOR) if blocks else 0
        available = max_characters - used_characters - separator_length
        complete_block = _render_block(
            item,
            index=index,
            shown_characters=len(item.content),
            truncated=False,
        )
        if len(complete_block) <= available:
            blocks.append(complete_block)
            used_characters += separator_length + len(complete_block)
            continue

        truncated_block = _largest_truncated_block_that_fits(
            item,
            index=index,
            max_characters=available,
        )
        if truncated_block is not None:
            blocks.append(truncated_block)
            used_characters += separator_length + len(truncated_block)
        truncated = True
        break

    text = _BLOCK_SEPARATOR.join(blocks)
    return FormattedContext(
        text=text,
        included_evidence_count=len(blocks),
        truncated=truncated,
        used_characters=len(text),
        max_characters=max_characters,
    )


def _largest_truncated_block_that_fits(
    evidence: ContextEvidence,
    *,
    index: int,
    max_characters: int,
) -> str | None:
    if not evidence.content or max_characters < 0:
        return None

    best: str | None = None
    lower = 0
    upper = len(evidence.content) - 1
    while lower <= upper:
        shown = (lower + upper) // 2
        candidate = _render_block(
            evidence,
            index=index,
            shown_characters=shown,
            truncated=True,
        )
        if len(candidate) <= max_characters:
            best = candidate
            lower = shown + 1
        else:
            upper = shown - 1
    return best


def _render_block(
    evidence: ContextEvidence,
    *,
    index: int,
    shown_characters: int,
    truncated: bool,
) -> str:
    line_metadata = (
        f"Line: {evidence.start_line}"
        if evidence.start_line == evidence.end_line
        else f"Lines: {evidence.start_line}-{evidence.end_line}"
    )
    symbol = evidence.symbol_name if evidence.symbol_name is not None else "<none>"
    header = "\n".join(
        (
            f"[Evidence {index}]",
            f"Repository: {evidence.repository_name}",
            f"Path: {evidence.path}",
            f"Symbol: {symbol}",
            line_metadata,
            f"Content-Characters: {shown_characters}/{len(evidence.content)}",
            "Content:",
        )
    )
    content = evidence.content[:shown_characters]
    marker = f"\n{_TRUNCATION_MARKER}" if truncated else ""
    return f"{header}\n{content}{marker}\n[End Evidence {index}]"


def _validate_metadata(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must contain non-whitespace characters")
    if any(ord(character) <= 0x1F or ord(character) == 0x7F for character in value):
        raise ValueError(f"{label} must not contain ASCII control characters")


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


def _validate_max_characters(max_characters: int) -> None:
    if (
        isinstance(max_characters, bool)
        or not isinstance(max_characters, int)
        or max_characters <= 0
    ):
        raise ValueError("Maximum characters must be a positive integer")
