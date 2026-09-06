from dataclasses import dataclass
from pathlib import PurePosixPath

from app.code_parser import CodeUnitKind, ParsedCodeUnit

DEFAULT_MAX_CHUNK_CHARACTERS = 2_000

_MARKDOWN_SUFFIXES = frozenset({".markdown", ".md"})


@dataclass(frozen=True, slots=True)
class _SourceLine:
    start: int
    end: int
    number: int


@dataclass(frozen=True, slots=True)
class _Section:
    first_line_index: int
    end_line_index: int
    symbol_name: str | None


class DocumentationTextParser:
    def __init__(
        self,
        *,
        max_chunk_characters: int = DEFAULT_MAX_CHUNK_CHARACTERS,
    ) -> None:
        self._max_chunk_characters = _validate_max_chunk_characters(
            max_chunk_characters
        )

    def parse(
        self,
        *,
        content: str,
        relative_path: str,
    ) -> list[ParsedCodeUnit]:
        if not content or not content.strip():
            return []

        lines = _source_lines(content)
        if _is_markdown_path(relative_path):
            sections = _markdown_sections(content, lines)
        else:
            sections = [_Section(0, len(lines), None)]

        return _chunk_sections(
            content=content,
            lines=lines,
            sections=sections,
            max_chunk_characters=self._max_chunk_characters,
            kind=CodeUnitKind.DOCUMENT,
            language="documentation",
            relative_path=relative_path,
        )


class ConfigurationTextParser:
    def __init__(
        self,
        *,
        max_chunk_characters: int = DEFAULT_MAX_CHUNK_CHARACTERS,
    ) -> None:
        self._max_chunk_characters = _validate_max_chunk_characters(
            max_chunk_characters
        )

    def parse(
        self,
        *,
        content: str,
        relative_path: str,
    ) -> list[ParsedCodeUnit]:
        if not content or not content.strip():
            return []

        lines = _source_lines(content)
        return _chunk_sections(
            content=content,
            lines=lines,
            sections=[_Section(0, len(lines), None)],
            max_chunk_characters=self._max_chunk_characters,
            kind=CodeUnitKind.CONFIG,
            language="config",
            relative_path=relative_path,
        )


def _validate_max_chunk_characters(max_chunk_characters: int) -> int:
    if (
        isinstance(max_chunk_characters, bool)
        or not isinstance(max_chunk_characters, int)
        or max_chunk_characters <= 0
    ):
        raise ValueError("Maximum chunk characters must be a positive integer")
    return max_chunk_characters


def _is_markdown_path(relative_path: str) -> bool:
    return PurePosixPath(relative_path).suffix.casefold() in _MARKDOWN_SUFFIXES


def _source_lines(content: str) -> list[_SourceLine]:
    lines: list[_SourceLine] = []
    offset = 0
    for number, line in enumerate(content.splitlines(keepends=True), start=1):
        end = offset + len(line)
        lines.append(_SourceLine(start=offset, end=end, number=number))
        offset = end
    return lines


def _markdown_sections(content: str, lines: list[_SourceLine]) -> list[_Section]:
    sections: list[_Section] = []
    section_start = 0
    section_symbol: str | None = None
    fence_character: str | None = None
    fence_length = 0

    for index, line in enumerate(lines):
        line_text = content[line.start : line.end]

        if fence_character is not None:
            if _is_closing_fence(line_text, fence_character, fence_length):
                fence_character = None
                fence_length = 0
            continue

        opening_fence = _opening_fence(line_text)
        if opening_fence is not None:
            fence_character, fence_length = opening_fence
            continue

        is_heading, heading_name = _markdown_heading(line_text)
        if not is_heading:
            continue

        if index > section_start and _line_range_has_content(
            content,
            lines,
            section_start,
            index,
        ):
            sections.append(_Section(section_start, index, section_symbol))
        section_start = index
        section_symbol = heading_name

    if section_start < len(lines) and _line_range_has_content(
        content,
        lines,
        section_start,
        len(lines),
    ):
        sections.append(_Section(section_start, len(lines), section_symbol))

    return sections


def _opening_fence(line_text: str) -> tuple[str, int] | None:
    candidate = _fence_candidate(line_text)
    if not candidate or candidate[0] not in {"`", "~"}:
        return None

    character = candidate[0]
    length = _leading_character_count(candidate, character)
    if length < 3:
        return None
    return character, length


def _is_closing_fence(
    line_text: str,
    character: str,
    opening_length: int,
) -> bool:
    candidate = _fence_candidate(line_text)
    length = _leading_character_count(candidate, character)
    return length >= opening_length and not candidate[length:].strip()


def _fence_candidate(line_text: str) -> str:
    line_body = _without_line_ending(line_text)
    indentation = len(line_body) - len(line_body.lstrip(" "))
    if indentation > 3:
        return ""
    return line_body[indentation:]


def _leading_character_count(value: str, character: str) -> int:
    count = 0
    while count < len(value) and value[count] == character:
        count += 1
    return count


def _markdown_heading(line_text: str) -> tuple[bool, str | None]:
    line_body = _without_line_ending(line_text)
    indentation = len(line_body) - len(line_body.lstrip(" "))
    if indentation > 3:
        return False, None

    candidate = line_body[indentation:]
    marker_count = _leading_character_count(candidate, "#")
    if marker_count < 1 or marker_count > 6:
        return False, None
    if len(candidate) > marker_count and not candidate[marker_count].isspace():
        return False, None

    heading_name = candidate[marker_count:].strip()
    heading_name = _remove_closing_heading_markers(heading_name)
    return True, heading_name or None


def _remove_closing_heading_markers(heading_name: str) -> str:
    without_markers = heading_name.rstrip("#")
    if without_markers != heading_name and without_markers.endswith((" ", "\t")):
        return without_markers.rstrip()
    return heading_name


def _without_line_ending(line_text: str) -> str:
    if line_text.endswith("\r\n"):
        return line_text[:-2]
    if line_text.endswith(("\n", "\r")):
        return line_text[:-1]
    return line_text


def _line_range_has_content(
    content: str,
    lines: list[_SourceLine],
    first_line_index: int,
    end_line_index: int,
) -> bool:
    start = lines[first_line_index].start
    end = lines[end_line_index - 1].end
    return bool(content[start:end].strip())


def _chunk_sections(
    *,
    content: str,
    lines: list[_SourceLine],
    sections: list[_Section],
    max_chunk_characters: int,
    kind: CodeUnitKind,
    language: str,
    relative_path: str,
) -> list[ParsedCodeUnit]:
    units: list[ParsedCodeUnit] = []
    for section in sections:
        units.extend(
            _chunk_section(
                content=content,
                lines=lines,
                section=section,
                max_chunk_characters=max_chunk_characters,
                kind=kind,
                language=language,
                relative_path=relative_path,
            )
        )
    return units


def _chunk_section(
    *,
    content: str,
    lines: list[_SourceLine],
    section: _Section,
    max_chunk_characters: int,
    kind: CodeUnitKind,
    language: str,
    relative_path: str,
) -> list[ParsedCodeUnit]:
    units: list[ParsedCodeUnit] = []
    chunk_start: int | None = None
    chunk_end = 0
    chunk_start_line = 0
    chunk_end_line = 0

    def flush() -> None:
        nonlocal chunk_start, chunk_end, chunk_start_line, chunk_end_line
        if chunk_start is not None:
            chunk_content = content[chunk_start:chunk_end]
            units.append(
                ParsedCodeUnit(
                    kind=kind,
                    content=chunk_content,
                    relative_path=relative_path,
                    language=language,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    symbol_name=section.symbol_name,
                )
            )
        chunk_start = None
        chunk_end = 0
        chunk_start_line = 0
        chunk_end_line = 0

    for line in lines[section.first_line_index : section.end_line_index]:
        fragment_start = line.start
        while fragment_start < line.end:
            available = max_chunk_characters
            if chunk_start is not None:
                available -= chunk_end - chunk_start
            if available == 0:
                flush()
                available = max_chunk_characters

            fragment_end = min(fragment_start + available, line.end)
            if chunk_start is None:
                chunk_start = fragment_start
                chunk_start_line = line.number
            chunk_end = fragment_end
            chunk_end_line = line.number
            fragment_start = fragment_end

            if chunk_end - chunk_start == max_chunk_characters:
                flush()

    flush()
    return units
