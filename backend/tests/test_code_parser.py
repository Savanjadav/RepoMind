from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from app.code_parser import CodeParser, CodeUnitKind, ParsedCodeUnit


class FakeParser:
    def __init__(self) -> None:
        self.received_content: str | None = None
        self.received_relative_path: str | None = None

    def parse(
        self,
        *,
        content: str,
        relative_path: str,
    ) -> list[ParsedCodeUnit]:
        self.received_content = content
        self.received_relative_path = relative_path
        return [
            ParsedCodeUnit(
                kind=CodeUnitKind.FUNCTION,
                content=content,
                relative_path=relative_path,
                language="python",
                start_line=10,
                end_line=11,
                symbol_name="authenticate_user",
            )
        ]


def _parse_with(
    parser: CodeParser,
    *,
    content: str,
    relative_path: str,
) -> list[ParsedCodeUnit]:
    return parser.parse(content=content, relative_path=relative_path)


def _unit(**overrides: object) -> ParsedCodeUnit:
    values: dict[str, object] = {
        "kind": CodeUnitKind.FUNCTION,
        "content": "def authenticate_user():\n    return True\n",
        "relative_path": "src/auth.py",
        "language": "python",
        "start_line": 10,
        "end_line": 11,
        "symbol_name": "authenticate_user",
    }
    values.update(overrides)
    return ParsedCodeUnit(
        kind=cast(CodeUnitKind, values["kind"]),
        content=cast(str, values["content"]),
        relative_path=cast(str, values["relative_path"]),
        language=cast(str, values["language"]),
        start_line=cast(int, values["start_line"]),
        end_line=cast(int, values["end_line"]),
        symbol_name=cast(str | None, values["symbol_name"]),
    )


def test_fake_parser_structurally_satisfies_contract() -> None:
    parser = FakeParser()
    content = "def authenticate_user():\n    return True\n"

    units = _parse_with(parser, content=content, relative_path="src/auth.py")

    assert parser.received_content == content
    assert parser.received_relative_path == "src/auth.py"
    assert units == [
        ParsedCodeUnit(
            kind=CodeUnitKind.FUNCTION,
            content=content,
            relative_path="src/auth.py",
            language="python",
            start_line=10,
            end_line=11,
            symbol_name="authenticate_user",
        )
    ]


@pytest.mark.parametrize(
    "kind",
    [CodeUnitKind.CLASS, CodeUnitKind.FUNCTION, CodeUnitKind.IMPORT],
)
def test_supported_code_unit_kinds(kind: CodeUnitKind) -> None:
    assert _unit(kind=kind).kind is kind


def test_values_and_inclusive_line_range_are_preserved_exactly() -> None:
    content = "class Café:\n\tpass\r\n"
    unit = _unit(
        kind=CodeUnitKind.CLASS,
        content=content,
        relative_path="src/café.py",
        language="python",
        start_line=10,
        end_line=11,
        symbol_name="Café",
    )

    assert unit.content == content
    assert unit.relative_path == "src/café.py"
    assert unit.language == "python"
    assert unit.start_line == 10
    assert unit.end_line == 11
    assert unit.symbol_name == "Café"


def test_unnamed_code_unit_is_supported() -> None:
    unit = _unit(kind=CodeUnitKind.IMPORT, symbol_name=None)

    assert unit.symbol_name is None


@pytest.mark.parametrize("start_line", [0, -1, True])
def test_invalid_start_line_is_rejected(start_line: object) -> None:
    with pytest.raises(ValueError):
        _unit(start_line=start_line)


@pytest.mark.parametrize("end_line", [9, True])
def test_invalid_end_line_is_rejected(end_line: object) -> None:
    with pytest.raises(ValueError):
        _unit(end_line=end_line)


def test_empty_content_is_rejected_without_stripping_valid_content() -> None:
    with pytest.raises(ValueError):
        _unit(content="")

    assert _unit(content="   ").content == "   "


@pytest.mark.parametrize("language", ["", " python", "python ", "Python", "pyTHon"])
def test_invalid_language_is_rejected(language: object) -> None:
    with pytest.raises(ValueError):
        _unit(language=language)


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "/src/main.py",
        "src\\main.py",
        "./src/main.py",
        "src/./main.py",
        "../secret.py",
        "src/../secret.py",
        "src//main.py",
    ],
)
def test_invalid_relative_path_is_rejected(relative_path: object) -> None:
    with pytest.raises(ValueError):
        _unit(relative_path=relative_path)


@pytest.mark.parametrize(
    "relative_path",
    [".gitignore", ".github/workflows/ci.yml", "源代码/主程序.py"],
)
def test_valid_dotfile_and_unicode_paths_are_preserved(relative_path: str) -> None:
    assert _unit(relative_path=relative_path).relative_path == relative_path


@pytest.mark.parametrize("symbol_name", ["", "   "])
def test_invalid_symbol_name_is_rejected(symbol_name: object) -> None:
    with pytest.raises(ValueError):
        _unit(symbol_name=symbol_name)


def test_code_unit_equality_is_value_based() -> None:
    assert _unit() == _unit()
    assert _unit(end_line=12) != _unit()


def test_parsed_code_unit_is_frozen() -> None:
    unit = _unit()

    with pytest.raises(FrozenInstanceError):
        unit.end_line = 12  # type: ignore[misc]
