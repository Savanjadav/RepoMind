import pytest

from app.code_parser import (
    CodeParser,
    CodeParsingError,
    CodeUnitKind,
    ParsedCodeUnit,
)
from app.python_code_parser import PythonCodeParser


@pytest.fixture
def parser() -> PythonCodeParser:
    return PythonCodeParser()


def _parse_with(
    parser: CodeParser,
    *,
    content: str,
    relative_path: str = "src/example.py",
) -> list[ParsedCodeUnit]:
    return parser.parse(content=content, relative_path=relative_path)


def test_python_parser_structurally_satisfies_code_parser(
    parser: PythonCodeParser,
) -> None:
    units = _parse_with(parser, content="def ready(): pass")

    assert units[0].symbol_name == "ready"


@pytest.mark.parametrize("content", ["", "# comment only\n"])
def test_source_without_symbols_returns_empty_list(
    parser: PythonCodeParser,
    content: str,
) -> None:
    assert parser.parse(content=content, relative_path="empty.py") == []


def test_parses_top_level_function_exactly(parser: PythonCodeParser) -> None:
    content = "def authenticate(user):\n    return True"

    units = parser.parse(content=content, relative_path="src/auth.py")

    assert len(units) == 1
    assert units[0].kind is CodeUnitKind.FUNCTION
    assert units[0].symbol_name == "authenticate"
    assert units[0].content == content
    assert units[0].relative_path == "src/auth.py"
    assert units[0].language == "python"
    assert (units[0].start_line, units[0].end_line) == (1, 2)


def test_parses_async_function(parser: PythonCodeParser) -> None:
    units = parser.parse(
        content="async def fetch_user():\n    return None",
        relative_path="fetch.py",
    )

    assert len(units) == 1
    assert units[0].kind is CodeUnitKind.FUNCTION
    assert units[0].symbol_name == "fetch_user"


def test_parses_class_with_docstring_exactly(parser: PythonCodeParser) -> None:
    content = 'class UserService:\n    """Manage users."""\n\n    enabled = True'

    units = parser.parse(content=content, relative_path="services.py")

    assert len(units) == 1
    assert units[0].kind is CodeUnitKind.CLASS
    assert units[0].symbol_name == "UserService"
    assert units[0].content == content
    assert (units[0].start_line, units[0].end_line) == (1, 4)


def test_function_docstring_remains_in_exact_content(
    parser: PythonCodeParser,
) -> None:
    content = 'def documented():\n    """Explain the function."""\n    return True'

    units = parser.parse(content=content, relative_path="documented.py")

    assert units[0].content == content


@pytest.mark.parametrize(
    "content",
    [
        "import os",
        "from pathlib import Path",
        "from __future__ import annotations",
    ],
)
def test_parses_each_import_statement_as_one_unnamed_unit(
    parser: PythonCodeParser,
    content: str,
) -> None:
    units = parser.parse(content=content, relative_path="imports.py")

    assert len(units) == 1
    assert units[0].kind is CodeUnitKind.IMPORT
    assert units[0].symbol_name is None
    assert units[0].content == content
    assert (units[0].start_line, units[0].end_line) == (1, 1)


@pytest.mark.parametrize(
    "content",
    ["import os, sys", "from app.auth import login, logout"],
)
def test_multiple_imported_names_remain_one_unit(
    parser: PythonCodeParser,
    content: str,
) -> None:
    units = parser.parse(content=content, relative_path="imports.py")

    assert len(units) == 1
    assert units[0].content == content


def test_units_are_returned_in_source_order(parser: PythonCodeParser) -> None:
    content = "import os\n\nclass Service:\n    pass\n\ndef run():\n    pass"

    units = parser.parse(content=content, relative_path="ordered.py")

    assert [(unit.kind, unit.symbol_name) for unit in units] == [
        (CodeUnitKind.IMPORT, None),
        (CodeUnitKind.CLASS, "Service"),
        (CodeUnitKind.FUNCTION, "run"),
    ]


def test_single_line_function_has_one_inclusive_line(
    parser: PythonCodeParser,
) -> None:
    units = parser.parse(content="def ready(): pass\n", relative_path="single.py")

    assert units[0].content == "def ready(): pass"
    assert (units[0].start_line, units[0].end_line) == (1, 1)


def test_decorated_function_includes_all_decorators_once(
    parser: PythonCodeParser,
) -> None:
    content = '@route("/users")\n@authorized\ndef users():\n    return []'

    units = parser.parse(content=content, relative_path="routes.py")

    assert len(units) == 1
    assert units[0].kind is CodeUnitKind.FUNCTION
    assert units[0].symbol_name == "users"
    assert units[0].content == content
    assert (units[0].start_line, units[0].end_line) == (1, 4)


def test_decorated_class_includes_decorator_once(parser: PythonCodeParser) -> None:
    content = "@dataclass\nclass User:\n    name: str"

    units = parser.parse(content=content, relative_path="models.py")

    assert len(units) == 1
    assert units[0].kind is CodeUnitKind.CLASS
    assert units[0].symbol_name == "User"
    assert units[0].content == content


def test_methods_and_nested_symbols_are_extracted_in_source_order(
    parser: PythonCodeParser,
) -> None:
    content = """\
class Service:
    import logging

    async def run(self):
        def inner():
            return True

        class Result:
            pass

        return inner()
"""

    units = parser.parse(content=content, relative_path="nested.py")

    assert [(unit.kind, unit.symbol_name) for unit in units] == [
        (CodeUnitKind.CLASS, "Service"),
        (CodeUnitKind.IMPORT, None),
        (CodeUnitKind.FUNCTION, "run"),
        (CodeUnitKind.FUNCTION, "inner"),
        (CodeUnitKind.CLASS, "Result"),
    ]


def test_unicode_before_definition_does_not_shift_byte_slice(
    parser: PythonCodeParser,
) -> None:
    content = 'message = "café"\n\ndef login():\n    return True'

    units = parser.parse(content=content, relative_path="unicode.py")

    assert units[0].content == "def login():\n    return True"
    assert units[0].content.startswith("def login():")
    assert (units[0].start_line, units[0].end_line) == (3, 4)


def test_unicode_content_and_identifiers_are_preserved(
    parser: PythonCodeParser,
) -> None:
    content = 'def calcular_área():\n    return "π × radio²"\n\nclass Café:\n    pass'

    units = parser.parse(content=content, relative_path="módulos/geometría.py")

    assert [unit.symbol_name for unit in units] == ["calcular_área", "Café"]
    assert units[0].content == 'def calcular_área():\n    return "π × radio²"'
    assert units[1].content == "class Café:\n    pass"
    assert all(unit.relative_path == "módulos/geometría.py" for unit in units)


def test_crlf_content_and_line_numbers_are_preserved(
    parser: PythonCodeParser,
) -> None:
    content = (
        "def first():\r\n    return True\r\n\r\ndef second():\r\n    return False\r\n"
    )

    units = parser.parse(content=content, relative_path="windows.py")

    assert units[0].content == "def first():\r\n    return True"
    assert units[1].content == "def second():\r\n    return False"
    assert (units[0].start_line, units[0].end_line) == (1, 2)
    assert (units[1].start_line, units[1].end_line) == (4, 5)


def test_malformed_source_raises_without_partial_units(
    parser: PythonCodeParser,
) -> None:
    content = "def valid():\n    return True\n\ndef broken(:\n    pass"

    with pytest.raises(CodeParsingError, match="syntax errors"):
        parser.parse(content=content, relative_path="broken.py")


def test_parsing_never_executes_source(parser: PythonCodeParser) -> None:
    content = 'raise RuntimeError("must not execute")\n\ndef safe():\n    return True'

    units = parser.parse(content=content, relative_path="untrusted.py")

    assert [unit.symbol_name for unit in units] == ["safe"]
    assert all(unit.kind in set(CodeUnitKind) for unit in units)
