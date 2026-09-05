from collections.abc import Callable

import pytest

from app.code_parser import CodeParser, CodeParsingError, CodeUnitKind, ParsedCodeUnit
from app.javascript_typescript_code_parser import (
    JavaScriptCodeParser,
    TypeScriptCodeParser,
)

ParserFactory = Callable[[], CodeParser]


def _parse_with(
    parser: CodeParser,
    *,
    content: str,
    relative_path: str,
) -> list[ParsedCodeUnit]:
    return parser.parse(content=content, relative_path=relative_path)


@pytest.mark.parametrize(
    ("parser_factory", "language", "relative_path"),
    [
        (JavaScriptCodeParser, "javascript", "src/app.js"),
        (TypeScriptCodeParser, "typescript", "src/app.ts"),
    ],
)
def test_public_parsers_structurally_satisfy_code_parser(
    parser_factory: ParserFactory,
    language: str,
    relative_path: str,
) -> None:
    units = _parse_with(
        parser_factory(),
        content="function login() { return true; }",
        relative_path=relative_path,
    )

    assert len(units) == 1
    assert units[0].language == language
    assert units[0].relative_path == relative_path


@pytest.mark.parametrize("content", ["", "// comment only\n", "/* comment only */"])
def test_javascript_empty_or_comment_only_source_returns_no_units(
    content: str,
) -> None:
    assert (
        JavaScriptCodeParser().parse(content=content, relative_path="src/app.js") == []
    )


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("function login() {}", "login"),
        ("async function loadUser() {}", "loadUser"),
        ("function* generate() {}", "generate"),
    ],
)
def test_javascript_function_declarations(source: str, name: str) -> None:
    unit = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )[0]

    assert unit.kind is CodeUnitKind.FUNCTION
    assert unit.symbol_name == name
    assert unit.content == source
    assert unit.language == "javascript"


def test_javascript_class_and_implemented_class_methods() -> None:
    source = """class Service {
  constructor() {}
  run() {}
  async load() {}
  get value() { return 1; }
  set value(next) {}
  *items() {}
  [computed]() {}
}
"""

    units = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/service.js",
    )

    assert [(unit.kind, unit.symbol_name) for unit in units] == [
        (CodeUnitKind.CLASS, "Service"),
        (CodeUnitKind.FUNCTION, "constructor"),
        (CodeUnitKind.FUNCTION, "run"),
        (CodeUnitKind.FUNCTION, "load"),
        (CodeUnitKind.FUNCTION, "value"),
        (CodeUnitKind.FUNCTION, "value"),
        (CodeUnitKind.FUNCTION, "items"),
    ]
    assert units[0].content == source.rstrip("\n")
    assert units[0].start_line == 1
    assert units[0].end_line == 9


@pytest.mark.parametrize(
    "source",
    [
        'import React from "react";',
        'import { foo, bar } from "./utils";',
        'import * as api from "./api";',
    ],
)
def test_javascript_import_is_one_exact_unnamed_unit(source: str) -> None:
    units = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )

    assert len(units) == 1
    assert units[0].kind is CodeUnitKind.IMPORT
    assert units[0].symbol_name is None
    assert units[0].content == source


def test_javascript_imports_remain_in_source_order() -> None:
    source = 'import z from "z";\nimport { a, b } from "a";\n'

    units = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )

    assert [unit.content for unit in units] == [
        'import z from "z";',
        'import { a, b } from "a";',
    ]


def test_javascript_named_exports_use_wrapper_content_without_duplicates() -> None:
    source = """export function login() {
  function nested() {}
}
export class Service {
  run() {}
}
export default function load() {}
export default class Store {}
"""

    units = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )

    assert [unit.symbol_name for unit in units] == [
        "login",
        "nested",
        "Service",
        "run",
        "load",
        "Store",
    ]
    assert units[0].content.startswith("export function login()")
    assert units[2].content.startswith("export class Service")
    assert units[4].content == "export default function load() {}"
    assert units[5].content == "export default class Store {}"
    assert sum(unit.symbol_name == "login" for unit in units) == 1
    assert sum(unit.symbol_name == "Service" for unit in units) == 1


def test_javascript_variable_bound_functions_and_noise_policy() -> None:
    source = """const arrow = (value) => value;
const expression = function (value) { return value; };
const generator = function* () { yield 1; };
users.map(user => user.id);
users.map(function (user) { return user.id; });
const service = { run() {} };
const { skipped } = { skipped: () => true };
"""

    units = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )

    assert [unit.symbol_name for unit in units] == [
        "arrow",
        "expression",
        "generator",
    ]
    assert units[0].content == "const arrow = (value) => value;"
    assert units[1].content.startswith("const expression = function")
    assert all(unit.kind is CodeUnitKind.FUNCTION for unit in units)


def test_exported_single_variable_function_uses_export_wrapper() -> None:
    source = "export const handler = () => true;"

    unit = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )[0]

    assert unit.symbol_name == "handler"
    assert unit.content == source


def test_exported_multiple_function_declarators_use_distinct_narrow_spans() -> None:
    source = "export const a = () => {}, b = function () {};"

    units = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )

    assert [unit.symbol_name for unit in units] == ["a", "b"]
    assert [unit.content for unit in units] == [
        "a = () => {}",
        "b = function () {}",
    ]
    assert all(unit.content != source for unit in units)


def test_anonymous_default_exports_get_no_manufactured_units() -> None:
    source = """export default function () {
  function nested() {}
}
export default class {
  run() {}
}
"""

    units = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )

    assert [unit.symbol_name for unit in units] == ["nested", "run"]


def test_nested_javascript_symbols_are_source_ordered() -> None:
    source = """function outer() {
  const handler = () => true;
  function inner() {}
  class Nested {}
}
"""

    units = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )

    assert [unit.symbol_name for unit in units] == [
        "outer",
        "handler",
        "inner",
        "Nested",
    ]


def test_javascript_unicode_byte_slicing_and_identifier() -> None:
    source = """const message = "café";

function 登入() {
  return "✓";
}
"""

    unit = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/登入.js",
    )[0]

    assert unit.symbol_name == "登入"
    assert unit.content == 'function 登入() {\n  return "✓";\n}'
    assert unit.relative_path == "src/登入.js"
    assert (unit.start_line, unit.end_line) == (3, 5)


@pytest.mark.parametrize(
    ("source", "expected_content", "expected_lines"),
    [
        ("function one() {}\n", "function one() {}", (1, 1)),
        (
            "function two() {\n  return true;\n}\n",
            "function two() {\n  return true;\n}",
            (1, 3),
        ),
        (
            "function crlf() {\r\n  return true;\r\n}\r\n",
            "function crlf() {\r\n  return true;\r\n}",
            (1, 3),
        ),
    ],
)
def test_javascript_exact_content_and_inclusive_line_ranges(
    source: str,
    expected_content: str,
    expected_lines: tuple[int, int],
) -> None:
    unit = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/app.js",
    )[0]

    assert unit.content == expected_content
    assert (unit.start_line, unit.end_line) == expected_lines


def test_javascript_grammar_naturally_accepts_jsx_inside_function() -> None:
    source = "function App() {\n  return <div>Hello</div>;\n}"

    units = JavaScriptCodeParser().parse(
        content=source,
        relative_path="src/App.jsx",
    )

    assert [(unit.kind, unit.symbol_name) for unit in units] == [
        (CodeUnitKind.FUNCTION, "App")
    ]
    assert units[0].content == source


@pytest.mark.parametrize(
    "source",
    ["function broken( {", "export class {", "const value: = 1;"],
)
def test_javascript_malformed_source_raises(source: str) -> None:
    with pytest.raises(CodeParsingError, match="JavaScript source contains"):
        JavaScriptCodeParser().parse(content=source, relative_path="src/app.js")


@pytest.mark.parametrize("content", ["", "// comment only\n", "/* comment */"])
def test_typescript_empty_or_comment_only_source_returns_no_units(
    content: str,
) -> None:
    assert (
        TypeScriptCodeParser().parse(content=content, relative_path="src/app.ts") == []
    )


def test_typescript_implementation_bearing_functions() -> None:
    source = """function add(a: number, b: number): number {
  return a + b;
}
async function load<T>(value: T): Promise<T> {
  return value;
}
function identity<T>(value: T): T { return value; }
"""

    units = TypeScriptCodeParser().parse(
        content=source,
        relative_path="src/app.ts",
    )

    assert [unit.symbol_name for unit in units] == ["add", "load", "identity"]
    assert all(unit.kind is CodeUnitKind.FUNCTION for unit in units)
    assert all(unit.language == "typescript" for unit in units)
    assert units[0].content == source.split("async function", maxsplit=1)[0].rstrip()


def test_typescript_classes_methods_and_abstract_class() -> None:
    source = """class Service<T> {
  run(value: T): T { return value; }
}
abstract class Base {
  abstract missing(): void;
  implemented(): boolean { return true; }
}
"""

    units = TypeScriptCodeParser().parse(
        content=source,
        relative_path="src/service.ts",
    )

    assert [(unit.kind, unit.symbol_name) for unit in units] == [
        (CodeUnitKind.CLASS, "Service"),
        (CodeUnitKind.FUNCTION, "run"),
        (CodeUnitKind.CLASS, "Base"),
        (CodeUnitKind.FUNCTION, "implemented"),
    ]


@pytest.mark.parametrize(
    "source",
    [
        'import { value } from "./value";',
        'import type { User } from "./types";',
    ],
)
def test_typescript_imports_are_exact_unnamed_units(source: str) -> None:
    unit = TypeScriptCodeParser().parse(
        content=source,
        relative_path="src/app.ts",
    )[0]

    assert unit.kind is CodeUnitKind.IMPORT
    assert unit.symbol_name is None
    assert unit.content == source
    assert unit.language == "typescript"


def test_typescript_exports_and_typed_variable_functions() -> None:
    source = """export function parse(value: string): string { return value; }
export class Parser<T> { parse(value: T): T { return value; } }
export const convert = (value: number): string => String(value);
const transform = function (value: string): string { return value; };
"""

    units = TypeScriptCodeParser().parse(
        content=source,
        relative_path="src/app.ts",
    )

    assert [unit.symbol_name for unit in units] == [
        "parse",
        "Parser",
        "parse",
        "convert",
        "transform",
    ]
    assert units[0].content.startswith("export function parse")
    assert units[1].content.startswith("export class Parser")
    assert units[3].content.startswith("export const convert")
    assert units[4].content.endswith(";")


def test_typescript_omits_type_only_and_ambient_constructs() -> None:
    source = """interface User { id: string; }
type UserId = string;
enum Status { Ready }
declare function load(): void;
declare class Declared { run(): void; }
const callback = [1].map(value => value + 1);
"""

    assert (
        TypeScriptCodeParser().parse(
            content=source,
            relative_path="src/types.ts",
        )
        == []
    )


def test_typescript_namespace_is_omitted_but_implementation_is_emitted() -> None:
    source = """namespace Tools {
  export function run(): boolean {
    return true;
  }
}
"""

    units = TypeScriptCodeParser().parse(
        content=source,
        relative_path="src/tools.ts",
    )

    assert [(unit.kind, unit.symbol_name) for unit in units] == [
        (CodeUnitKind.FUNCTION, "run")
    ]
    assert units[0].content.startswith("export function run")


def test_typescript_unicode_crlf_content_and_lines() -> None:
    source = (
        'const café = "✓";\r\n\r\n'
        "function 登录(value: string): string {\r\n"
        "  return value;\r\n"
        "}\r\n"
    )

    unit = TypeScriptCodeParser().parse(
        content=source,
        relative_path="src/国际.ts",
    )[0]

    assert unit.symbol_name == "登录"
    assert unit.content == (
        "function 登录(value: string): string {\r\n  return value;\r\n}"
    )
    assert unit.relative_path == "src/国际.ts"
    assert (unit.start_line, unit.end_line) == (3, 5)


def test_typescript_malformed_source_raises_without_partial_units() -> None:
    source = "function valid(): void {}\nfunction broken( {"

    with pytest.raises(CodeParsingError, match="TypeScript source contains"):
        TypeScriptCodeParser().parse(content=source, relative_path="src/app.ts")
