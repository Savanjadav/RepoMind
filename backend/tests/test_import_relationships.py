import pytest

from app.import_relationships import extract_imports, resolve_imports
from app.javascript_typescript_code_parser import (
    JavaScriptCodeParser,
    TypeScriptCodeParser,
)
from app.python_code_parser import PythonCodeParser


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import pkg.worker as w, other", ["other.py", "pkg/worker.py"]),
        ("from pkg.worker import x, y", ["pkg/worker.py"]),
        ("from pkg.worker import *", ["pkg/worker.py"]),
        ("from .worker import x", ["pkg/worker.py"]),
        ("from . import worker", ["pkg/__init__.py"]),
        ("from ..other import x", []),
        ("import os, missing", []),
        ("from __future__ import annotations", []),
        ("import pkg.source", []),
        ("import other\nimport other", ["other.py"]),
        ('value = "import other" # import pkg.worker', []),
    ],
)
def test_python_static_imports(source: str, expected: list[str]) -> None:
    paths = {"pkg/source.py", "pkg/__init__.py", "pkg/worker.py", "other.py", "os.py"}
    units = PythonCodeParser().parse(content=source, relative_path="pkg/source.py")
    references = extract_imports(units)
    assert resolve_imports(references, paths) == [
        ("pkg/source.py", path) for path in expected
    ]
    assert resolve_imports(references, paths) == resolve_imports(
        list(reversed(references)), paths
    )


@pytest.mark.parametrize(
    "paths",
    [
        {"pkg/worker.py"},
        {"pkg.py", "pkg/__init__.py", "pkg/worker.py"},
        {"pkg/__init__.py", "pkg/worker.py", "pkg/worker/__init__.py"},
        {"src/pkg/__init__.py", "src/pkg/worker.py"},
    ],
)
def test_python_missing_package_ambiguous_or_inferred_root(paths: set[str]) -> None:
    units = PythonCodeParser().parse(
        content="import pkg.worker", relative_path="main.py"
    )
    assert resolve_imports(extract_imports(units), paths | {"main.py"}) == []


def test_python_parent_relative_package() -> None:
    units = PythonCodeParser().parse(
        content="from ..worker import x", relative_path="pkg/sub/a.py"
    )
    paths = {"pkg/sub/a.py", "pkg/sub/__init__.py", "pkg/__init__.py", "pkg/worker.py"}
    assert resolve_imports(extract_imports(units), paths) == [
        ("pkg/sub/a.py", "pkg/worker.py")
    ]


@pytest.mark.parametrize("language", ["javascript", "typescript"])
@pytest.mark.parametrize(
    ("source", "targets", "expected"),
    [
        ('import x from "./b.js";', {"src/b.js"}, "src/b.js"),
        ('import {x as y, z} from "./b";', {"src/b.ts"}, "src/b.ts"),
        ('import * as b from "./b";', {"src/b/index.js"}, "src/b/index.js"),
        ('import "../b";', {"b.jsx"}, "b.jsx"),
        ('import "./b";', {"src/b.js", "src/b.ts"}, None),
        ('import "./b";', {"src/b.js", "src/b/index.js"}, None),
        ('import "react";', {"react.js"}, None),
        ('import "@/b";', {"src/b.js"}, None),
        ('import "https://example.com/b.js";', set(), None),
        ('import "/b.js";', {"b.js"}, None),
        ('import "../../b.js";', {"b.js"}, None),
        ('import "./b.js?raw";', {"src/b.js"}, None),
        ('import "./b.tsx";', {"src/b.tsx"}, None),
        ('import "./b.js";', {"src/b.ts"}, None),
        ('import "./a.js";', set(), None),
        ('import "./b\\x2ejs";', {"src/b.js"}, None),
        (
            'const x = require("./b"); import("./b"); export * from "./b";',
            {"src/b.js"},
            None,
        ),
    ],
)
def test_es_imports(
    language: str, source: str, targets: set[str], expected: str | None
) -> None:
    parser = (
        JavaScriptCodeParser() if language == "javascript" else TypeScriptCodeParser()
    )
    units = parser.parse(content=source, relative_path="src/a.js")
    assert resolve_imports(extract_imports(units), targets | {"src/a.js"}) == (
        [] if expected is None else [("src/a.js", expected)]
    )


def test_typescript_type_import() -> None:
    units = TypeScriptCodeParser().parse(
        content='import type {User} from "./types";', relative_path="a.ts"
    )
    assert resolve_imports(extract_imports(units), {"a.ts", "types.ts"}) == [
        ("a.ts", "types.ts")
    ]


def test_root_directory_index() -> None:
    units = JavaScriptCodeParser().parse(content='import "./";', relative_path="a.js")
    assert resolve_imports(extract_imports(units), {"a.js", "index.js"}) == [
        ("a.js", "index.js")
    ]


@pytest.mark.parametrize("language", ["javascript", "typescript"])
@pytest.mark.parametrize(
    ("specifier", "targets", "expected"),
    [
        ("./b.js/", {"b.js"}, None),
        ("./b/", {"b.js"}, None),
        ("./b/", {"b/index.js"}, "b/index.js"),
        ("./b/", {"b/index.ts"}, "b/index.ts"),
        ("./b/", {"b/index.js", "b/index.ts"}, None),
        ("./", {"index.js"}, "index.js"),
        ("./b.js/", {"b.js/index.jsx"}, "b.js/index.jsx"),
        ("./b/", {"b.js", "b/index.js"}, "b/index.js"),
        ("./missing/", set(), None),
    ],
)
def test_directory_specifiers_only_consider_index_candidates(
    language: str, specifier: str, targets: set[str], expected: str | None
) -> None:
    parser = (
        JavaScriptCodeParser() if language == "javascript" else TypeScriptCodeParser()
    )
    source_path = "a.js" if language == "javascript" else "a.ts"
    units = parser.parse(content=f'import "{specifier}";', relative_path=source_path)
    assert resolve_imports(extract_imports(units), targets | {source_path}) == (
        [] if expected is None else [(source_path, expected)]
    )


@pytest.mark.parametrize("language", ["javascript", "typescript"])
def test_directory_index_self_import_is_omitted(language: str) -> None:
    parser = (
        JavaScriptCodeParser() if language == "javascript" else TypeScriptCodeParser()
    )
    path = "index.js" if language == "javascript" else "index.ts"
    units = parser.parse(content='import "./";', relative_path=path)
    assert resolve_imports(extract_imports(units), {path}) == []
