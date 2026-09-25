from uuid import uuid4

import pytest

import app.code_relationships as module
from app.code_relationships import FileCallAnalysis, analyze_calls, resolve_calls
from app.javascript_typescript_code_parser import (
    JavaScriptCodeParser,
    TypeScriptCodeParser,
)
from app.models.code_unit import CodeUnit
from app.models.file import File
from app.python_code_parser import PythonCodeParser


def _analyze(path: str, content: str, *, duplicate: bool = False) -> FileCallAnalysis:
    parser = (
        PythonCodeParser()
        if path.endswith(".py")
        else TypeScriptCodeParser()
        if path.endswith(".ts")
        else JavaScriptCodeParser()
    )
    language = (
        "python"
        if path.endswith(".py")
        else "typescript"
        if path.endswith(".ts")
        else "javascript"
    )
    file = File(id=uuid4(), repository_id=uuid4(), path=path)
    units = [
        CodeUnit(
            id=uuid4(),
            file_id=file.id,
            kind=u.kind.value,
            content=u.content,
            language=u.language,
            symbol_name=u.symbol_name,
            start_line=u.start_line,
            end_line=u.end_line,
        )
        for u in parser.parse(content=content, relative_path=path)
    ]
    if duplicate:
        units.extend(list(units))
    result = analyze_calls(content=content, file=file, language=language, units=units)
    assert all(u.content in content for u in units)
    return result


def _edges(files: dict[str, str]) -> list[tuple[str, str, str, str]]:
    analyses = [_analyze(path, source) for path, source in files.items()]
    paths = {e.file_id: a.path for a in analyses for e in a.endpoints}
    return [
        (paths[s.file_id], s.name, paths[t.file_id], t.name)
        for s, t in resolve_calls(analyses, set(files))
    ]


@pytest.mark.parametrize("prefix", ["", "async "])
def test_python_local_recursion_and_dedup(prefix: str) -> None:
    assert _edges(
        {
            "a.py": (
                f"{prefix}def helper():\n helper()\n\n"
                f"{prefix}def caller():\n helper()\n helper()\n"
            )
        }
    ) == [("a.py", "helper", "a.py", "helper"), ("a.py", "caller", "a.py", "helper")]


@pytest.mark.parametrize(
    ("statement", "call"),
    [
        ("from b import helper", "helper()"),
        ("from b import helper as run", "run()"),
        ("import b", "b.helper()"),
        ("import b as mod", "mod.helper()"),
    ],
)
def test_python_imports(statement: str, call: str) -> None:
    assert _edges(
        {
            "a.py": f"{statement}\ndef caller():\n {call}\n",
            "b.py": "def helper(): pass\n",
        }
    ) == [("a.py", "caller", "b.py", "helper")]


@pytest.mark.parametrize(
    "statement", ["from .b import helper", "from pkg.b import helper"]
)
def test_python_packages(statement: str) -> None:
    assert _edges(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": f"{statement}\ndef caller(): helper()\n",
            "pkg/b.py": "def helper(): pass\n",
        }
    ) == [("pkg/a.py", "caller", "pkg/b.py", "helper")]


@pytest.mark.parametrize(
    "body",
    [
        "helper = other\n helper()",
        "helper()\n helper = other",
        "helper += other\n helper()",
        "for helper in things:\n  helper()",
        "helper, x = things\n helper()",
        "try:\n  helper()\n except Error as helper:\n  pass",
        "(helper := other)\n helper()",
        "del helper\n helper()",
        "global helper\n helper()",
        "nonlocal helper\n helper()",
        "def helper(): pass\n helper()",
        "class helper: pass\n helper()",
        "from b import helper\n helper()",
        "[helper() for helper in things]",
        "with other as helper:\n  helper()",
    ],
)
def test_python_local_shadowing(body: str) -> None:
    assert _edges({"a.py": "def helper(): pass\ndef caller():\n " + body + "\n"}) == []


@pytest.mark.parametrize(
    "source",
    [
        "def helper(): pass\ndef caller(helper): helper()\n",
        "def helper(): pass\ndef helper(): pass\ndef caller(): helper()\n",
        "def helper(): pass\ndef caller(): helper()\nhelper = other\n",
        "from b import helper\ndef helper(): pass\ndef caller(): helper()\n",
        "from b import *\ndef helper(): pass\ndef caller(): helper()\n",
        "@decorate\ndef helper(): pass\ndef caller(): helper()\n",
        "class helper: pass\ndef caller(): helper()\n",
        "def helper(): pass\ndef caller():\n def nested(): helper()\n",
        "def helper(): pass\nclass C:\n def method(self): helper()\n",
        "def helper(): pass\ndef caller():\n obj.helper()\n"
        " self.helper()\n cls.helper()\n funcs[0]()\n",
        "import pkg.b\ndef caller(): pkg.b.helper()\n",
        "from os import helper\ndef caller(): helper()\n",
    ],
)
def test_python_unsupported_and_ambiguous(source: str) -> None:
    assert (
        _edges(
            {
                "a.py": source,
                "b.py": "def helper(): pass\n",
                "os.py": "def helper(): pass\n",
            }
        )
        == []
    )


def test_python_relative_parent_and_explicit_dotted_alias() -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg/sub/__init__.py": "",
        "pkg/b.py": "def helper(): pass\n",
        "pkg/sub/a.py": (
            "from ..b import helper\nimport pkg.b as mod\n"
            "def caller():\n helper()\n mod.helper()\n"
        ),
    }
    assert _edges(files) == [("pkg/sub/a.py", "caller", "pkg/b.py", "helper")]


@pytest.mark.parametrize("suffix", ["js", "ts"])
@pytest.mark.parametrize(
    ("statement", "call", "definition"),
    [
        ('import {helper} from "./b";', "helper()", "export function helper() {}"),
        ('import {helper as run} from "./b";', "run()", "export function helper() {}"),
        ('import run from "./b";', "run()", "export default function helper() {}"),
        ('import * as ns from "./b";', "ns.helper()", "export function helper() {}"),
        ('import {helper} from "./b";', "helper()", "export const helper = () => 1;"),
    ],
)
def test_js_imports(suffix: str, statement: str, call: str, definition: str) -> None:
    assert _edges(
        {
            f"a.{suffix}": f"{statement}\nfunction caller() {{ {call}; }}",
            f"b.{suffix}": definition,
        }
    ) == [(f"a.{suffix}", "caller", f"b.{suffix}", "helper")]


@pytest.mark.parametrize("suffix", ["js", "ts"])
@pytest.mark.parametrize(
    "definition",
    [
        "function helper() { helper(); }",
        "const helper = () => helper();",
        "const helper = function() { helper(); };",
        "export const helper = () => helper();",
    ],
)
def test_js_local_const_and_recursion(suffix: str, definition: str) -> None:
    path = f"a.{suffix}"
    assert _edges({path: definition + "\nfunction caller() {helper(); helper();}"}) == [
        (path, "helper", path, "helper"),
        (path, "caller", path, "helper"),
    ]


@pytest.mark.parametrize(
    "body",
    [
        "helper(); let helper;",
        "helper(); const helper = other;",
        "helper(); helper = other;",
        "helper(); helper++;",
        "helper(); let {helper} = other;",
        "helper(); [helper] = other;",
        "for (let helper of things) {helper();}",
        "try {helper();} catch(helper) {}",
        "function helper() {} helper();",
        "class helper {} helper();",
        "obj.helper(); this.helper(); obj[key](); new helper();",
        "const nested = () => helper();",
        "function nested() {helper();}",
    ],
)
@pytest.mark.parametrize("suffix", ["js", "ts"])
def test_js_shadowing_and_unsupported(body: str, suffix: str) -> None:
    assert (
        _edges(
            {f"a.{suffix}": "function helper() {}\nfunction caller() {" + body + "}"}
        )
        == []
    )


@pytest.mark.parametrize(
    "source",
    [
        'import type {helper} from "./b"; function caller(){helper();}',
        'import {type helper} from "./b"; function caller(){helper();}',
        'import {helper} from "external"; function caller(){helper();}',
        'const helper = require("./b"); function caller(){helper();}',
        'export {helper} from "./b"; function caller(){helper();}',
        "function helper(){} function caller(helper){helper();}",
        "function helper(){} function caller(){helper();} helper = other;",
        "function helper(){} function helper(){} function caller(){helper();}",
        "let helper = () => 1; function caller(){helper();}",
        'function caller(){import("./b");}',
    ],
)
def test_typescript_exclusions(source: str) -> None:
    assert _edges({"a.ts": source, "b.ts": "export function helper(){}"}) == []


@pytest.mark.parametrize(
    "definition",
    [
        "function helper() {}",
        'export {helper} from "./c";',
        'import {helper} from "./c";',
    ],
)
def test_no_private_export_or_reexport_inference(definition: str) -> None:
    assert (
        _edges(
            {
                "a.js": 'import {helper} from "./b"; function caller(){helper();}',
                "b.js": definition,
                "c.js": "export function helper(){}",
            }
        )
        == []
    )


def test_ambiguous_file_and_persisted_identity() -> None:
    assert (
        _edges(
            {
                "a.js": 'import {helper} from "./b"; function caller(){helper();}',
                "b.js": "export function helper(){}",
                "b.ts": "export function helper(){}",
            }
        )
        == []
    )
    analysis = _analyze("a.py", "def helper(): helper()\n", duplicate=True)
    assert analysis.endpoints == ()
    assert resolve_calls([analysis], {"a.py"}) == []


def test_bounds_and_stable_order(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "b.py": "def helper(): pass\ndef other(): pass\n",
        "a.py": (
            "from b import helper, other\ndef caller():\n"
            " other()\n helper()\n other()\n"
        ),
    }
    expected = [
        ("a.py", "caller", "b.py", "other"),
        ("a.py", "caller", "b.py", "helper"),
    ]
    assert _edges(files) == expected
    assert _edges(dict(reversed(list(files.items())))) == expected
    monkeypatch.setattr(module, "MAX_EDGES_PER_REPOSITORY", 1)
    assert _edges(files) == expected[:1]
    monkeypatch.setattr(module, "MAX_CALLS_PER_FILE", 2)
    assert _edges(files) == []


def test_nested_bound_and_sources_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    source = (
        "def helper(): pass\ndef caller():\n def nested():\n"
        "  helper()\n  helper()\n helper()\n"
    )
    files = {"a.py": source}
    assert _edges(files) == [("a.py", "caller", "a.py", "helper")]
    assert files["a.py"] == source
    monkeypatch.setattr(module, "MAX_CALLS_PER_FILE", 2)
    assert _edges(files) == []


@pytest.mark.parametrize(
    "module_path",
    ["../b", "/b.js", "https://x/b.js", "@alias/b", "./b.js/", "./b.json"],
)
def test_module_resolution_stays_within_day36(module_path: str) -> None:
    assert (
        _edges(
            {
                "a.js": (
                    f'import {{helper}} from "{module_path}"; '
                    "function caller(){helper();}"
                ),
                "b.js": "export function helper(){}",
            }
        )
        == []
    )


def test_python_target_reexport_and_module_ambiguity() -> None:
    caller = "from b import helper\ndef caller(): helper()\n"
    assert (
        _edges(
            {
                "a.py": caller,
                "b.py": "from c import helper\n",
                "c.py": "def helper(): pass\n",
            }
        )
        == []
    )
    assert (
        _edges(
            {
                "a.py": caller,
                "b.py": "def helper(): pass\n",
                "b/__init__.py": "def helper(): pass\n",
            }
        )
        == []
    )


def test_same_names_in_separate_modules_are_not_conflated() -> None:
    assert _edges(
        {
            "a.py": (
                "from b import helper as bhelp\nfrom c import helper as chelp\n"
                "def caller():\n bhelp()\n chelp()\n"
            ),
            "b.py": "def helper(): pass\n",
            "c.py": "def helper(): pass\n",
        }
    ) == [("a.py", "caller", "b.py", "helper"), ("a.py", "caller", "c.py", "helper")]


@pytest.mark.parametrize("suffix", ["js", "ts"])
def test_js_multiple_const_bindings_map_to_parser_spans(suffix: str) -> None:
    path = f"a.{suffix}"
    assert _edges(
        {path: "export const helper = () => 1, caller = () => helper();"}
    ) == [(path, "caller", path, "helper")]


@pytest.mark.parametrize("suffix", ["js", "ts"])
def test_closure_reassignment_does_not_guess_outer_binding(suffix: str) -> None:
    assert (
        _edges(
            {
                f"a.{suffix}": (
                    "function helper(){} function caller(){"
                    "function nested(){helper = other;} nested(); helper();}"
                )
            }
        )
        == []
    )


def test_invalid_unsupported_language_and_duplicate_analysis() -> None:
    file = File(id=uuid4(), repository_id=uuid4(), path="a.py")
    assert (
        analyze_calls(
            content="def broken(", file=file, language="python", units=[]
        ).endpoints
        == ()
    )
    analysis = analyze_calls(content="", file=file, language="config", units=[])
    assert resolve_calls([analysis], {"a.py"}) == []
    with pytest.raises(ValueError, match="unique paths"):
        resolve_calls([analysis, analysis], {"a.py"})


def test_declared_bounds_and_exact_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    assert module.MAX_CALLS_PER_FILE == 10_000
    assert module.MAX_EDGES_PER_REPOSITORY == 10_000
    monkeypatch.setattr(module, "MAX_CALLS_PER_FILE", 1)
    assert _edges({"a.py": "def helper(): helper()\n"}) == [
        ("a.py", "helper", "a.py", "helper")
    ]


def test_python_default_walrus_rebinding_is_not_ignored() -> None:
    assert (
        _edges(
            {"a.py": "def helper(): pass\ndef caller(x=(helper := other)): helper()\n"}
        )
        == []
    )


@pytest.mark.parametrize("suffix", ["js", "ts"])
@pytest.mark.parametrize("operator", ["of", "in"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("target", ["helper", "[helper]", "{helper}"])
def test_loop_assignment_invalidates_call_binding(
    suffix: str, operator: str, nested: bool, target: str
) -> None:
    loop = f"for ({target} {operator} values) {{}}"
    body = f"function reset() {{{loop}}} reset();" if nested else loop
    source = f"function helper() {{}} function caller() {{{body} helper();}}"
    assert _edges({f"a.{suffix}": source}) == []


@pytest.mark.parametrize("suffix", ["js", "ts"])
@pytest.mark.parametrize("operator", ["of", "in"])
@pytest.mark.parametrize("kind", ["const", "let", "var"])
def test_nested_loop_declaration_does_not_rebind_module_function(
    suffix: str, operator: str, kind: str
) -> None:
    path = f"a.{suffix}"
    source = (
        "function helper() {} function caller() {"
        f"function nested() {{for ({kind} helper {operator} values) {{helper();}}}}"
        "helper();}"
    )
    assert _edges({path: source}) == [(path, "caller", path, "helper")]


@pytest.mark.parametrize("suffix", ["js", "ts"])
@pytest.mark.parametrize("operator", ["of", "in"])
@pytest.mark.parametrize("kind", ["const", "let", "var"])
def test_caller_local_loop_declaration_still_shadows(
    suffix: str, operator: str, kind: str
) -> None:
    source = (
        "function helper() {} function caller() {"
        f"for ({kind} helper {operator} values) {{helper();}}"
        "}"
    )
    assert _edges({f"a.{suffix}": source}) == []


@pytest.mark.parametrize("suffix", ["js", "ts"])
@pytest.mark.parametrize("target", ["item", "const item"])
def test_unrelated_loop_binding_preserves_call(suffix: str, target: str) -> None:
    path = f"a.{suffix}"
    source = (
        "function helper() {} function caller() {"
        f"for ({target} of values) {{}} helper();"
        "}"
    )
    assert _edges({path: source}) == [(path, "caller", path, "helper")]


@pytest.mark.parametrize("local", [False, True])
def test_python_type_alias_shadows_function(local: bool) -> None:
    source = (
        "def helper(): pass\ndef caller():\n type helper = int\n helper()\n"
        if local
        else "def helper(): pass\ntype helper = int\ndef caller(): helper()\n"
    )
    assert _edges({"a.py": source}) == []


@pytest.mark.parametrize("prefix", ["", "async "])
@pytest.mark.parametrize("parameters", ["helper", "T, helper", "helper, T"])
def test_python_generic_parameters_shadow_function(
    prefix: str, parameters: str
) -> None:
    source = f"def helper(): pass\n{prefix}def caller[{parameters}](): helper()\n"
    assert _edges({"a.py": source}) == []


@pytest.mark.parametrize("prefix", ["", "async "])
def test_unrelated_python_type_bindings_preserve_call(prefix: str) -> None:
    source = (
        f"def helper(): pass\ntype Alias = int\n{prefix}def caller[T, U](): helper()\n"
    )
    assert _edges({"a.py": source}) == [("a.py", "caller", "a.py", "helper")]


@pytest.mark.parametrize("prefix", ["", "async "])
@pytest.mark.parametrize(
    "parameters",
    ["T: helper", "T, U: helper", "T: (helper, int)", "T: list[helper]", "T: int"],
)
def test_python_generic_bounds_are_references(prefix: str, parameters: str) -> None:
    source = f"def helper(): pass\n{prefix}def caller[{parameters}](): helper()\n"
    assert _edges({"a.py": source}) == [("a.py", "caller", "a.py", "helper")]


@pytest.mark.parametrize("prefix", ["", "async "])
@pytest.mark.parametrize(
    "parameters", ["helper: int", "T: int, helper: str", "*helper", "**helper"]
)
def test_python_declared_generic_name_still_shadows(
    prefix: str, parameters: str
) -> None:
    source = f"def helper(): pass\n{prefix}def caller[{parameters}](): helper()\n"
    assert _edges({"a.py": source}) == []


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize(
    "alias",
    [
        "Alias[helper] = list[helper]",
        "Alias[T, helper] = tuple[T, helper]",
        "Alias[T: helper] = list[T]",
        "Alias[T] = tuple[T, helper]",
        "Alias = helper",
    ],
)
def test_python_alias_parameters_and_rhs_are_not_enclosing_bindings(
    local: bool, alias: str
) -> None:
    source = (
        f"def helper(): pass\ndef caller():\n type {alias}\n helper()\n"
        if local
        else f"def helper(): pass\ntype {alias}\ndef caller(): helper()\n"
    )
    assert _edges({"a.py": source}) == [("a.py", "caller", "a.py", "helper")]


@pytest.mark.parametrize("local", [False, True])
def test_python_generic_alias_name_still_shadows(local: bool) -> None:
    source = (
        "def helper(): pass\ndef caller():\n type helper[T] = list[T]\n helper()\n"
        if local
        else "def helper(): pass\ntype helper[T] = list[T]\ndef caller(): helper()\n"
    )
    assert _edges({"a.py": source}) == []


@pytest.mark.parametrize("suffix", ["js", "ts"])
@pytest.mark.parametrize(
    "caller",
    [
        "function caller() {arguments();}",
        "const caller = function () {arguments();};",
    ],
)
@pytest.mark.parametrize(
    "definition",
    [
        "function arguments() {}",
        "const arguments = () => {};",
    ],
)
def test_ordinary_function_arguments_is_not_module_binding(
    suffix: str, caller: str, definition: str
) -> None:
    assert _edges({f"a.{suffix}": definition + caller}) == []


@pytest.mark.parametrize("suffix", ["js", "ts"])
def test_arrow_does_not_introduce_arguments_binding(suffix: str) -> None:
    path = f"a.{suffix}"
    source = "const arguments = () => {}; const caller = () => arguments();"
    assert _edges({path: source}) == [(path, "caller", path, "arguments")]
