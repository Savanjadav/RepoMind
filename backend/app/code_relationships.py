"""Bounded static call hints, not runtime dispatch or compiler symbol binding."""

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
from sqlalchemy.orm import Session
from tree_sitter import Language, Node, Parser

from app.import_relationships import ImportReference, resolve_imports
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import CodeUnit
from app.models.file import File

MAX_CALLS_PER_FILE = 10_000
MAX_EDGES_PER_REPOSITORY = 10_000
_SCOPES = frozenset(
    {
        "function_definition",
        "class_definition",
        "lambda",
        "function_declaration",
        "generator_function_declaration",
        "function_expression",
        "generator_function",
        "arrow_function",
        "class_declaration",
        "class",
        "method_definition",
        "abstract_class_declaration",
        "function_signature",
    }
)
_FUNCTIONS = {"function_definition", "function_declaration"}
_GRAMMARS = {
    "python": tree_sitter_python.language,
    "javascript": tree_sitter_javascript.language,
    "typescript": tree_sitter_typescript.language_typescript,
}


@dataclass(frozen=True, slots=True)
class CallableEndpoint:
    file_id: UUID
    code_unit_id: UUID
    name: str
    start_line: int
    end_line: int
    export: str | None


@dataclass(frozen=True, slots=True)
class ImportBinding:
    local: str
    module: str
    symbol: str | None  # None means an explicit module/namespace binding.


@dataclass(frozen=True, slots=True)
class CallReference:
    caller: CallableEndpoint
    binding: str
    member: str | None


@dataclass(frozen=True, slots=True)
class FileCallAnalysis:
    path: str
    language: str
    endpoints: tuple[CallableEndpoint, ...]
    imports: tuple[ImportBinding, ...]
    calls: tuple[CallReference, ...]


def _text(node: Node | None) -> str:
    return "" if node is None or node.text is None else node.text.decode("utf-8")


def _walk(node: Node, *, scopes: bool = False) -> Iterator[Node]:
    pending = [node]
    while pending:
        current = pending.pop()
        yield current
        if scopes or current.type not in _SCOPES:
            pending.extend(reversed(current.named_children))


def _names(node: Node | None) -> set[str]:
    # Over-collecting identifiers in a binding pattern deliberately loses edges,
    # rather than guessing about destructuring/defaults or computed properties.
    return (
        set()
        if node is None
        else {
            _text(item)
            for item in _walk(node)
            if item.type
            in {
                "identifier",
                "type_identifier",
                "shorthand_property_identifier_pattern",
            }
        }
    )


def _python_type_binding_name(node: Node | None) -> str | None:
    """Read the declaration side only, never bounds or alias parameters."""
    while node is not None and node.type in {
        "type",
        "constrained_type",
        "splat_type",
        "generic_type",
    }:
        node = node.named_children[0] if node.named_children else None
    return _text(node) if node is not None and node.type == "identifier" else None


def _python_type_parameter_names(node: Node | None) -> set[str]:
    if node is None:
        return set()
    return {
        name
        for parameter in node.named_children
        if (name := _python_type_binding_name(parameter)) is not None
    }


def _invalidated(node: Node, ignored: set[int] | None = None) -> set[str]:
    """Names bound or modified anywhere in this scope, irrespective of order."""
    invalid: set[str] = set()
    pending = [node]
    fields = {
        "assignment": "left",
        "augmented_assignment": "left",
        "named_expression": "name",
        "assignment_expression": "left",
        "augmented_assignment_expression": "left",
        "variable_declarator": "name",
        "for_statement": "left",
        "for_in_statement": "left",
        "for_in_clause": "left",
        "as_pattern": "alias",
        "catch_clause": "parameter",
        "update_expression": "argument",
    }
    while pending:
        item = pending.pop()
        if item != node and ignored and item.start_byte in ignored:
            continue
        if item.type in _SCOPES:
            invalid.update(_names(item.child_by_field_name("name")))
            continue
        if item.type in fields:
            invalid.update(_names(item.child_by_field_name(fields[item.type])))
        if item.type == "type_alias_statement":
            name = _python_type_binding_name(item.child_by_field_name("left"))
            if name is not None:
                invalid.add(name)
        if item.type in {
            "delete_statement",
            "global_statement",
            "nonlocal_statement",
            "import_statement",
            "import_from_statement",
            "future_import_statement",
            "with_statement",
            "match_statement",
            "ambient_declaration",
            "enum_declaration",
            "internal_module",
        }:
            invalid.update(_names(item))
        if item.type == "unary_expression" and any(
            c.type == "delete" for c in item.children
        ):
            invalid.update(_names(item))
        pending.extend(reversed(item.named_children))
    return invalid


def _import_bindings(node: Node, language: str) -> list[ImportBinding]:
    bindings: list[ImportBinding] = []
    if language == "python":
        module = _text(node.child_by_field_name("module_name"))
        for item in node.children_by_field_name("name"):
            alias = item.child_by_field_name("alias")
            original = item.child_by_field_name("name") if alias else item
            name = _text(original)
            local = _text(alias) if alias else name
            if node.type == "import_statement":
                if "." not in name or alias:
                    bindings.append(ImportBinding(local, name, None))
            elif node.type == "import_from_statement" and name.isidentifier():
                bindings.append(ImportBinding(local, module, name))
        return bindings
    source = _text(node.child_by_field_name("source"))
    if not source or "\\" in source or any(c.type == "type" for c in node.children):
        return []
    module = source[1:-1]
    for item in node.named_children:
        if item.type != "import_clause":
            continue
        for child in item.named_children:
            if child.type == "identifier":
                bindings.append(ImportBinding(_text(child), module, "default"))
            elif child.type == "namespace_import":
                bindings.append(
                    ImportBinding(_text(child.named_children[-1]), module, None)
                )
            elif child.type == "named_imports":
                for specifier in child.named_children:
                    if any(c.type == "type" for c in specifier.children):
                        continue
                    imported_name = specifier.child_by_field_name("name")
                    alias = specifier.child_by_field_name("alias")
                    if imported_name is not None and imported_name.type == "identifier":
                        bindings.append(
                            ImportBinding(
                                _text(alias or imported_name),
                                module,
                                _text(imported_name),
                            )
                        )
    return bindings


def _declarations(
    root: Node, language: str
) -> Iterator[tuple[Node, Node, str, str | None]]:
    """Yield (callable, parser-compatible content span, binding, direct export)."""
    for wrapper in root.named_children:
        node = wrapper
        export: str | None = None
        if wrapper.type == "export_statement":
            declaration = wrapper.child_by_field_name("declaration")
            if declaration is None:
                continue
            node = declaration
            export = (
                "default"
                if any(c.type == "default" for c in wrapper.children)
                else "named"
            )
        if node.type in _FUNCTIONS:
            name = _text(node.child_by_field_name("name"))
            if name:
                yield node, wrapper, name, name if export == "named" else export
        elif language != "python" and node.type == "lexical_declaration":
            if not any(c.type == "const" for c in node.children):
                continue
            declarations = [
                c for c in node.named_children if c.type == "variable_declarator"
            ]
            for item in declarations:
                name_node = item.child_by_field_name("name")
                value = item.child_by_field_name("value")
                if (
                    name_node is not None
                    and name_node.type == "identifier"
                    and value is not None
                    and value.type in {"arrow_function", "function_expression"}
                ):
                    yield (
                        value,
                        wrapper if len(declarations) == 1 else item,
                        _text(name_node),
                        _text(name_node) if export == "named" else None,
                    )


def analyze_calls(
    *,
    content: str,
    file: File,
    language: str,
    units: Sequence[CodeUnit],
) -> FileCallAnalysis:
    """Analyze once while safely read source is available; retain no AST nodes."""
    empty = FileCallAnalysis(file.path, language, (), (), ())
    if language not in _GRAMMARS:
        return empty
    root = (
        Parser(Language(_GRAMMARS[language]())).parse(content.encode("utf-8")).root_node
    )
    if root.has_error:
        return empty
    # Count all occurrences, including unsupported scopes, before any resolution.
    count = 0
    global_invalid: set[str] = set()
    wildcard = False
    for item in _walk(root, scopes=True):
        if item.type in {"call", "call_expression"}:
            count += 1
            if count > MAX_CALLS_PER_FILE:
                return empty
        if item.type in {"global_statement", "nonlocal_statement"}:
            global_invalid.update(_names(item))
        # JS writes may affect an enclosing binding, including from closures.
        # Without a scope/control-flow engine, invalidate the name file-wide.
        if language != "python" and item.type in {
            "assignment_expression",
            "augmented_assignment_expression",
            "update_expression",
        }:
            global_invalid.update(
                _names(
                    item.child_by_field_name("left")
                    or item.child_by_field_name("argument")
                )
            )
        if (
            language != "python"
            and item.type == "for_in_statement"
            and item.child_by_field_name("kind") is None
        ):
            global_invalid.update(_names(item.child_by_field_name("left")))
        # A Python walrus can also bind outside a callable (e.g. its defaults).
        if item.type == "named_expression":
            global_invalid.update(_names(item.child_by_field_name("name")))
        if item.type == "wildcard_import":
            wildcard = True
    if wildcard:
        return empty

    declarations = list(_declarations(root, language))
    imports: list[ImportBinding] = []
    ignored: set[int] = set()
    bindings: dict[str, int] = defaultdict(int)
    for callable_node, span, name, _ in declarations:
        # Skip only the eligible binding, not sibling variable declarations.
        parent = callable_node.parent
        ignored.add(
            parent.start_byte
            if parent is not None and parent.type == "variable_declarator"
            else span.start_byte
        )
        bindings[name] += 1
    for node in root.named_children:
        if node.type in {"import_statement", "import_from_statement"}:
            found = _import_bindings(node, language)
            imports.extend(found)
            for imported in found:
                bindings[imported.local] += 1
            # Unsupported/type-only bindings still invalidate names they occupy.
            unsupported = _names(node) - {binding.local for binding in found}
            global_invalid.update(unsupported)
            ignored.add(node.start_byte)
    global_invalid.update(_invalidated(root, ignored))
    global_invalid.update(name for name, count in bindings.items() if count != 1)

    endpoints: list[CallableEndpoint] = []
    calls: list[CallReference] = []
    identities: dict[tuple[str | None, int, int], list[CodeUnit]] = defaultdict(list)
    for unit in units:
        if unit.file_id == file.id and unit.kind == "function":
            identities[(unit.symbol_name, unit.start_line, unit.end_line)].append(unit)
    for node, span, name, export in declarations:
        if name in global_invalid:
            continue
        raw = span.text or b""
        start = span.start_point.row + 1
        end = start + raw.count(b"\n") - int(raw.endswith(b"\n"))
        matches = identities.get((name, start, end), [])
        if len(matches) != 1:
            continue
        endpoint = CallableEndpoint(file.id, matches[0].id, name, start, end, export)
        endpoints.append(endpoint)
        body = node.child_by_field_name("body")
        if body is None:
            continue
        invalid = global_invalid | _invalidated(body)
        invalid.update(_names(node.child_by_field_name("parameters")))
        invalid.update(_names(node.child_by_field_name("parameter")))
        if language == "python":
            invalid.update(
                _python_type_parameter_names(
                    node.child_by_field_name("type_parameters")
                )
            )
        elif node.type in {"function_declaration", "function_expression"}:
            invalid.add("arguments")
        if node.type == "function_expression":
            invalid.update(_names(node.child_by_field_name("name")))
        # Comprehensions and JS with have implicit/dynamic scope; skip their
        # entire caller rather than approximating scope or control flow.
        if any(
            item.type
            in {
                "list_comprehension",
                "set_comprehension",
                "dictionary_comprehension",
                "generator_expression",
                "with_statement",
            }
            for item in _walk(body)
        ):
            continue
        for item in _walk(body):
            if item.type not in {"call", "call_expression"}:
                continue
            function = item.child_by_field_name("function")
            member: str | None = None
            binding = ""
            if function is not None and function.type == "identifier":
                binding = _text(function)
            elif function is not None and function.type in {
                "attribute",
                "member_expression",
            }:
                obj = function.child_by_field_name("object")
                prop = function.child_by_field_name(
                    "attribute"
                ) or function.child_by_field_name("property")
                if (
                    obj is not None
                    and obj.type == "identifier"
                    and prop is not None
                    and prop.type in {"identifier", "property_identifier"}
                ):
                    binding, member = _text(obj), _text(prop)
            if binding and binding not in invalid:
                calls.append(CallReference(endpoint, binding, member))
    return FileCallAnalysis(
        file.path,
        language,
        tuple(endpoints),
        tuple(b for b in imports if b.local not in global_invalid),
        tuple(calls),
    )


def resolve_calls(
    analyses: Sequence[FileCallAnalysis],
    paths: set[str],
) -> list[tuple[CallableEndpoint, CallableEndpoint]]:
    """Resolve only explicit bindings; Day 36 owns local module resolution."""
    by_path = {analysis.path: analysis for analysis in analyses}
    if len(by_path) != len(analyses):
        raise ValueError("Call analyses must have unique paths")
    edges: list[tuple[CallableEndpoint, CallableEndpoint]] = []
    seen: set[tuple[UUID, UUID]] = set()
    for analysis in sorted(analyses, key=lambda item: item.path):
        if analysis.path not in paths:
            continue
        local = {endpoint.name: endpoint for endpoint in analysis.endpoints}
        imports = {binding.local: binding for binding in analysis.imports}
        for call in analysis.calls:
            target = local.get(call.binding) if call.member is None else None
            binding = imports.get(call.binding)
            if binding is not None:
                target = None
                module_edges = resolve_imports(
                    [ImportReference(analysis.path, analysis.language, binding.module)],
                    paths,
                )
                if not module_edges or (binding.symbol is None) != (
                    call.member is not None
                ):
                    continue
                other = by_path.get(module_edges[0][1])
                if (
                    other is None
                    or other.language != analysis.language
                    and {other.language, analysis.language}
                    != {"javascript", "typescript"}
                ):
                    continue
                symbol = call.member if binding.symbol is None else binding.symbol
                matches = [
                    endpoint
                    for endpoint in other.endpoints
                    if (
                        endpoint.name
                        if analysis.language == "python"
                        else endpoint.export
                    )
                    == symbol
                ]
                if len(matches) == 1:
                    target = matches[0]
            if (
                target is not None
                and (call.caller.code_unit_id, target.code_unit_id) not in seen
            ):
                seen.add((call.caller.code_unit_id, target.code_unit_id))
                edges.append((call.caller, target))
                if len(edges) >= MAX_EDGES_PER_REPOSITORY:
                    return edges
    return edges


def persist_call_relationships(
    session: Session,
    *,
    repository_id: UUID,
    files_by_path: Mapping[str, File],
    analyses: Sequence[FileCallAnalysis],
) -> None:
    if any(
        file.repository_id != repository_id or file.path != path
        for path, file in files_by_path.items()
    ):
        raise ValueError("Call files must belong to the repository and match paths")
    if any(
        analysis.path not in files_by_path
        or any(
            endpoint.file_id != files_by_path[analysis.path].id
            for endpoint in analysis.endpoints
        )
        for analysis in analyses
    ):
        raise ValueError("Call endpoints must belong to their discovered file")
    session.add_all(
        [
            CodeRelationship(
                repository_id=repository_id,
                source_file_id=source.file_id,
                source_code_unit_id=source.code_unit_id,
                target_file_id=target.file_id,
                target_code_unit_id=target.code_unit_id,
                relationship_type="calls",
            )
            for source, target in resolve_calls(analyses, set(files_by_path))
        ]
    )
    session.flush()
