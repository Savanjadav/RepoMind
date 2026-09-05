import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from app.code_parser import CodeParsingError, CodeUnitKind, ParsedCodeUnit

_DEFINITION_KINDS = {
    "abstract_class_declaration": CodeUnitKind.CLASS,
    "class_declaration": CodeUnitKind.CLASS,
    "function_declaration": CodeUnitKind.FUNCTION,
    "generator_function_declaration": CodeUnitKind.FUNCTION,
}
_FUNCTION_VALUE_TYPES = {
    "arrow_function",
    "function_expression",
    "generator_function",
}
_VARIABLE_DECLARATION_TYPES = {
    "lexical_declaration",
    "variable_declaration",
}
_SIMPLE_METHOD_NAME_TYPES = {
    "private_property_identifier",
    "property_identifier",
}
_LANGUAGE_DISPLAY_NAMES = {
    "javascript": "JavaScript",
    "typescript": "TypeScript",
}


class JavaScriptCodeParser:
    def __init__(self) -> None:
        language = Language(tree_sitter_javascript.language())
        self._engine = _JavaScriptTypeScriptParser(
            language=language,
            language_label="javascript",
        )

    def parse(
        self,
        *,
        content: str,
        relative_path: str,
    ) -> list[ParsedCodeUnit]:
        return self._engine.parse(content=content, relative_path=relative_path)


class TypeScriptCodeParser:
    def __init__(self) -> None:
        language = Language(tree_sitter_typescript.language_typescript())
        self._engine = _JavaScriptTypeScriptParser(
            language=language,
            language_label="typescript",
        )

    def parse(
        self,
        *,
        content: str,
        relative_path: str,
    ) -> list[ParsedCodeUnit]:
        return self._engine.parse(content=content, relative_path=relative_path)


class _JavaScriptTypeScriptParser:
    def __init__(self, *, language: Language, language_label: str) -> None:
        self._parser = Parser(language)
        self._language_label = language_label
        self._language_display_name = _LANGUAGE_DISPLAY_NAMES[language_label]

    def parse(
        self,
        *,
        content: str,
        relative_path: str,
    ) -> list[ParsedCodeUnit]:
        try:
            source_bytes = content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise CodeParsingError(
                f"{self._language_display_name} source is not valid UTF-8 text"
            ) from error

        tree = self._parser.parse(source_bytes)
        if tree is None:
            raise CodeParsingError(
                f"{self._language_display_name} source could not be parsed"
            )
        if tree.root_node.has_error:
            raise CodeParsingError(
                f"{self._language_display_name} source contains syntax errors"
            )

        extracted: list[tuple[int, int, ParsedCodeUnit]] = []
        nodes: list[tuple[Node, bool]] = [(tree.root_node, False)]
        while nodes:
            node, suppress_emission = nodes.pop()

            if node.type == "ambient_declaration":
                continue

            if node.type == "export_statement":
                self._handle_export(
                    node=node,
                    source_bytes=source_bytes,
                    relative_path=relative_path,
                    extracted=extracted,
                    nodes=nodes,
                )
                continue

            if not suppress_emission:
                self._emit_node(
                    node=node,
                    source_bytes=source_bytes,
                    relative_path=relative_path,
                    extracted=extracted,
                )

            nodes.extend((child, False) for child in reversed(node.named_children))

        extracted.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in extracted]

    def _handle_export(
        self,
        *,
        node: Node,
        source_bytes: bytes,
        relative_path: str,
        extracted: list[tuple[int, int, ParsedCodeUnit]],
        nodes: list[tuple[Node, bool]],
    ) -> None:
        declaration = node.child_by_field_name("declaration")
        if declaration is None:
            nodes.extend((child, False) for child in reversed(node.named_children))
            return

        if declaration.type in _DEFINITION_KINDS:
            self._append_definition(
                content_node=node,
                definition_node=declaration,
                source_bytes=source_bytes,
                relative_path=relative_path,
                extracted=extracted,
            )
            replacement_children = [
                (child, False) for child in declaration.named_children
            ]
            self._push_export_children(
                export_node=node,
                declaration=declaration,
                replacement_children=replacement_children,
                nodes=nodes,
            )
            return

        if declaration.type in _VARIABLE_DECLARATION_TYPES:
            declarators = _direct_variable_declarators(declaration)
            eligible = [item for item in declarators if _is_function_declarator(item)]
            use_export_span = len(declarators) == 1 and len(eligible) == 1
            for declarator in eligible:
                self._append_variable_function(
                    content_node=node if use_export_span else declarator,
                    declarator=declarator,
                    source_bytes=source_bytes,
                    relative_path=relative_path,
                    extracted=extracted,
                )
            replacement_children = [
                (child, child in eligible) for child in declaration.named_children
            ]
            self._push_export_children(
                export_node=node,
                declaration=declaration,
                replacement_children=replacement_children,
                nodes=nodes,
            )
            return

        nodes.extend((child, False) for child in reversed(node.named_children))

    @staticmethod
    def _push_export_children(
        *,
        export_node: Node,
        declaration: Node,
        replacement_children: list[tuple[Node, bool]],
        nodes: list[tuple[Node, bool]],
    ) -> None:
        children: list[tuple[Node, bool]] = []
        for child in export_node.named_children:
            if child == declaration:
                children.extend(replacement_children)
            else:
                children.append((child, False))
        nodes.extend(reversed(children))

    def _emit_node(
        self,
        *,
        node: Node,
        source_bytes: bytes,
        relative_path: str,
        extracted: list[tuple[int, int, ParsedCodeUnit]],
    ) -> None:
        if node.type in _DEFINITION_KINDS:
            self._append_definition(
                content_node=node,
                definition_node=node,
                source_bytes=source_bytes,
                relative_path=relative_path,
                extracted=extracted,
            )
            return

        if node.type == "method_definition" and _is_class_method(node):
            name_node = node.child_by_field_name("name")
            if name_node is None or name_node.type not in _SIMPLE_METHOD_NAME_TYPES:
                return
            self._append_unit(
                content_node=node,
                kind=CodeUnitKind.FUNCTION,
                name_node=name_node,
                source_bytes=source_bytes,
                relative_path=relative_path,
                extracted=extracted,
            )
            return

        if node.type == "import_statement":
            self._append_unit(
                content_node=node,
                kind=CodeUnitKind.IMPORT,
                name_node=None,
                source_bytes=source_bytes,
                relative_path=relative_path,
                extracted=extracted,
            )
            return

        if node.type == "variable_declarator" and _is_function_declarator(node):
            declaration = node.parent
            if (
                declaration is None
                or declaration.type not in _VARIABLE_DECLARATION_TYPES
            ):
                return
            declarators = _direct_variable_declarators(declaration)
            self._append_variable_function(
                content_node=declaration if len(declarators) == 1 else node,
                declarator=node,
                source_bytes=source_bytes,
                relative_path=relative_path,
                extracted=extracted,
            )

    def _append_definition(
        self,
        *,
        content_node: Node,
        definition_node: Node,
        source_bytes: bytes,
        relative_path: str,
        extracted: list[tuple[int, int, ParsedCodeUnit]],
    ) -> None:
        name_node = definition_node.child_by_field_name("name")
        if name_node is None:
            raise CodeParsingError(
                f"{self._language_display_name} definition is missing its name"
            )
        self._append_unit(
            content_node=content_node,
            kind=_DEFINITION_KINDS[definition_node.type],
            name_node=name_node,
            source_bytes=source_bytes,
            relative_path=relative_path,
            extracted=extracted,
        )

    def _append_variable_function(
        self,
        *,
        content_node: Node,
        declarator: Node,
        source_bytes: bytes,
        relative_path: str,
        extracted: list[tuple[int, int, ParsedCodeUnit]],
    ) -> None:
        name_node = declarator.child_by_field_name("name")
        if name_node is None or name_node.type != "identifier":
            return
        self._append_unit(
            content_node=content_node,
            kind=CodeUnitKind.FUNCTION,
            name_node=name_node,
            source_bytes=source_bytes,
            relative_path=relative_path,
            extracted=extracted,
        )

    def _append_unit(
        self,
        *,
        content_node: Node,
        kind: CodeUnitKind,
        name_node: Node | None,
        source_bytes: bytes,
        relative_path: str,
        extracted: list[tuple[int, int, ParsedCodeUnit]],
    ) -> None:
        unit_bytes = _node_bytes(source_bytes, content_node)
        start_line, end_line = _inclusive_line_range(content_node, unit_bytes)
        symbol_name = (
            None
            if name_node is None
            else _node_bytes(source_bytes, name_node).decode("utf-8")
        )
        unit = ParsedCodeUnit(
            kind=kind,
            content=unit_bytes.decode("utf-8"),
            relative_path=relative_path,
            language=self._language_label,
            start_line=start_line,
            end_line=end_line,
            symbol_name=symbol_name,
        )
        extracted.append((content_node.start_byte, content_node.end_byte, unit))


def _direct_variable_declarators(declaration: Node) -> list[Node]:
    return [
        child
        for child in declaration.named_children
        if child.type == "variable_declarator"
    ]


def _is_function_declarator(node: Node) -> bool:
    name_node = node.child_by_field_name("name")
    value_node = node.child_by_field_name("value")
    return (
        name_node is not None
        and name_node.type == "identifier"
        and value_node is not None
        and value_node.type in _FUNCTION_VALUE_TYPES
    )


def _is_class_method(node: Node) -> bool:
    parent = node.parent
    return parent is not None and parent.type == "class_body"


def _node_bytes(source_bytes: bytes, node: Node) -> bytes:
    return source_bytes[node.start_byte : node.end_byte]


def _inclusive_line_range(node: Node, unit_bytes: bytes) -> tuple[int, int]:
    start_line = node.start_point.row + 1
    end_line = start_line + unit_bytes.count(b"\n")
    if unit_bytes.endswith(b"\n") and end_line > start_line:
        end_line -= 1
    return start_line, end_line
