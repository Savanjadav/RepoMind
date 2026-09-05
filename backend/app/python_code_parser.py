import tree_sitter_python
from tree_sitter import Language, Node, Parser

from app.code_parser import CodeParsingError, CodeUnitKind, ParsedCodeUnit

_DEFINITION_KINDS = {
    "class_definition": CodeUnitKind.CLASS,
    "function_definition": CodeUnitKind.FUNCTION,
}
_IMPORT_NODE_TYPES = {
    "future_import_statement",
    "import_from_statement",
    "import_statement",
}


class PythonCodeParser:
    def __init__(self) -> None:
        language = Language(tree_sitter_python.language())
        self._parser = Parser(language)

    def parse(
        self,
        *,
        content: str,
        relative_path: str,
    ) -> list[ParsedCodeUnit]:
        try:
            source_bytes = content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise CodeParsingError("Python source is not valid UTF-8 text") from error

        tree = self._parser.parse(source_bytes)
        if tree is None:
            raise CodeParsingError("Python source could not be parsed")
        if tree.root_node.has_error:
            raise CodeParsingError("Python source contains syntax errors")

        extracted: list[tuple[int, int, ParsedCodeUnit]] = []
        nodes = [tree.root_node]
        while nodes:
            node = nodes.pop()

            if node.type == "decorated_definition":
                definition = node.child_by_field_name("definition")
                if definition is None or definition.type not in _DEFINITION_KINDS:
                    raise CodeParsingError(
                        "Python decorated definition is missing its definition"
                    )
                extracted.append(
                    (
                        node.start_byte,
                        node.end_byte,
                        _definition_unit(
                            source_bytes=source_bytes,
                            content_node=node,
                            definition_node=definition,
                            relative_path=relative_path,
                        ),
                    )
                )
                nodes.extend(reversed(definition.named_children))
                continue

            if node.type in _DEFINITION_KINDS:
                unit = _definition_unit(
                    source_bytes=source_bytes,
                    content_node=node,
                    definition_node=node,
                    relative_path=relative_path,
                )
                extracted.append((node.start_byte, node.end_byte, unit))
            elif node.type in _IMPORT_NODE_TYPES:
                unit = _import_unit(
                    source_bytes=source_bytes,
                    node=node,
                    relative_path=relative_path,
                )
                extracted.append((node.start_byte, node.end_byte, unit))

            nodes.extend(reversed(node.named_children))

        extracted.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in extracted]


def _definition_unit(
    *,
    source_bytes: bytes,
    content_node: Node,
    definition_node: Node,
    relative_path: str,
) -> ParsedCodeUnit:
    name_node = definition_node.child_by_field_name("name")
    if name_node is None:
        raise CodeParsingError("Python definition is missing its name")

    unit_bytes = _node_bytes(source_bytes, content_node)
    start_line, end_line = _inclusive_line_range(content_node, unit_bytes)
    return ParsedCodeUnit(
        kind=_DEFINITION_KINDS[definition_node.type],
        content=unit_bytes.decode("utf-8"),
        relative_path=relative_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        symbol_name=_node_bytes(source_bytes, name_node).decode("utf-8"),
    )


def _import_unit(
    *,
    source_bytes: bytes,
    node: Node,
    relative_path: str,
) -> ParsedCodeUnit:
    unit_bytes = _node_bytes(source_bytes, node)
    start_line, end_line = _inclusive_line_range(node, unit_bytes)
    return ParsedCodeUnit(
        kind=CodeUnitKind.IMPORT,
        content=unit_bytes.decode("utf-8"),
        relative_path=relative_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
    )


def _node_bytes(source_bytes: bytes, node: Node) -> bytes:
    return source_bytes[node.start_byte : node.end_byte]


def _inclusive_line_range(node: Node, unit_bytes: bytes) -> tuple[int, int]:
    start_line = node.start_point.row + 1
    end_line = start_line + unit_bytes.count(b"\n")
    if unit_bytes.endswith(b"\n") and end_line > start_line:
        end_line -= 1
    return start_line, end_line
