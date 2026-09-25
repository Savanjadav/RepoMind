"""Conservative structural imports; not runtime module or symbol resolution."""

import posixpath
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from uuid import UUID

import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
from sqlalchemy.orm import Session
from tree_sitter import Language, Parser

from app.code_parser import CodeUnitKind, ParsedCodeUnit
from app.models.file import File
from app.models.relationship import Relationship

_SUFFIXES = (".js", ".jsx", ".ts")


@dataclass(frozen=True, slots=True)
class ImportReference:
    source_path: str
    language: str
    module: str


def extract_imports(units: Sequence[ParsedCodeUnit]) -> list[ImportReference]:
    parsers: dict[str, Parser] = {}
    references: list[ImportReference] = []
    grammars = {
        "python": tree_sitter_python.language,
        "javascript": tree_sitter_javascript.language,
        "typescript": tree_sitter_typescript.language_typescript,
    }
    for unit in units:
        if unit.kind != CodeUnitKind.IMPORT or unit.language not in grammars:
            continue
        if unit.language not in parsers:
            parsers[unit.language] = Parser(Language(grammars[unit.language]()))
        tree = parsers[unit.language].parse(unit.content.encode("utf-8"))
        if tree.root_node.has_error:
            continue
        for node in tree.root_node.named_children:
            modules: list[str] = []
            if unit.language == "python":
                if node.type == "import_statement":
                    for name in node.children_by_field_name("name"):
                        if name.type == "aliased_import":
                            original = name.child_by_field_name("name")
                            if original is None:
                                continue
                            name = original
                        if name.text is not None:
                            modules.append(name.text.decode("utf-8"))
                elif node.type == "import_from_statement":
                    module = node.child_by_field_name("module_name")
                    if module is not None and module.text is not None:
                        modules.append(module.text.decode("utf-8"))
            elif node.type == "import_statement":
                source = node.child_by_field_name("source")
                if source is not None and source.type == "string" and source.text:
                    literal = source.text.decode("utf-8")
                    if "\\" not in literal:
                        modules.append(literal[1:-1])
            references.extend(
                ImportReference(unit.relative_path, unit.language, module)
                for module in modules
            )
    return references


def _python_target(reference: ImportReference, paths: set[str]) -> str | None:
    module = reference.module
    level = len(module) - len(module.lstrip("."))
    if level:
        package = list(PurePosixPath(reference.source_path).parent.parts)
        if not package or level > len(package):
            return None
        if any(
            "/".join(package[:i]) + "/__init__.py" not in paths
            or "/".join(package[:i]) + ".py" in paths
            for i in range(1, len(package) + 1)
        ):
            return None
        parts = package[: len(package) - level + 1]
        if module[level:]:
            parts.extend(module[level:].split("."))
    else:
        parts = module.split(".")
        if (
            parts[0] in sys.stdlib_module_names
            or parts[0] in sys.builtin_module_names
            or parts[0] == "__future__"
        ):
            return None
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    if any(
        "/".join(parts[:i]) + "/__init__.py" not in paths
        or "/".join(parts[:i]) + ".py" in paths
        for i in range(1, len(parts))
    ):
        return None
    base = "/".join(parts)
    candidates = {base + ".py", base + "/__init__.py"} & paths
    return next(iter(candidates)) if len(candidates) == 1 else None


def _javascript_target(reference: ImportReference, paths: set[str]) -> str | None:
    module = reference.module
    if not module.startswith(("./", "../")) or any(
        char in module for char in "\\?#\x00"
    ):
        return None
    directory_shaped = module.endswith("/")
    target = posixpath.normpath(
        posixpath.join(posixpath.dirname(reference.source_path), module)
    )
    if target == ".." or target.startswith(("../", "/")):
        return None
    suffix = PurePosixPath(target).suffix
    if suffix and not directory_shaped:
        return target if suffix in _SUFFIXES and target in paths else None
    candidates = (
        set() if directory_shaped else {target + suffix for suffix in _SUFFIXES}
    )
    candidates.update(posixpath.join(target, "index" + suffix) for suffix in _SUFFIXES)
    candidates = {posixpath.normpath(candidate) for candidate in candidates}
    matches = candidates & paths
    return next(iter(matches)) if len(matches) == 1 else None


def resolve_imports(
    references: Sequence[ImportReference], paths: set[str]
) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for reference in references:
        if reference.source_path not in paths:
            continue
        if reference.language == "python":
            target = _python_target(reference, paths)
        elif reference.language in {"javascript", "typescript"}:
            target = _javascript_target(reference, paths)
        else:
            continue
        if target is not None and target != reference.source_path:
            edges.add((reference.source_path, target))
    return sorted(edges)


def persist_import_relationships(
    session: Session,
    *,
    repository_id: UUID,
    files_by_path: Mapping[str, File],
    references: Sequence[ImportReference],
) -> None:
    if any(
        file.repository_id != repository_id or file.path != path
        for path, file in files_by_path.items()
    ):
        raise ValueError("Import files must belong to the repository and match paths")
    edges = resolve_imports(references, set(files_by_path))
    session.add_all(
        [
            Relationship(
                repository_id=repository_id,
                source_file_id=files_by_path[source].id,
                target_file_id=files_by_path[target].id,
                relationship_type="imports",
            )
            for source, target in edges
        ]
    )
    session.flush()
