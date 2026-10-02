import hashlib
import os
import shutil
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, defer

from app.code_parser import CodeParser, CodeUnitKind, ParsedCodeUnit
from app.code_relationships import (
    FileCallAnalysis,
    analyze_calls,
    persist_call_relationships,
)
from app.code_unit_embedding_persistence import (
    CodeUnitEmbedding,
    persist_code_unit_embeddings,
)
from app.code_unit_persistence import persist_code_units
from app.embedding_provider import EmbeddingProvider
from app.import_relationships import (
    ImportReference,
    extract_imports,
    persist_import_relationships,
)
from app.javascript_typescript_code_parser import (
    JavaScriptCodeParser,
    TypeScriptCodeParser,
)
from app.models.code_relationship import CodeRelationship
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.indexing_job import IndexingJob
from app.models.relationship import Relationship
from app.models.repository import Repository
from app.python_code_parser import PythonCodeParser
from app.repository_clone import ClonedRepository, clone_repository
from app.repository_file_persistence import _validate_relative_path
from app.repository_files import (
    RepositoryFileCandidate,
    discover_repository_files,
)
from app.repository_source import ValidatedRepositorySource, validate_repository_source
from app.text_code_parser import ConfigurationTextParser, DocumentationTextParser

EMBEDDING_BATCH_SIZE = 64

_JAVASCRIPT_SUFFIXES = frozenset({".js", ".jsx"})
_TYPESCRIPT_SUFFIXES = frozenset({".ts"})
_DOCUMENTATION_SUFFIXES = frozenset({".markdown", ".md", ".rst", ".txt"})
_DOCUMENTATION_NAMES = frozenset({"changelog", "license", "notice", "readme"})
_CONFIGURATION_SUFFIXES = frozenset({".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"})
_CONFIGURATION_NAMES = frozenset(
    {
        ".editorconfig",
        ".env.example",
        ".env.sample",
        ".env.template",
        ".gitignore",
        "dockerfile",
    }
)
_BLOCKING_JOB_STATUSES = ("pending", "running")


class RepositoryIndexingError(RuntimeError):
    """Raised when the indexing pipeline cannot safely complete."""


def index_repository(
    session: Session,
    *,
    repository_id: UUID,
    embedding_provider: EmbeddingProvider,
    workspace_root: Path,
) -> IndexingJob:
    # Serialize snapshot writers for this repository until the CALLER ends its
    # transaction. The service neither commits nor releases this row lock early.
    repository = session.scalar(
        select(Repository)
        .where(Repository.id == repository_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if repository is None:
        raise ValueError("Repository does not exist")
    if embedding_provider.dimension != EMBEDDING_DIMENSION:
        raise ValueError("Embedding provider dimension does not match storage")

    source = validate_repository_source(repository.source)
    _ensure_repository_can_be_indexed(session, repository_id)

    job = IndexingJob(repository_id=repository_id)
    session.add(job)
    session.flush()
    job.status = "running"
    session.flush()

    return _execute_indexing_job(
        session,
        repository=repository,
        job=job,
        source=source,
        embedding_provider=embedding_provider,
        workspace_root=workspace_root,
    )


def execute_reserved_indexing_job(
    session: Session,
    *,
    repository_id: UUID,
    job_id: UUID,
    embedding_provider: EmbeddingProvider,
    workspace_root: Path,
) -> IndexingJob:
    """Execute an already-claimed job; the runner owns the outer transaction."""
    repository = session.scalar(
        select(Repository)
        .where(Repository.id == repository_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    job = session.scalar(
        select(IndexingJob)
        .where(IndexingJob.id == job_id, IndexingJob.repository_id == repository_id)
        .execution_options(populate_existing=True)
    )
    if repository is None or job is None or job.status != "running":
        raise ValueError("Reserved indexing job is not running for this repository")
    if embedding_provider.dimension != EMBEDDING_DIMENSION:
        raise ValueError("Embedding provider dimension does not match storage")
    return _execute_indexing_job(
        session,
        repository=repository,
        job=job,
        source=validate_repository_source(repository.source),
        embedding_provider=embedding_provider,
        workspace_root=workspace_root,
    )


def _execute_indexing_job(
    session: Session,
    *,
    repository: Repository,
    job: IndexingJob,
    source: ValidatedRepositorySource,
    embedding_provider: EmbeddingProvider,
    workspace_root: Path,
) -> IndexingJob:

    try:
        with session.begin_nested():
            cloned_repository = clone_repository(source, workspace_root)
            try:
                _index_cloned_repository(
                    session=session,
                    repository_id=repository.id,
                    cloned_repository=cloned_repository,
                    embedding_provider=embedding_provider,
                )
            except BaseException as indexing_error:
                try:
                    _remove_owned_clone(cloned_repository.path, workspace_root)
                except Exception:
                    indexing_error.add_note("Cloned repository cleanup also failed")
                raise
            else:
                _remove_owned_clone(cloned_repository.path, workspace_root)
    except Exception:
        job.status = "failed"
        session.flush()
        raise

    job.status = "completed"
    session.flush()
    return job


def _ensure_repository_can_be_indexed(
    session: Session,
    repository_id: UUID,
) -> None:
    blocking_status = session.scalar(
        select(IndexingJob.status)
        .where(
            IndexingJob.repository_id == repository_id,
            IndexingJob.status.in_(_BLOCKING_JOB_STATUSES),
        )
        .limit(1)
    )
    if blocking_status in {"pending", "running"}:
        raise ValueError("Repository already has an active indexing job")


def _index_cloned_repository(
    *,
    session: Session,
    repository_id: UUID,
    cloned_repository: ClonedRepository,
    embedding_provider: EmbeddingProvider,
) -> None:
    candidates = discover_repository_files(cloned_repository.path, strict_errors=True)
    hashes: dict[str, str] = {}
    for candidate in candidates:
        _validate_relative_path(candidate.relative_path)
        hashes[candidate.relative_path] = hashlib.sha256(
            _read_candidate_bytes(candidate)
        ).hexdigest()
    files_by_path = {
        file.path: file
        for file in session.scalars(
            select(File)
            .where(File.repository_id == repository_id)
            .execution_options(populate_existing=True)
        )
    }
    refresh, removed = _classify_snapshot(
        hashes, {path: file.content_hash for path, file in files_by_path.items()}
    )
    if not refresh and not removed:
        return

    # Reconcile the complete graph, including edges originating in unchanged files.
    # All mutations belong to the public caller's indexing savepoint.
    session.execute(
        delete(CodeRelationship).where(CodeRelationship.repository_id == repository_id)
    )
    session.execute(
        delete(Relationship).where(Relationship.repository_id == repository_id)
    )
    removed_ids = [files_by_path[path].id for path in sorted(removed)]
    changed_ids = [
        files_by_path[path].id for path in sorted(refresh) if path in files_by_path
    ]
    if removed_ids or changed_ids:
        session.execute(
            delete(CodeUnit).where(CodeUnit.file_id.in_(removed_ids + changed_ids))
        )
    if removed_ids:
        session.execute(
            delete(File).where(
                File.repository_id == repository_id, File.id.in_(removed_ids)
            )
        )
        for path in removed:
            del files_by_path[path]
    for path in sorted(refresh):
        if path not in files_by_path:
            file = File(repository_id=repository_id, path=path)
            session.add(file)
            files_by_path[path] = file
    session.flush()

    unchanged_units: dict[UUID, list[CodeUnit]] = {}
    for unit in session.scalars(
        select(CodeUnit)
        .join(File, CodeUnit.file_id == File.id)
        .where(File.repository_id == repository_id)
        .options(defer(CodeUnit.embedding, raiseload=True))
        .order_by(
            CodeUnit.file_id,
            CodeUnit.start_line,
            CodeUnit.end_line.desc(),
            CodeUnit.kind,
            CodeUnit.id,
        )
    ):
        unchanged_units.setdefault(unit.file_id, []).append(unit)
    parsers: dict[str, CodeParser] = {
        "configuration": ConfigurationTextParser(),
        "documentation": DocumentationTextParser(),
        "javascript": JavaScriptCodeParser(),
        "python": PythonCodeParser(),
        "typescript": TypeScriptCodeParser(),
    }
    embedding_batch: list[CodeUnit] = []
    imports: list[ImportReference] = []
    call_analyses: list[FileCallAnalysis] = []

    for candidate in candidates:
        path = candidate.relative_path
        file = files_by_path[path]
        parser = _parser_for_path(path, parsers)
        if parser is None:
            file.content_hash = hashes[path]
            continue

        content_bytes = _read_candidate_bytes(candidate)
        if hashlib.sha256(content_bytes).hexdigest() != hashes[path]:
            raise RepositoryIndexingError("Repository file changed during indexing")
        content = _decode_utf8(content_bytes)
        if path in refresh:
            parsed_units = parser.parse(content=content, relative_path=path)
            stored_units = persist_code_units(session, file.id, parsed_units)
            embedding_batch.extend(stored_units)
        else:
            stored_units = unchanged_units.get(file.id, [])
            # Import extraction consumes ParsedCodeUnits, but reused units are
            # reconstructed from persisted metadata, not reparsed or reinserted.
            parsed_units = [
                ParsedCodeUnit(
                    kind=CodeUnitKind(unit.kind),
                    content=unit.content,
                    relative_path=path,
                    language=unit.language,
                    start_line=unit.start_line,
                    end_line=unit.end_line,
                    symbol_name=unit.symbol_name,
                )
                for unit in stored_units
                if unit.kind == CodeUnitKind.IMPORT.value
            ]
        file.content_hash = hashes[path]
        imports.extend(extract_imports(parsed_units))
        if stored_units:
            call_analyses.append(
                analyze_calls(
                    content=content,
                    file=file,
                    language=stored_units[0].language,
                    units=stored_units,
                )
            )

        while len(embedding_batch) >= EMBEDDING_BATCH_SIZE:
            batch = embedding_batch[:EMBEDDING_BATCH_SIZE]
            del embedding_batch[:EMBEDDING_BATCH_SIZE]
            _embed_and_persist(session, embedding_provider, batch)

    if embedding_batch:
        _embed_and_persist(session, embedding_provider, embedding_batch)
    persist_import_relationships(
        session,
        repository_id=repository_id,
        files_by_path=files_by_path,
        references=imports,
    )
    persist_call_relationships(
        session,
        repository_id=repository_id,
        files_by_path=files_by_path,
        analyses=call_analyses,
    )
    # The public lifecycle holds the repository lock and snapshot savepoint.
    # No-change runs returned above; cleanup/outer rollback also undo this bump.
    session.execute(
        update(Repository)
        .where(Repository.id == repository_id)
        .values(index_generation=Repository.index_generation + 1)
    )


def _classify_snapshot(
    hashes: dict[str, str], persisted: dict[str, str | None]
) -> tuple[set[str], set[str]]:
    """Return new/changed/unverified paths and removed paths, without rename guesses."""
    return (
        {path for path, digest in hashes.items() if persisted.get(path) != digest},
        set(persisted) - set(hashes),
    )


def _parser_for_path(
    relative_path: str,
    parsers: dict[str, CodeParser],
) -> CodeParser | None:
    path = PurePosixPath(relative_path)
    suffix = path.suffix.casefold()
    name = path.name.casefold()

    if suffix == ".py":
        return parsers["python"]
    if suffix in _JAVASCRIPT_SUFFIXES:
        return parsers["javascript"]
    if suffix in _TYPESCRIPT_SUFFIXES:
        return parsers["typescript"]
    if suffix in _DOCUMENTATION_SUFFIXES or (
        not suffix and name in _DOCUMENTATION_NAMES
    ):
        return parsers["documentation"]
    if suffix in _CONFIGURATION_SUFFIXES or name in _CONFIGURATION_NAMES:
        return parsers["configuration"]
    return None


def _read_candidate_bytes(candidate: RepositoryFileCandidate) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)

    try:
        descriptor = os.open(candidate.path, flags)
        with os.fdopen(descriptor, "rb") as file:
            metadata = os.fstat(file.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise RepositoryIndexingError("Repository file is no longer regular")
            if metadata.st_size != candidate.size_bytes:
                raise RepositoryIndexingError("Repository file changed during indexing")
            content_bytes = file.read(candidate.size_bytes + 1)
            if len(content_bytes) != candidate.size_bytes:
                raise RepositoryIndexingError("Repository file changed during indexing")
            after = os.fstat(file.fileno())
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            ):
                raise RepositoryIndexingError("Repository file changed during indexing")
    except RepositoryIndexingError:
        raise
    except OSError as error:
        raise RepositoryIndexingError("Repository file could not be read") from error

    return content_bytes


def _decode_utf8(content_bytes: bytes) -> str:
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RepositoryIndexingError("Repository file is not valid UTF-8") from error


def _embed_and_persist(
    session: Session,
    embedding_provider: EmbeddingProvider,
    units: Sequence[CodeUnit],
) -> None:
    vectors = embedding_provider.embed([unit.content for unit in units])
    if len(vectors) != len(units):
        raise RepositoryIndexingError(
            "Embedding provider returned an unexpected vector count"
        )

    requests = [
        CodeUnitEmbedding(code_unit_id=unit.id, vector=vector)
        for unit, vector in zip(units, vectors, strict=True)
    ]
    persist_code_unit_embeddings(session, requests)


def _remove_owned_clone(clone_path: Path, workspace_root: Path) -> None:
    try:
        workspace = workspace_root.resolve(strict=True)
        destination = clone_path.resolve(strict=True)
        if destination == workspace or destination.parent != workspace:
            raise RepositoryIndexingError("Clone destination is not owned by workspace")
        shutil.rmtree(destination)
    except RepositoryIndexingError:
        raise
    except OSError as error:
        raise RepositoryIndexingError("Cloned repository cleanup failed") from error
