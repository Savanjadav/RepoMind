import os
import shutil
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.code_parser import CodeParser
from app.code_unit_embedding_persistence import (
    CodeUnitEmbedding,
    persist_code_unit_embeddings,
)
from app.code_unit_persistence import persist_code_units
from app.embedding_provider import EmbeddingProvider
from app.javascript_typescript_code_parser import (
    JavaScriptCodeParser,
    TypeScriptCodeParser,
)
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.indexing_job import IndexingJob
from app.models.repository import Repository
from app.python_code_parser import PythonCodeParser
from app.repository_clone import ClonedRepository, clone_repository
from app.repository_file_persistence import persist_repository_files
from app.repository_files import (
    RepositoryFileCandidate,
    discover_repository_files,
)
from app.repository_source import validate_repository_source
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
_BLOCKING_JOB_STATUSES = ("completed", "pending", "running")


class RepositoryIndexingError(RuntimeError):
    """Raised when the indexing pipeline cannot safely complete."""


def index_repository(
    session: Session,
    *,
    repository_id: UUID,
    embedding_provider: EmbeddingProvider,
    workspace_root: Path,
) -> IndexingJob:
    repository = session.get(Repository, repository_id)
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

    try:
        with session.begin_nested():
            cloned_repository = clone_repository(source, workspace_root)
            try:
                _index_cloned_repository(
                    session=session,
                    repository_id=repository_id,
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
    if blocking_status == "completed":
        raise ValueError("Repository has already been indexed")

    existing_unit_id = session.scalar(
        select(CodeUnit.id)
        .join(File, CodeUnit.file_id == File.id)
        .where(File.repository_id == repository_id)
        .limit(1)
    )
    if existing_unit_id is not None:
        raise ValueError("Repository already has persisted CodeUnits")


def _index_cloned_repository(
    *,
    session: Session,
    repository_id: UUID,
    cloned_repository: ClonedRepository,
    embedding_provider: EmbeddingProvider,
) -> None:
    candidates = discover_repository_files(cloned_repository.path)
    files_by_path = _persisted_files_by_path(session, repository_id, candidates)
    parsers: dict[str, CodeParser] = {
        "configuration": ConfigurationTextParser(),
        "documentation": DocumentationTextParser(),
        "javascript": JavaScriptCodeParser(),
        "python": PythonCodeParser(),
        "typescript": TypeScriptCodeParser(),
    }
    embedding_batch: list[CodeUnit] = []

    for candidate in candidates:
        parser = _parser_for_path(candidate.relative_path, parsers)
        if parser is None:
            continue

        content = _read_utf8_candidate(candidate)
        parsed_units = parser.parse(
            content=content,
            relative_path=candidate.relative_path,
        )
        if not parsed_units:
            continue

        file = files_by_path.get(candidate.relative_path)
        if file is None:
            raise RepositoryIndexingError("Persisted repository file is missing")
        embedding_batch.extend(persist_code_units(session, file.id, parsed_units))

        while len(embedding_batch) >= EMBEDDING_BATCH_SIZE:
            batch = embedding_batch[:EMBEDDING_BATCH_SIZE]
            del embedding_batch[:EMBEDDING_BATCH_SIZE]
            _embed_and_persist(session, embedding_provider, batch)

    if embedding_batch:
        _embed_and_persist(session, embedding_provider, embedding_batch)


def _persisted_files_by_path(
    session: Session,
    repository_id: UUID,
    candidates: Sequence[RepositoryFileCandidate],
) -> dict[str, File]:
    inserted_files = persist_repository_files(session, repository_id, candidates)
    files_by_path = {file.path: file for file in inserted_files}
    missing_paths = [
        candidate.relative_path
        for candidate in candidates
        if candidate.relative_path not in files_by_path
    ]
    if missing_paths:
        existing_files = session.scalars(
            select(File).where(
                File.repository_id == repository_id,
                File.path.in_(missing_paths),
            )
        ).all()
        files_by_path.update({file.path: file for file in existing_files})

    if len(files_by_path) != len(candidates):
        raise RepositoryIndexingError("Repository file metadata could not be persisted")
    return files_by_path


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


def _read_utf8_candidate(candidate: RepositoryFileCandidate) -> str:
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
    except RepositoryIndexingError:
        raise
    except OSError as error:
        raise RepositoryIndexingError("Repository file could not be read") from error

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
