from collections.abc import Sequence
from pathlib import PurePosixPath
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.file import File
from app.repository_files import RepositoryFileCandidate


def persist_repository_files(
    session: Session,
    repository_id: UUID,
    candidates: Sequence[RepositoryFileCandidate],
) -> list[File]:
    candidate_paths = _validated_candidate_paths(candidates)
    if not candidate_paths:
        return []

    existing_paths = set(
        session.scalars(
            select(File.path).where(
                File.repository_id == repository_id,
                File.path.in_(candidate_paths),
            )
        )
    )
    new_files = [
        File(repository_id=repository_id, path=path)
        for path in candidate_paths
        if path not in existing_paths
    ]
    if not new_files:
        return []

    session.add_all(new_files)
    session.flush()
    return new_files


def _validated_candidate_paths(
    candidates: Sequence[RepositoryFileCandidate],
) -> list[str]:
    paths: set[str] = set()
    for candidate in candidates:
        path = candidate.relative_path
        _validate_relative_path(path)
        paths.add(path)
    return sorted(paths)


def _validate_relative_path(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise ValueError("Repository file path must be a non-empty string")
    if "\\" in path:
        raise ValueError("Repository file path must use POSIX separators")
    if any(ord(character) <= 0x1F or ord(character) == 0x7F for character in path):
        raise ValueError("Repository file path contains a control character")

    components = path.split("/")
    if PurePosixPath(path).is_absolute() or any(
        component in {"", ".", ".."} for component in components
    ):
        raise ValueError("Repository file path must be a safe relative POSIX path")
