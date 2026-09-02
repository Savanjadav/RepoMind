import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_MAX_FILE_SIZE_BYTES = 1_048_576
BINARY_SNIFF_BYTES = 8_192

_DENIED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".gradle",
        ".idea",
        ".mypy_cache",
        ".next",
        ".nuxt",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".vscode",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "out",
        "target",
        "vendor",
        "venv",
    }
)

_DENIED_FILE_NAMES = frozenset(
    {
        ".ds_store",
        "bun.lock",
        "bun.lockb",
        "cargo.lock",
        "composer.lock",
        "gemfile.lock",
        "go.sum",
        "package-lock.json",
        "pipfile.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "yarn.lock",
    }
)

_SECRET_FILE_NAMES = frozenset(
    {
        ".env",
        ".envrc",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "_netrc",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "service-account.json",
        "service_account.json",
    }
)

_SAFE_ENV_TEMPLATE_NAMES = frozenset(
    {
        ".env.example",
        ".env.sample",
        ".env.template",
    }
)

_DENIED_EXTENSIONS = frozenset(
    {
        ".7z",
        ".a",
        ".bin",
        ".bmp",
        ".bz2",
        ".class",
        ".db",
        ".dll",
        ".doc",
        ".docx",
        ".dylib",
        ".exe",
        ".flac",
        ".gif",
        ".gz",
        ".ico",
        ".jar",
        ".jpeg",
        ".jpg",
        ".key",
        ".m4a",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".o",
        ".obj",
        ".ogg",
        ".otf",
        ".p12",
        ".pfx",
        ".pdf",
        ".pem",
        ".png",
        ".ppt",
        ".pptx",
        ".pyc",
        ".pyo",
        ".rar",
        ".so",
        ".sqlite",
        ".sqlite3",
        ".svg",
        ".tar",
        ".tgz",
        ".tif",
        ".tiff",
        ".ttf",
        ".war",
        ".wasm",
        ".wav",
        ".webm",
        ".webp",
        ".whl",
        ".woff",
        ".woff2",
        ".xls",
        ".xlsx",
        ".xz",
        ".zip",
    }
)


class RepositoryFileDiscoveryError(RuntimeError):
    """Raised when a repository root cannot be safely scanned."""


@dataclass(frozen=True, slots=True)
class RepositoryFileCandidate:
    path: Path
    relative_path: str
    size_bytes: int


def discover_repository_files(
    repository_root: Path,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> list[RepositoryFileCandidate]:
    if max_file_size_bytes <= 0:
        raise RepositoryFileDiscoveryError("Maximum file size must be positive")

    root = Path(os.path.abspath(repository_root))
    root_entries = _open_root(root)
    candidates: list[RepositoryFileCandidate] = []
    directories: list[tuple[Path, tuple[str, ...], list[os.DirEntry[str]]]] = [
        (root, (), root_entries)
    ]

    while directories:
        directory, components, entries = directories.pop()

        if components and any(entry.name.casefold() == ".git" for entry in entries):
            continue

        for entry in entries:
            if _has_unsafe_control_character(entry.name):
                continue

            name = entry.name.casefold()
            child_components = (*components, entry.name)
            child_path = directory / entry.name

            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                continue

            if stat.S_ISLNK(metadata.st_mode):
                continue

            if stat.S_ISDIR(metadata.st_mode):
                if name in _DENIED_DIRECTORY_NAMES:
                    continue
                nested_entries = _open_nested_directory(child_path)
                if nested_entries is not None:
                    directories.append((child_path, child_components, nested_entries))
                continue

            if not stat.S_ISREG(metadata.st_mode):
                continue
            if metadata.st_size > max_file_size_bytes:
                continue
            if _is_denied_file_name(entry.name):
                continue

            inspected_size = _inspect_regular_file(
                child_path,
                metadata,
                max_file_size_bytes=max_file_size_bytes,
            )
            if inspected_size is None:
                continue

            relative_path = PurePosixPath(*child_components).as_posix()
            candidates.append(
                RepositoryFileCandidate(
                    path=child_path,
                    relative_path=relative_path,
                    size_bytes=inspected_size,
                )
            )

    return sorted(candidates, key=lambda candidate: candidate.relative_path)


def _open_root(root: Path) -> list[os.DirEntry[str]]:
    try:
        metadata = root.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RepositoryFileDiscoveryError("Repository root cannot be scanned")
        if not stat.S_ISDIR(metadata.st_mode):
            raise RepositoryFileDiscoveryError("Repository root cannot be scanned")
        with os.scandir(root) as entries:
            return list(entries)
    except RepositoryFileDiscoveryError:
        raise
    except OSError as error:
        raise RepositoryFileDiscoveryError(
            "Repository root cannot be scanned"
        ) from error


def _open_nested_directory(path: Path) -> list[os.DirEntry[str]] | None:
    try:
        with os.scandir(path) as entries:
            return list(entries)
    except OSError:
        return None


def _has_unsafe_control_character(name: str) -> bool:
    return any(ord(character) <= 0x1F or ord(character) == 0x7F for character in name)


def _is_denied_file_name(name: str) -> bool:
    normalized = name.casefold()

    if normalized in _SAFE_ENV_TEMPLATE_NAMES:
        return False
    if normalized in _SECRET_FILE_NAMES or normalized.startswith(".env."):
        return True
    if normalized in _DENIED_FILE_NAMES:
        return True
    if normalized.endswith((".min.js", ".min.css", ".map", ".log")):
        return True
    return Path(normalized).suffix in _DENIED_EXTENSIONS


def _inspect_regular_file(
    path: Path,
    initial_metadata: os.stat_result,
    *,
    max_file_size_bytes: int,
) -> int | None:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)

    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None

    inspected_size: int | None = None
    try:
        metadata = os.fstat(descriptor)
        is_same_regular_file = stat.S_ISREG(metadata.st_mode) and not (
            metadata.st_dev != initial_metadata.st_dev
            or metadata.st_ino != initial_metadata.st_ino
        )
        if (
            is_same_regular_file
            and metadata.st_size <= max_file_size_bytes
            and b"\x00" not in os.read(descriptor, BINARY_SNIFF_BYTES)
        ):
            inspected_size = metadata.st_size
    except OSError:
        return None
    finally:
        try:
            os.close(descriptor)
        except OSError:
            inspected_size = None

    return inspected_size
