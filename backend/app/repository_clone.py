import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from app.repository_source import RepositorySourceKind, ValidatedRepositorySource

DEFAULT_CLONE_TIMEOUT_SECONDS = 120.0
GITHUB_HOSTNAME = "github.com"


class RepositoryCloneError(RuntimeError):
    """Raised when a repository cannot be cloned into the controlled workspace."""


@dataclass(frozen=True, slots=True)
class ClonedRepository:
    source: str
    path: Path


def clone_repository(
    source: ValidatedRepositorySource,
    workspace_root: Path,
    *,
    timeout_seconds: float = DEFAULT_CLONE_TIMEOUT_SECONDS,
) -> ClonedRepository:
    _require_github_https_source(source)
    destination = _prepare_clone_destination(workspace_root)

    command = [
        "git",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "credential.helper=",
        "-c",
        f"core.attributesFile={os.devnull}",
        "-c",
        "http.followRedirects=false",
        "clone",
        "--quiet",
        "--no-recurse-submodules",
        "--",
        source.value,
        str(destination),
    ]

    try:
        subprocess.run(
            command,
            shell=False,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_git_environment(),
        )
    except FileNotFoundError as error:
        _cleanup_partial_clone(destination)
        raise RepositoryCloneError("Git executable is not available") from error
    except subprocess.TimeoutExpired as error:
        _cleanup_partial_clone(destination)
        raise RepositoryCloneError("Repository clone timed out") from error
    except subprocess.CalledProcessError as error:
        _cleanup_partial_clone(destination)
        raise RepositoryCloneError("Git could not clone the repository") from error
    except OSError as error:
        _cleanup_partial_clone(destination)
        raise RepositoryCloneError("Git could not be started") from error

    return ClonedRepository(source=source.value, path=destination)


def _require_github_https_source(source: ValidatedRepositorySource) -> None:
    if source.kind is not RepositorySourceKind.HTTPS:
        raise RepositoryCloneError("Only validated HTTPS sources can be cloned")

    parsed = urlsplit(source.value)
    if parsed.scheme.lower() != "https" or parsed.hostname != GITHUB_HOSTNAME:
        raise RepositoryCloneError(
            "Only public GitHub repository sources can be cloned"
        )


def _prepare_clone_destination(workspace_root: Path) -> Path:
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = workspace_root.resolve(strict=True)
        if not workspace.is_dir():
            raise NotADirectoryError
        destination = Path(tempfile.mkdtemp(prefix="repository-", dir=workspace))
    except (OSError, RuntimeError) as error:
        raise RepositoryCloneError("Clone workspace could not be prepared") from error

    return destination


def _git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.pop("SSH_ASKPASS", None)
    environment.pop("SSH_ASKPASS_REQUIRE", None)
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _cleanup_partial_clone(destination: Path) -> None:
    try:
        shutil.rmtree(destination)
    except FileNotFoundError:
        return
    except OSError as error:
        raise RepositoryCloneError("Partial repository clone cleanup failed") from error
