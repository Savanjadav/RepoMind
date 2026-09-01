import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from app import repository_clone
from app.repository_clone import RepositoryCloneError, clone_repository
from app.repository_source import validate_repository_source

GITHUB_SOURCE = "https://github.com/example/project.git"


def test_successful_clone_uses_controlled_git_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def successful_run(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(repository_clone.subprocess, "run", successful_run)
    monkeypatch.setenv("GIT_DIR", "/untrusted/git-dir")
    monkeypatch.setenv("GIT_TRACE", "1")
    monkeypatch.setenv("SSH_ASKPASS", "/untrusted/askpass")

    workspace = tmp_path / "workspace"
    sibling = workspace / "keep.txt"
    workspace.mkdir()
    sibling.write_text("keep", encoding="utf-8")
    source = validate_repository_source(GITHUB_SOURCE)

    result = clone_repository(source, workspace, timeout_seconds=7.5)

    assert result.source == GITHUB_SOURCE
    assert result.path.is_absolute()
    assert result.path.parent == workspace.resolve()
    assert sibling.read_text(encoding="utf-8") == "keep"
    assert len(calls) == 1

    command, options = calls[0]
    separator_index = command.index("--")

    assert command[separator_index + 1 :] == [GITHUB_SOURCE, str(result.path)]
    assert "--quiet" in command
    assert "--no-recurse-submodules" in command
    assert "--recurse-submodules" not in command
    assert "--depth" not in command
    assert not any(argument.startswith("--filter") for argument in command)
    assert f"core.hooksPath={os.devnull}" in command
    assert "credential.helper=" in command
    assert f"core.attributesFile={os.devnull}" in command
    assert "http.followRedirects=false" in command
    assert options["shell"] is False
    assert options["check"] is True
    assert options["capture_output"] is True
    assert options["text"] is True
    assert options["timeout"] == 7.5

    environment = options["env"]
    assert isinstance(environment, dict)
    assert "PATH" in environment
    assert "GIT_DIR" not in environment
    assert "GIT_TRACE" not in environment
    assert "SSH_ASKPASS" not in environment
    assert environment["GIT_ATTR_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_destinations_are_unique(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def successful_run(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(repository_clone.subprocess, "run", successful_run)
    source = validate_repository_source(GITHUB_SOURCE)
    workspace = tmp_path / "workspace"

    first = clone_repository(source, workspace)
    second = clone_repository(source, workspace)

    assert first.path != second.path
    assert first.path.parent == workspace.resolve()
    assert second.path.parent == workspace.resolve()


@pytest.mark.parametrize(
    "source_value",
    [
        "../local-repository",
        "https://gitlab.com/example/project.git",
    ],
)
def test_unsupported_sources_are_rejected_before_workspace_or_git(
    source_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(command: list[str], **kwargs: Any) -> None:
        pytest.fail(f"Git must not be called: {command!r}, {kwargs!r}")

    monkeypatch.setattr(repository_clone.subprocess, "run", unexpected_run)
    workspace = tmp_path / "workspace"
    source = validate_repository_source(source_value)

    with pytest.raises(RepositoryCloneError):
        clone_repository(source, workspace)

    assert not workspace.exists()


@pytest.mark.parametrize(
    ("git_error", "expected_message"),
    [
        (
            subprocess.CalledProcessError(
                128,
                ["git", "clone"],
                stderr="sensitive Git stderr",
            ),
            "Git could not clone the repository",
        ),
        (
            subprocess.TimeoutExpired(
                ["git", "clone"],
                120,
                stderr="sensitive timeout stderr",
            ),
            "Repository clone timed out",
        ),
        (
            FileNotFoundError("sensitive executable details"),
            "Git executable is not available",
        ),
    ],
)
def test_git_failures_are_sanitized_and_remove_partial_clone(
    git_error: Exception,
    expected_message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_run(command: list[str], **kwargs: Any) -> None:
        raise git_error

    monkeypatch.setattr(repository_clone.subprocess, "run", failed_run)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sibling = workspace / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")
    source = validate_repository_source(GITHUB_SOURCE)

    with pytest.raises(RepositoryCloneError) as captured_error:
        clone_repository(source, workspace)

    assert str(captured_error.value) == expected_message
    assert "sensitive" not in str(captured_error.value)
    assert sibling.read_text(encoding="utf-8") == "keep"
    assert list(workspace.iterdir()) == [sibling]


def test_workspace_preparation_failure_is_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(command: list[str], **kwargs: Any) -> None:
        pytest.fail(f"Git must not be called: {command!r}, {kwargs!r}")

    monkeypatch.setattr(repository_clone.subprocess, "run", unexpected_run)
    workspace_file = tmp_path / "workspace"
    workspace_file.write_text("not a directory", encoding="utf-8")
    source = validate_repository_source(GITHUB_SOURCE)

    with pytest.raises(
        RepositoryCloneError,
        match="Clone workspace could not be prepared",
    ):
        clone_repository(source, workspace_file)


def test_destination_creation_failure_is_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_destination_creation(*args: Any, **kwargs: Any) -> str:
        raise OSError("sensitive workspace details")

    def unexpected_run(command: list[str], **kwargs: Any) -> None:
        pytest.fail(f"Git must not be called: {command!r}, {kwargs!r}")

    monkeypatch.setattr(
        repository_clone.tempfile,
        "mkdtemp",
        failed_destination_creation,
    )
    monkeypatch.setattr(repository_clone.subprocess, "run", unexpected_run)
    source = validate_repository_source(GITHUB_SOURCE)

    with pytest.raises(RepositoryCloneError) as captured_error:
        clone_repository(source, tmp_path / "workspace")

    assert str(captured_error.value) == "Clone workspace could not be prepared"
    assert "sensitive" not in str(captured_error.value)


def test_cleanup_failure_is_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_run(command: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(128, command)

    def failed_cleanup(destination: Path) -> None:
        raise OSError("sensitive cleanup details")

    monkeypatch.setattr(repository_clone.subprocess, "run", failed_run)
    monkeypatch.setattr(repository_clone.shutil, "rmtree", failed_cleanup)
    source = validate_repository_source(GITHUB_SOURCE)

    with pytest.raises(RepositoryCloneError) as captured_error:
        clone_repository(source, tmp_path / "workspace")

    assert str(captured_error.value) == "Partial repository clone cleanup failed"
    assert "sensitive" not in str(captured_error.value)
