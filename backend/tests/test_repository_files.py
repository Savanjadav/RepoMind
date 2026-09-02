import os
from pathlib import Path
from typing import NoReturn

import pytest

from app import repository_files
from app.repository_files import (
    RepositoryFileDiscoveryError,
    discover_repository_files,
)


def _relative_paths(root: Path, **kwargs: int) -> list[str]:
    return [
        candidate.relative_path
        for candidate in discover_repository_files(root, **kwargs)
    ]


def test_discovers_source_documentation_and_configuration_files(
    tmp_path: Path,
) -> None:
    files = {
        "src/main.py": "print('safe')\n",
        "README.md": "# Example\n",
        "pyproject.toml": "[project]\n",
        "package.json": "{}\n",
        "Dockerfile": "FROM scratch\n",
        "compose.yaml": "services: {}\n",
        ".github/workflows/check.yml": "name: Check\n",
        ".gitignore": "build/\n",
        ".editorconfig": "root = true\n",
    }
    for relative_path, contents in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    assert _relative_paths(tmp_path) == sorted(files)


def test_root_git_directory_is_pruned_without_excluding_root_files(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("unsafe", encoding="utf-8")
    (tmp_path / "safe.py").write_text("safe = True\n", encoding="utf-8")

    assert _relative_paths(tmp_path) == ["safe.py"]


@pytest.mark.parametrize("git_marker_kind", ["directory", "file"])
def test_nested_git_marker_excludes_entire_nested_repository(
    git_marker_kind: str,
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "excluded.py").write_text("unsafe = True\n", encoding="utf-8")
    marker = nested / ".git"
    if git_marker_kind == "directory":
        marker.mkdir()
    else:
        marker.write_text("gitdir: elsewhere\n", encoding="utf-8")
    (tmp_path / "safe.py").write_text("safe = True\n", encoding="utf-8")

    assert _relative_paths(tmp_path) == ["safe.py"]


def test_nested_git_symlink_excludes_entire_nested_repository(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "excluded.py").write_text("unsafe = True\n", encoding="utf-8")
    try:
        (nested / ".git").symlink_to(tmp_path / "outside")
    except (NotImplementedError, OSError):
        pytest.skip("Symlink creation is not supported")

    assert _relative_paths(tmp_path) == []


@pytest.mark.parametrize(
    "directory_name",
    [
        "node_modules",
        "build",
        "__pycache__",
        ".pytest_cache",
        ".next",
        ".vscode",
    ],
)
def test_denied_directories_are_not_traversed(
    directory_name: str,
    tmp_path: Path,
) -> None:
    denied = tmp_path / directory_name
    denied.mkdir()
    (denied / "excluded.py").write_text("unsafe = True\n", encoding="utf-8")
    (tmp_path / "safe.py").write_text("safe = True\n", encoding="utf-8")

    assert _relative_paths(tmp_path) == ["safe.py"]


def test_symlink_file_and_directory_outside_root_are_excluded(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("outside = True\n", encoding="utf-8")
    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    (outside_directory / "outside.py").write_text("outside = True\n", encoding="utf-8")
    try:
        (repository / "file-link.py").symlink_to(outside_file)
        (repository / "directory-link").symlink_to(
            outside_directory,
            target_is_directory=True,
        )
    except (NotImplementedError, OSError):
        pytest.skip("Symlink creation is not supported")
    (repository / "safe.py").write_text("safe = True\n", encoding="utf-8")

    candidates = discover_repository_files(repository)

    assert [candidate.relative_path for candidate in candidates] == ["safe.py"]
    assert all(
        candidate.path.is_relative_to(repository.absolute()) for candidate in candidates
    )


def test_size_limit_and_reported_size(tmp_path: Path) -> None:
    (tmp_path / "small.txt").write_bytes(b"1234")
    (tmp_path / "at-limit.txt").write_bytes(b"12345")
    (tmp_path / "too-large.txt").write_bytes(b"123456")

    candidates = discover_repository_files(tmp_path, max_file_size_bytes=5)

    assert [
        (candidate.relative_path, candidate.size_bytes) for candidate in candidates
    ] == [
        ("at-limit.txt", 5),
        ("small.txt", 4),
    ]


def test_binary_prefix_is_excluded_and_text_is_included(tmp_path: Path) -> None:
    (tmp_path / "binary.data").write_bytes(b"header\x00binary")
    (tmp_path / "unicode.txt").write_text("Hello, 世界\n", encoding="utf-8")

    assert _relative_paths(tmp_path) == ["unicode.txt"]


@pytest.mark.parametrize(
    "secret_name",
    [
        ".env",
        ".env.local",
        "ID_RSA",
        "credentials.json",
        "service_account.json",
        ".netrc",
        ".npmrc",
        "certificate.pem",
        "private.KEY",
        "certificate.p12",
        "certificate.pfx",
    ],
)
def test_secret_like_files_are_excluded(secret_name: str, tmp_path: Path) -> None:
    (tmp_path / secret_name).write_text("secret", encoding="utf-8")

    assert _relative_paths(tmp_path) == []


def test_safe_environment_templates_are_included(tmp_path: Path) -> None:
    expected = [".env.example", ".env.sample", ".env.template"]
    for name in expected:
        (tmp_path / name).write_text("SETTING=example\n", encoding="utf-8")

    assert _relative_paths(tmp_path) == expected


@pytest.mark.parametrize(
    "artifact_name",
    [
        ".DS_Store",
        "package-lock.json",
        "yarn.lock",
        "bundle.min.js",
        "styles.MIN.CSS",
        "app.js.map",
        "debug.log",
        "archive.zip",
        "photo.PNG",
        "program.exe",
        "manual.pdf",
    ],
)
def test_noisy_and_binary_artifacts_are_excluded(
    artifact_name: str,
    tmp_path: Path,
) -> None:
    (tmp_path / artifact_name).write_bytes(b"artifact")

    assert _relative_paths(tmp_path) == []


def test_control_character_filename_is_skipped(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe\nname.py"
    try:
        unsafe.write_text("unsafe = True\n", encoding="utf-8")
    except OSError:
        pytest.skip("Control characters in filenames are not supported")
    (tmp_path / "safe.py").write_text("safe = True\n", encoding="utf-8")

    assert _relative_paths(tmp_path) == ["safe.py"]


def test_unicode_filename_is_preserved(tmp_path: Path) -> None:
    (tmp_path / "résumé.py").write_text("safe = True\n", encoding="utf-8")

    candidates = discover_repository_files(tmp_path)

    assert candidates[0].relative_path == "résumé.py"
    assert candidates[0].path.is_absolute()
    assert candidates[0].path == tmp_path / "résumé.py"


def test_results_are_sorted_and_use_posix_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "z.py").write_text("z = 1\n", encoding="utf-8")
    nested = tmp_path / "a"
    nested.mkdir()
    (nested / "b.py").write_text("b = 1\n", encoding="utf-8")

    candidates = discover_repository_files(tmp_path)

    assert [candidate.relative_path for candidate in candidates] == ["a/b.py", "z.py"]
    assert all("\\" not in candidate.relative_path for candidate in candidates)
    assert all(candidate.path.is_absolute() for candidate in candidates)


def test_executable_text_file_is_included_without_execution(tmp_path: Path) -> None:
    executable = tmp_path / "script.sh"
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    executable.chmod(0o755)

    assert _relative_paths(tmp_path) == ["script.sh"]


def test_fifo_is_excluded_when_supported(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are not supported")
    try:
        os.mkfifo(tmp_path / "pipe")
    except OSError:
        pytest.skip("FIFO creation is not supported")
    (tmp_path / "safe.py").write_text("safe = True\n", encoding="utf-8")

    assert _relative_paths(tmp_path) == ["safe.py"]


@pytest.mark.parametrize("root_kind", ["missing", "file", "symlink"])
def test_invalid_repository_root_raises_sanitized_error(
    root_kind: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    if root_kind == "file":
        root.write_text("not a directory", encoding="utf-8")
    elif root_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        try:
            root.symlink_to(target, target_is_directory=True)
        except (NotImplementedError, OSError):
            pytest.skip("Symlink creation is not supported")

    with pytest.raises(
        RepositoryFileDiscoveryError,
        match="Repository root cannot be scanned",
    ):
        discover_repository_files(root)


@pytest.mark.parametrize("size_limit", [0, -1])
def test_non_positive_size_limit_is_rejected(
    size_limit: int,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RepositoryFileDiscoveryError,
        match="Maximum file size must be positive",
    ):
        discover_repository_files(tmp_path, max_file_size_bytes=size_limit)


def test_unreadable_entry_is_skipped_while_safe_sibling_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "blocked.py"
    blocked.write_text("blocked = True\n", encoding="utf-8")
    (tmp_path / "safe.py").write_text("safe = True\n", encoding="utf-8")
    real_open = repository_files.os.open

    def selective_open(path: os.PathLike[str], flags: int) -> int:
        if Path(path).name == "blocked.py":
            raise PermissionError("untrusted entry cannot be opened")
        return real_open(path, flags)

    monkeypatch.setattr(repository_files.os, "open", selective_open)

    assert _relative_paths(tmp_path) == ["safe.py"]


def test_root_initial_scan_failure_is_fatal_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_scan(path: os.PathLike[str]) -> NoReturn:
        raise PermissionError("sensitive root details")

    monkeypatch.setattr(repository_files.os, "scandir", failed_scan)

    with pytest.raises(RepositoryFileDiscoveryError) as captured_error:
        discover_repository_files(tmp_path)

    assert str(captured_error.value) == "Repository root cannot be scanned"
    assert "sensitive" not in str(captured_error.value)
