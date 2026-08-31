import pytest

from app.repository_source import (
    RepositorySourceKind,
    RepositorySourceValidationError,
    validate_repository_source,
)


@pytest.mark.parametrize(
    "source",
    [
        "https://github.com/example/project.git",
        "https://github.com/example/project",
        "https://git.example.com/groups/team/project.git",
        "https://git.example.com:8443/team/project.git",
    ],
)
def test_valid_https_sources_are_preserved(source: str) -> None:
    result = validate_repository_source(source)

    assert result.kind is RepositorySourceKind.HTTPS
    assert result.value == source


@pytest.mark.parametrize(
    "source",
    [
        "/Users/example/projects/RepoMind",
        "RepoMind",
        "./RepoMind",
        "../RepoMind",
        "~/projects/RepoMind",
        "/Users/example/My Projects/RepoMind",
        "C:\\projects\\RepoMind",
        "D:/code/RepoMind",
        "C:projects\\RepoMind",
        "D:code/RepoMind",
    ],
)
def test_valid_local_sources_are_preserved(source: str) -> None:
    result = validate_repository_source(source)

    assert result.kind is RepositorySourceKind.LOCAL
    assert result.value == source


@pytest.mark.parametrize(
    "source",
    [
        "",
        "   ",
        " ./RepoMind",
        "./RepoMind ",
        "./Repo\x00Mind",
        "./Repo\nMind",
        "./Repo\x1fMind",
    ],
)
def test_empty_whitespace_and_control_characters_are_rejected(source: str) -> None:
    with pytest.raises(RepositorySourceValidationError):
        validate_repository_source(source)


@pytest.mark.parametrize(
    "source",
    [
        "http://example.com/team/project.git",
        "git://example.com/team/project.git",
        "ssh://git@example.com/team/project.git",
        "file:///projects/RepoMind",
        "git@example.com:team/project.git",
        "c:https://example.com/team/project.git",
    ],
)
def test_unsupported_remote_source_types_are_rejected(source: str) -> None:
    with pytest.raises(RepositorySourceValidationError):
        validate_repository_source(source)


@pytest.mark.parametrize(
    "source",
    [
        "https:///team/project.git",
        "https://example.com",
        "https://example.com/",
        "https://example.com:not-a-port/team/project.git",
        "https://example.com:70000/team/project.git",
        "https://example.com:0/team/project.git",
        "https://example.com:/team/project.git",
        "https://user@example.com/team/project.git",
        "https://user:password@example.com/team/project.git",
        "https://token@example.com/team/project.git",
        "https://example.com/team/project.git?ref=main",
        "https://example.com/team/project.git#main",
        "https://example.com/team\\project.git",
        "https://example.com/team/my project.git",
        "https://localhost/team/project.git",
        "https://127.0.0.1/team/project.git",
        "https://[::1]/team/project.git",
        "https://10.0.0.1/team/project.git",
        "https://169.254.1.1/team/project.git",
        "https://example.com/team/./project.git",
        "https://example.com/team/../project.git",
    ],
)
def test_malformed_or_risky_https_sources_are_rejected(source: str) -> None:
    with pytest.raises(RepositorySourceValidationError):
        validate_repository_source(source)
