import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from ipaddress import ip_address
from urllib.parse import urlsplit

WINDOWS_DRIVE_PREFIX_PATTERN = re.compile(r"^[A-Za-z]:")
URL_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
SCP_STYLE_GIT_PATTERN = re.compile(r"^[^/\\\s@]+@[^/\\\s:]+:.+$")


class RepositorySourceKind(StrEnum):
    HTTPS = "https"
    LOCAL = "local"


@dataclass(frozen=True, slots=True)
class ValidatedRepositorySource:
    kind: RepositorySourceKind
    value: str


class RepositorySourceValidationError(ValueError):
    """Raised when a repository source has an unsupported or unsafe structure."""


def validate_repository_source(source: str) -> ValidatedRepositorySource:
    _validate_common_source_rules(source)

    if _is_windows_drive_path(source):
        return ValidatedRepositorySource(RepositorySourceKind.LOCAL, source)

    scheme_match = URL_SCHEME_PATTERN.match(source)
    if scheme_match:
        scheme = source[: source.index(":")].lower()
        if scheme != RepositorySourceKind.HTTPS:
            raise RepositorySourceValidationError(
                f"Unsupported repository source scheme: {scheme}"
            )
        return _validate_https_source(source)

    if SCP_STYLE_GIT_PATTERN.match(source):
        raise RepositorySourceValidationError(
            "SSH/SCP-style repository sources are not supported"
        )

    return ValidatedRepositorySource(RepositorySourceKind.LOCAL, source)


def _is_windows_drive_path(source: str) -> bool:
    if not WINDOWS_DRIVE_PREFIX_PATTERN.match(source):
        return False

    path_after_drive = source[2:]
    return bool(path_after_drive) and not URL_SCHEME_PATTERN.match(path_after_drive)


def _validate_common_source_rules(source: str) -> None:
    if not source or source.isspace():
        raise RepositorySourceValidationError("Repository source must not be empty")
    if source != source.strip():
        raise RepositorySourceValidationError(
            "Repository source must not have leading or trailing whitespace"
        )
    if any(unicodedata.category(character) == "Cc" for character in source):
        raise RepositorySourceValidationError(
            "Repository source must not contain control characters"
        )


def _validate_https_source(source: str) -> ValidatedRepositorySource:
    if any(character.isspace() for character in source):
        raise RepositorySourceValidationError(
            "HTTPS repository URLs must not contain whitespace"
        )
    if "\\" in source:
        raise RepositorySourceValidationError(
            "HTTPS repository URLs must not contain backslashes"
        )
    if "?" in source:
        raise RepositorySourceValidationError(
            "HTTPS repository URLs must not contain a query string"
        )
    if "#" in source:
        raise RepositorySourceValidationError(
            "HTTPS repository URLs must not contain a fragment"
        )

    try:
        parsed = urlsplit(source)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise RepositorySourceValidationError(
            "HTTPS repository URL is malformed"
        ) from error

    if hostname is None:
        raise RepositorySourceValidationError(
            "HTTPS repository URL must include a hostname"
        )
    if parsed.username is not None or parsed.password is not None:
        raise RepositorySourceValidationError(
            "HTTPS repository URLs must not contain credentials"
        )
    if parsed.netloc.endswith(":"):
        raise RepositorySourceValidationError(
            "HTTPS repository URL must not contain an empty explicit port"
        )
    if port == 0:
        raise RepositorySourceValidationError(
            "HTTPS repository URL must use a valid port"
        )

    path_segments = [segment for segment in parsed.path.split("/") if segment]
    if not path_segments:
        raise RepositorySourceValidationError(
            "HTTPS repository URL must include a repository path"
        )
    if any(segment in {".", ".."} for segment in path_segments):
        raise RepositorySourceValidationError(
            "HTTPS repository URL must not contain dot path segments"
        )

    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname == "localhost" or normalized_hostname.endswith(".localhost"):
        raise RepositorySourceValidationError(
            "HTTPS repository URL must not target localhost"
        )

    try:
        literal_ip = ip_address(hostname)
    except ValueError:
        pass
    else:
        if not literal_ip.is_global:
            raise RepositorySourceValidationError(
                "HTTPS repository URL must not target a non-public IP address"
            )

    return ValidatedRepositorySource(RepositorySourceKind.HTTPS, source)
