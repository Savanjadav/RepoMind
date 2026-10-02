import os
from urllib.parse import urlsplit

DATABASE_URL_ENV_VAR = "DATABASE_URL"


def get_database_url() -> str:
    database_url = os.getenv(DATABASE_URL_ENV_VAR)
    if not database_url:
        raise RuntimeError(f"{DATABASE_URL_ENV_VAR} environment variable is required")
    return database_url


def get_redis_url() -> str | None:
    value = os.getenv("REDIS_URL")
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        if (
            value != value.strip()
            or any(ord(c) <= 31 or ord(c) == 127 for c in value)
            or parsed.scheme not in {"redis", "rediss"}
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        if parsed.path not in {"", "/"}:
            selector = parsed.path[1:]
            if not selector or any(c < "0" or c > "9" for c in selector):
                raise ValueError
            # Prove conversion succeeds too; redis-py otherwise defaults to DB 0.
            if int(selector) < 0:
                raise ValueError
        _ = parsed.port
    except ValueError:
        raise ValueError("REDIS_URL is invalid") from None
    return value
