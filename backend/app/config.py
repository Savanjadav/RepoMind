import os

DATABASE_URL_ENV_VAR = "DATABASE_URL"


def get_database_url() -> str:
    database_url = os.getenv(DATABASE_URL_ENV_VAR)
    if not database_url:
        raise RuntimeError(f"{DATABASE_URL_ENV_VAR} environment variable is required")
    return database_url
