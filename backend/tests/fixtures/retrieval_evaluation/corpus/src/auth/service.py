from collections.abc import Callable


def authenticate_user(username: str, password: str) -> bool:
    return bool(username and password)


def login_user(
    username: str,
    password: str,
    token_creator: Callable[[str], str],
) -> str:
    if not authenticate_user(username, password):
        raise ValueError("invalid credentials")
    return token_creator(username)
