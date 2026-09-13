from collections.abc import Callable


def login_endpoint(
    username: str,
    password: str,
    login_user: Callable[[str, str], str],
) -> str:
    return login_user(username, password)


def register_routes(routes: dict[str, object]) -> None:
    routes["/api/login"] = login_endpoint
