def create_access_token(subject: str) -> str:
    return f"token:{subject}"


def decode_access_token(token: str) -> str:
    return token.removeprefix("token:")
