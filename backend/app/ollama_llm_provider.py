import json
from collections.abc import Sequence
from json import JSONDecodeError
from math import isfinite
from numbers import Real
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from app.llm_provider import (
    LLMMessage,
    LLMProviderError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        return None


class OllamaLLMProvider:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must contain non-whitespace characters")
        self._model_name = model_name
        self._base_url = _validate_base_url(base_url)
        self._timeout_seconds = _validate_timeout(timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._model_name

    def chat(self, messages: Sequence[LLMMessage]) -> LLMResponse:
        batch = tuple(messages)
        if not batch:
            raise ValueError("Message batch must not be empty")
        if any(not isinstance(message, LLMMessage) for message in batch):
            raise ValueError("Every message must be an LLMMessage")

        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {"role": message.role.value, "content": message.content}
                    for message in batch
                ],
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        try:
            response = _post_json(
                url=f"{self._base_url}/api/chat",
                payload=payload,
                timeout_seconds=self._timeout_seconds,
            )
        except HTTPError as error:
            raise LLMProviderError(
                f"Ollama request failed with HTTP status {error.code}"
            ) from error
        except TimeoutError as error:
            raise LLMTimeoutError("Ollama request timed out") from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise LLMTimeoutError("Ollama request timed out") from error
            raise LLMUnavailableError("Ollama service is unavailable") from error

        return _parse_response(response)


def _validate_base_url(base_url: str) -> str:
    if not isinstance(base_url, str):
        raise ValueError("Base URL must be a non-whitespace string")
    if any(ord(character) <= 0x1F or ord(character) == 0x7F for character in base_url):
        raise ValueError("Base URL must not contain ASCII control characters")
    if not base_url or base_url != base_url.strip():
        raise ValueError("Base URL must be a non-whitespace string")
    try:
        parsed = urlsplit(base_url)
        parsed_port = parsed.port
    except ValueError as error:
        raise ValueError("Base URL is malformed") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed_port is None
        and ":" in parsed.netloc.removeprefix("[::1]")
    ):
        raise ValueError("Base URL must be a loopback HTTP root URL")
    return base_url[:-1] if parsed.path == "/" else base_url


def _validate_timeout(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real):
        raise ValueError("Timeout must be a positive finite real number")
    timeout = float(timeout_seconds)
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be a positive finite real number")
    return timeout


def _post_json(*, url: str, payload: bytes, timeout_seconds: float) -> bytes:
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
    with opener.open(request, timeout=timeout_seconds) as response:
        return response.read()


def _parse_response(response: bytes) -> LLMResponse:
    try:
        document = json.loads(response.decode("utf-8"))
    except (UnicodeDecodeError, JSONDecodeError) as error:
        raise LLMResponseError("Ollama returned malformed JSON") from error
    if not isinstance(document, dict):
        raise LLMResponseError("Ollama response must be a JSON object")
    message = document.get("message")
    if not isinstance(message, dict):
        raise LLMResponseError("Ollama response message must be a JSON object")
    content = message.get("content")
    if not isinstance(content, str):
        raise LLMResponseError("Ollama response content must be a string")
    return LLMResponse(content=content)
