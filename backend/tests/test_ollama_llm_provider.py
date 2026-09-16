import json
import socket
from collections.abc import Sequence
from contextlib import AbstractContextManager
from email.message import Message
from fractions import Fraction
from http.client import HTTPMessage
from io import BytesIO
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request

import pytest

import app.ollama_llm_provider as provider_module
from app.llm_provider import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    LLMResponseError,
    LLMRole,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.ollama_llm_provider import OllamaLLMProvider


def _chat(provider: LLMProvider, messages: Sequence[LLMMessage]) -> LLMResponse:
    return provider.chat(messages)


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    response: bytes = b'{"message":{"content":"answer"}}',
) -> list[tuple[str, bytes, float]]:
    calls: list[tuple[str, bytes, float]] = []

    def post_json(*, url: str, payload: bytes, timeout_seconds: float) -> bytes:
        calls.append((url, payload, timeout_seconds))
        return response

    monkeypatch.setattr(provider_module, "_post_json", post_json)
    return calls


@pytest.mark.parametrize("model_name", ["model", "  model\n"])
def test_model_name_is_preserved(model_name: str) -> None:
    assert OllamaLLMProvider(model_name=model_name).model_name == model_name


@pytest.mark.parametrize("model_name", ["", " \n", None, 1, True])
def test_invalid_model_name_is_rejected(model_name: object) -> None:
    with pytest.raises(ValueError):
        OllamaLLMProvider(model_name=cast(str, model_name))


@pytest.mark.parametrize(
    ("timeout", "expected"),
    [(120.0, 120.0), (2, 2.0), (0.25, 0.25), (Fraction(3, 2), 1.5)],
)
def test_valid_timeout_is_passed_to_transport(
    monkeypatch: pytest.MonkeyPatch, timeout: object, expected: float
) -> None:
    calls = _install_transport(monkeypatch)
    provider = OllamaLLMProvider(
        model_name="model", timeout_seconds=cast(float, timeout)
    )
    provider.chat([LLMMessage(LLMRole.USER, "question")])
    assert calls[0][2] == expected
    assert type(calls[0][2]) is float


@pytest.mark.parametrize(
    "timeout", [True, False, "1", 0, -1, float("nan"), float("inf"), float("-inf")]
)
def test_invalid_timeout_is_rejected(timeout: object) -> None:
    with pytest.raises(ValueError):
        OllamaLLMProvider(model_name="model", timeout_seconds=cast(float, timeout))


@pytest.mark.parametrize(
    ("base_url", "endpoint"),
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434/api/chat"),
        ("http://127.0.0.1:11434/", "http://127.0.0.1:11434/api/chat"),
        ("http://localhost:11434", "http://localhost:11434/api/chat"),
        ("http://localhost:11434/", "http://localhost:11434/api/chat"),
        ("http://[::1]:11434", "http://[::1]:11434/api/chat"),
        ("http://[::1]:11434/", "http://[::1]:11434/api/chat"),
    ],
)
def test_loopback_base_urls_build_exact_endpoint(
    monkeypatch: pytest.MonkeyPatch, base_url: str, endpoint: str
) -> None:
    calls = _install_transport(monkeypatch)
    OllamaLLMProvider(model_name="model", base_url=base_url).chat(
        [LLMMessage(LLMRole.USER, "question")]
    )
    assert calls[0][0] == endpoint


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        " ",
        " http://localhost:11434",
        "https://localhost:11434",
        "http://example.com:11434",
        "http://192.168.1.2:11434",
        "http://0.0.0.0:11434",
        "http://user:pass@localhost:11434",
        "http://localhost:11434?query=yes",
        "http://localhost:11434#fragment",
        "http://localhost:11434/api",
        "http://localhost:not-a-port",
        "http:///missing-host",
        "not-a-url",
        None,
        1,
    ],
)
def test_invalid_base_url_is_rejected(base_url: object) -> None:
    with pytest.raises(ValueError):
        OllamaLLMProvider(model_name="model", base_url=cast(str, base_url))


@pytest.mark.parametrize(
    "base_url",
    [
        "http://local\thost:11434",
        "http://local\nhost:11434",
        "http://local\rhost:11434",
        "http://localhost:11434\x00",
        "http://local\x1fhost:11434",
        "http://localhost:11434\x7f",
    ],
)
def test_ascii_control_characters_are_rejected_before_url_parsing(
    base_url: str,
) -> None:
    with pytest.raises(ValueError) as raised:
        OllamaLLMProvider(model_name="model", base_url=base_url)

    assert str(raised.value) == "Base URL must not contain ASCII control characters"
    assert base_url not in str(raised.value)


def test_chat_sends_exact_semantic_payload_without_mutating_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_transport(
        monkeypatch,
        json.dumps({"message": {"content": "  Café ✓\n"}}).encode("utf-8"),
    )
    messages = [
        LLMMessage(LLMRole.SYSTEM, "  instruction\n"),
        LLMMessage(LLMRole.ASSISTANT, "previous answer"),
        LLMMessage(LLMRole.USER, "Café ✓\nquestion\n"),
    ]
    original = tuple(messages)
    result = _chat(OllamaLLMProvider(model_name="  exact-model\n"), messages)

    assert result.content == "  Café ✓\n"
    assert tuple(messages) == original
    assert len(calls) == 1
    url, payload, timeout = calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert timeout == 120.0
    assert json.loads(payload.decode("utf-8")) == {
        "model": "  exact-model\n",
        "messages": [
            {"role": "system", "content": "  instruction\n"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "Café ✓\nquestion\n"},
        ],
        "stream": False,
    }


@pytest.mark.parametrize(
    "messages", [[], ["user"], [LLMMessage(LLMRole.USER, "valid"), None]]
)
def test_invalid_batch_does_not_call_transport(
    monkeypatch: pytest.MonkeyPatch, messages: list[object]
) -> None:
    calls = _install_transport(monkeypatch)
    with pytest.raises(ValueError):
        OllamaLLMProvider(model_name="model").chat(cast(Sequence[LLMMessage], messages))
    assert calls == []


@pytest.mark.parametrize("content", ["answer", "", "   ", "Café ✓", "one\ntwo\n"])
def test_response_content_is_preserved(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    response = json.dumps({"message": {"content": content}}).encode()
    _install_transport(monkeypatch, response)
    result = OllamaLLMProvider(model_name="model").chat(
        [LLMMessage(LLMRole.USER, "question")]
    )
    assert result.content == content


@pytest.mark.parametrize(
    "response",
    [
        b"\xff",
        b"not json",
        b"null",
        b"[]",
        b'"text"',
        b"{}",
        b'{"message":null}',
        b'{"message":[]}',
        b'{"message":"text"}',
        b'{"message":{}}',
        b'{"message":{"content":null}}',
        b'{"message":{"content":1}}',
        b'{"message":{"content":[]}}',
        b'{"message":{"content":{}}}',
    ],
)
def test_malformed_response_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, response: bytes
) -> None:
    _install_transport(monkeypatch, response)
    with pytest.raises(LLMResponseError) as raised:
        OllamaLLMProvider(model_name="model").chat(
            [LLMMessage(LLMRole.USER, "SECRET_PROMPT")]
        )
    assert "SECRET" not in str(raised.value)


def test_malformed_response_body_is_not_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_transport(monkeypatch, b"SECRET_RESPONSE_BODY")
    with pytest.raises(LLMResponseError) as raised:
        OllamaLLMProvider(model_name="model").chat(
            [LLMMessage(LLMRole.USER, "SECRET_PROMPT")]
        )
    assert "SECRET" not in str(raised.value)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("SECRET"), LLMTimeoutError),
        (getattr(socket, "timeout")("SECRET"), LLMTimeoutError),
        (URLError(TimeoutError("SECRET")), LLMTimeoutError),
        (URLError(getattr(socket, "timeout")("SECRET")), LLMTimeoutError),
        (URLError("SECRET"), LLMUnavailableError),
    ],
)
def test_transport_errors_are_mapped_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected: type[LLMProviderError],
) -> None:
    def fail(**kwargs: object) -> bytes:
        raise error

    monkeypatch.setattr(provider_module, "_post_json", fail)
    with pytest.raises(expected) as raised:
        OllamaLLMProvider(model_name="model").chat(
            [LLMMessage(LLMRole.USER, "SECRET_PROMPT")]
        )
    assert "SECRET" not in str(raised.value)


@pytest.mark.parametrize("status", [302, 400, 404, 500])
def test_http_error_is_mapped_with_status_only(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    error = HTTPError(
        "http://127.0.0.1:11434/api/chat",
        status,
        "SECRET_RESPONSE_BODY",
        Message(),
        BytesIO(b"SECRET_RESPONSE_BODY"),
    )

    def fail(**kwargs: object) -> bytes:
        raise error

    monkeypatch.setattr(provider_module, "_post_json", fail)
    with pytest.raises(LLMProviderError) as raised:
        OllamaLLMProvider(model_name="model").chat(
            [LLMMessage(LLMRole.USER, "SECRET_PROMPT")]
        )
    assert str(status) in str(raised.value)
    assert "SECRET" not in str(raised.value)


def test_unexpected_exception_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("unexpected")

    def fail(**kwargs: object) -> bytes:
        raise error

    monkeypatch.setattr(provider_module, "_post_json", fail)
    with pytest.raises(RuntimeError) as raised:
        OllamaLLMProvider(model_name="model").chat(
            [LLMMessage(LLMRole.USER, "question")]
        )
    assert raised.value is error


def test_generic_error_inheritance() -> None:
    assert issubclass(LLMUnavailableError, LLMProviderError)
    assert issubclass(LLMTimeoutError, LLMProviderError)
    assert issubclass(LLMResponseError, LLMProviderError)


class _Response(AbstractContextManager["_Response"]):
    def read(self) -> bytes:
        return b"response"

    def __exit__(self, *args: object) -> None:
        return None


def test_private_transport_is_post_only_proxy_free_and_redirect_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_handlers: list[tuple[object, ...]] = []
    observed_open: list[tuple[Request, float]] = []

    class Opener:
        def open(
            self, request: Request, *, timeout: float
        ) -> AbstractContextManager[_Response]:
            observed_open.append((request, timeout))
            return _Response()

    def build(*handlers: object) -> Opener:
        observed_handlers.append(handlers)
        return Opener()

    monkeypatch.setattr(provider_module, "build_opener", build)
    result = provider_module._post_json(
        url="http://localhost:11434/api/chat",
        payload=b"payload",
        timeout_seconds=3.0,
    )

    assert result == b"response"
    assert len(observed_handlers) == 1
    proxy_handler, redirect_handler = observed_handlers[0]
    assert isinstance(proxy_handler, ProxyHandler)
    assert vars(proxy_handler)["proxies"] == {}
    assert isinstance(redirect_handler, provider_module._NoRedirectHandler)
    redirect_result = cast(HTTPRedirectHandler, redirect_handler).redirect_request(
        Request("http://localhost:11434/api/chat"),
        BytesIO(),
        302,
        "Found",
        HTTPMessage(),
        "http://remote.example/escape",
    )
    assert redirect_result is None
    request, timeout = observed_open[0]
    assert request.full_url == "http://localhost:11434/api/chat"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.data == b"payload"
    assert timeout == 3.0
