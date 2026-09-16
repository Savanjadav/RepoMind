from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from app.llm_provider import LLMMessage, LLMProvider, LLMResponse, LLMRole
from app.mock_llm_provider import MockLLMProvider


def _ask(provider: LLMProvider, messages: Sequence[LLMMessage]) -> LLMResponse:
    return provider.chat(messages)


@pytest.mark.parametrize("name", ["custom", "  custom\n"])
def test_custom_model_name_preserved(name: str) -> None:
    assert MockLLMProvider(response="answer", model_name=name).model_name == name


@pytest.mark.parametrize("name", ["", " \n", None, 1])
def test_invalid_model_name(name: object) -> None:
    with pytest.raises(ValueError):
        MockLLMProvider(response="answer", model_name=cast(str, name))


@pytest.mark.parametrize("content", ["  answer ✓\n", "", "   "])
def test_configured_response_and_provider_independence(content: str) -> None:
    provider = MockLLMProvider(response=content)
    messages = [LLMMessage(role=LLMRole.USER, content="question")]
    first = _ask(provider, messages)
    assert provider.model_name == "mock"
    assert first.content == content
    assert _ask(provider, messages) is first
    with pytest.raises(FrozenInstanceError):
        setattr(first, "content", "changed")


@pytest.mark.parametrize("response", [None, 1, True])
def test_invalid_configured_response(response: object) -> None:
    with pytest.raises(ValueError):
        MockLLMProvider(response=cast(str, response))


@pytest.mark.parametrize(
    "members", [[], ["user"], [LLMMessage(LLMRole.USER, "hi"), None]]
)
def test_invalid_calls_not_recorded(members: list[object]) -> None:
    provider = MockLLMProvider(response="answer")
    with pytest.raises(ValueError):
        provider.chat(cast(Sequence[LLMMessage], members))
    assert provider.calls == ()


def test_order_and_history_are_immutable_snapshots() -> None:
    provider = MockLLMProvider(response="answer")
    messages = [
        LLMMessage(LLMRole.ASSISTANT, "previous answer"),
        LLMMessage(LLMRole.SYSTEM, "first instruction"),
        LLMMessage(LLMRole.SYSTEM, "second instruction"),
        LLMMessage(LLMRole.USER, "  question\n"),
    ]
    original = tuple(messages)
    provider.chat(messages)
    snapshot = provider.calls
    assert snapshot == (original,)
    assert tuple(messages) == original
    messages.clear()
    messages.append(LLMMessage(LLMRole.USER, "next question"))
    provider.chat(messages)
    assert snapshot == (original,)
    assert provider.calls == (original, tuple(messages))
    extended = snapshot + ((),)
    assert extended != provider.calls
    assert provider.calls[0] == original


def test_model_name_has_no_setter() -> None:
    provider = MockLLMProvider(response="answer")
    with pytest.raises(AttributeError):
        setattr(provider, "model_name", "changed")
