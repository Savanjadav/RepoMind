from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from app.llm_provider import LLMMessage, LLMResponse, LLMRole


@pytest.mark.parametrize(
    ("role", "value"),
    [
        (LLMRole.SYSTEM, "system"),
        (LLMRole.USER, "user"),
        (LLMRole.ASSISTANT, "assistant"),
    ],
)
def test_roles_and_exact_message_content(role: LLMRole, value: str) -> None:
    content = "  Café ✓\nsecond line\n"
    message = LLMMessage(role=role, content=content)
    assert role.value == value
    assert message.role is role
    assert message.content == content


@pytest.mark.parametrize("role", ["user", "invalid", None, 1])
def test_invalid_role_is_not_coerced(role: object) -> None:
    with pytest.raises(ValueError):
        LLMMessage(role=cast(LLMRole, role), content="hello")


@pytest.mark.parametrize("content", [None, 1, True, "", " \n\t"])
def test_invalid_message_content(content: object) -> None:
    with pytest.raises(ValueError):
        LLMMessage(role=LLMRole.USER, content=cast(str, content))


@pytest.mark.parametrize("content", ["  answer ✓\n", "", "   \n"])
def test_response_preserves_content_including_empty(content: str) -> None:
    assert LLMResponse(content=content).content == content


@pytest.mark.parametrize("content", [None, 1, True])
def test_response_rejects_non_string(content: object) -> None:
    with pytest.raises(ValueError):
        LLMResponse(content=cast(str, content))


def test_message_and_response_are_frozen() -> None:
    message = LLMMessage(role=LLMRole.USER, content="hello")
    response = LLMResponse(content="answer")
    with pytest.raises(FrozenInstanceError):
        setattr(message, "content", "changed")
    with pytest.raises(FrozenInstanceError):
        setattr(response, "content", "changed")
