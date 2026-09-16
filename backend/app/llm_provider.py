from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class LLMRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: LLMRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, LLMRole):
            raise ValueError("Message role must be an LLMRole")
        if not isinstance(self.content, str):
            raise ValueError("Message content must be a string")
        if not self.content.strip():
            raise ValueError("Message content must contain non-whitespace characters")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise ValueError("Response content must be a string")


class LLMProvider(Protocol):
    @property
    def model_name(self) -> str:
        """Return the configured model identity."""
        ...

    def chat(self, messages: Sequence[LLMMessage]) -> LLMResponse:
        """Generate from a nonempty valid message batch without changing its order."""
        ...
