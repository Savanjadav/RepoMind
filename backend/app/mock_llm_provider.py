from collections.abc import Sequence

from app.llm_provider import LLMMessage, LLMResponse


class MockLLMProvider:
    def __init__(self, *, response: str, model_name: str = "mock") -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must contain non-whitespace characters")
        self._model_name = model_name
        self._response = LLMResponse(content=response)
        self._calls: list[tuple[LLMMessage, ...]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def calls(self) -> tuple[tuple[LLMMessage, ...], ...]:
        return tuple(self._calls)

    def chat(self, messages: Sequence[LLMMessage]) -> LLMResponse:
        batch = tuple(messages)
        if not batch:
            raise ValueError("Message batch must not be empty")
        if any(not isinstance(message, LLMMessage) for message in batch):
            raise ValueError("Every message must be an LLMMessage")
        self._calls.append(batch)
        return self._response
