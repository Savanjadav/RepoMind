from collections.abc import Sequence
from dataclasses import dataclass

import pytest

import app.sentence_transformer_embedding_provider as provider_module
from app.embedding_provider import EmbeddingProvider
from app.sentence_transformer_embedding_provider import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbeddingProvider,
)


class NumericValue:
    def __init__(self, value: float) -> None:
        self._value = value

    def __float__(self) -> float:
        return self._value


@dataclass(frozen=True, slots=True)
class EncodeCall:
    texts: tuple[str, ...]
    convert_to_numpy: bool
    convert_to_tensor: bool
    normalize_embeddings: bool
    show_progress_bar: bool


class FakeSentenceTransformer:
    def __init__(
        self,
        *,
        dimension: int | None = 3,
        encoded: list[list[object]] | None = None,
        encode_error: Exception | None = None,
    ) -> None:
        self._dimension = dimension
        self._encoded = encoded
        self._encode_error = encode_error
        self.encode_calls: list[EncodeCall] = []

    def get_embedding_dimension(self) -> int | None:
        return self._dimension

    def encode(
        self,
        texts: Sequence[str],
        *,
        convert_to_numpy: bool,
        convert_to_tensor: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[object]]:
        self.encode_calls.append(
            EncodeCall(
                texts=tuple(texts),
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
            )
        )
        if self._encode_error is not None:
            raise self._encode_error
        if self._encoded is not None:
            return self._encoded

        vectors: list[list[object]] = []
        for index, text in enumerate(texts):
            vectors.append([float(index), float(len(text)), float(text.count("\n"))])
        return vectors


def _install_fake_model(
    monkeypatch: pytest.MonkeyPatch,
    model: FakeSentenceTransformer,
) -> list[tuple[str, bool]]:
    constructor_calls: list[tuple[str, bool]] = []

    def construct(model_name: str, *, trust_remote_code: bool) -> object:
        constructor_calls.append((model_name, trust_remote_code))
        return model

    monkeypatch.setattr(provider_module, "SentenceTransformer", construct)
    return constructor_calls


def _use_provider(provider: EmbeddingProvider) -> EmbeddingProvider:
    return provider


def test_default_model_is_loaded_once_and_dimension_comes_from_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeSentenceTransformer(dimension=3)
    constructor_calls = _install_fake_model(monkeypatch, model)

    provider = _use_provider(SentenceTransformerEmbeddingProvider())
    provider.embed(["first"])
    provider.embed(["second"])

    assert constructor_calls == [(DEFAULT_MODEL_NAME, False)]
    assert provider.dimension == 3
    assert len(model.encode_calls) == 2


def test_custom_model_name_is_passed_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeSentenceTransformer()
    constructor_calls = _install_fake_model(monkeypatch, model)

    SentenceTransformerEmbeddingProvider("trusted/model-name")

    assert constructor_calls == [("trusted/model-name", False)]


@pytest.mark.parametrize("dimension", [None, 0, -1])
def test_invalid_model_dimension_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int | None,
) -> None:
    _install_fake_model(monkeypatch, FakeSentenceTransformer(dimension=dimension))

    with pytest.raises(RuntimeError, match="positive dimension"):
        SentenceTransformerEmbeddingProvider()


def test_batch_is_encoded_once_without_preprocessing_and_converted_to_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ["Café ✓", "def ready():\n    return True\n"]
    model = FakeSentenceTransformer(
        encoded=[
            [NumericValue(1.0), NumericValue(2.0), NumericValue(3.0)],
            [NumericValue(4.0), NumericValue(5.0), NumericValue(6.0)],
        ]
    )
    _install_fake_model(monkeypatch, model)
    provider = SentenceTransformerEmbeddingProvider()

    vectors = provider.embed(texts)

    assert model.encode_calls == [
        EncodeCall(
            texts=tuple(texts),
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
    ]
    assert vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert all(type(value) is float for vector in vectors for value in vector)


def test_empty_input_skips_model_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer()
    _install_fake_model(monkeypatch, model)
    provider = SentenceTransformerEmbeddingProvider()

    assert provider.embed([]) == []
    assert model.encode_calls == []


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        ([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], "vector count"),
        ([[0.0, 1.0]], "vector width"),
    ],
)
def test_malformed_model_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    encoded: list[list[object]],
    message: str,
) -> None:
    model = FakeSentenceTransformer(encoded=encoded)
    _install_fake_model(monkeypatch, model)
    provider = SentenceTransformerEmbeddingProvider()

    with pytest.raises(RuntimeError, match=message):
        provider.embed(["text"])


def test_encode_exception_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ModelFailure(RuntimeError):
        pass

    error = ModelFailure("model failed")
    model = FakeSentenceTransformer(encode_error=error)
    _install_fake_model(monkeypatch, model)
    provider = SentenceTransformerEmbeddingProvider()

    with pytest.raises(ModelFailure) as raised:
        provider.embed(["text"])

    assert raised.value is error
