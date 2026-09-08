from collections.abc import Sequence

from app.embedding_provider import EmbeddingProvider


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.received_texts: list[str] = []

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.received_texts = list(texts)
        return [
            [float(index), float(len(text)), float(text.count("\n"))]
            for index, text in enumerate(texts)
        ]


def _embed_with(
    provider: EmbeddingProvider,
    texts: Sequence[str],
) -> list[list[float]]:
    return provider.embed(texts)


def test_fake_provider_structurally_satisfies_batch_contract() -> None:
    provider = FakeEmbeddingProvider()
    texts = ["alpha", "beta", "gamma"]

    vectors = _embed_with(provider, texts)

    assert provider.received_texts == texts
    assert len(vectors) == len(texts)
    assert vectors == [
        [0.0, 5.0, 0.0],
        [1.0, 4.0, 0.0],
        [2.0, 5.0, 0.0],
    ]


def test_dimension_is_positive_and_matches_every_vector_width() -> None:
    provider = FakeEmbeddingProvider()

    vectors = _embed_with(provider, ["one", "two"])

    assert provider.dimension > 0
    assert all(len(vector) == provider.dimension for vector in vectors)
    assert all(isinstance(value, float) for vector in vectors for value in vector)


def test_empty_batch_returns_empty_list() -> None:
    provider = FakeEmbeddingProvider()

    assert _embed_with(provider, []) == []
    assert provider.received_texts == []


def test_unicode_and_multiline_text_are_received_unchanged() -> None:
    provider = FakeEmbeddingProvider()
    texts = ["Café ✓", "def ready():\n    return True\n"]

    _embed_with(provider, texts)

    assert provider.received_texts == texts


def test_fake_provider_is_deterministic() -> None:
    provider = FakeEmbeddingProvider()
    texts = ("first", "second")

    first = _embed_with(provider, texts)
    second = _embed_with(provider, texts)

    assert first == second
