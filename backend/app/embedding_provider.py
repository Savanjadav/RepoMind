from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        """Return the positive width of every embedding vector."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in input order, returning one vector per text."""
        ...
