from collections.abc import Sequence
from typing import Protocol


class RerankerUnavailableError(RuntimeError):
    """The configured local reranker could not be loaded."""


class RerankingProvider(Protocol):
    @property
    def model_name(self) -> str:
        """Return the configured reranking model identifier."""
        ...

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document, preserving input order."""
        ...
