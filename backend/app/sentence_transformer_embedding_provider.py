from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name, trust_remote_code=False)
        dimension = self._model.get_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise RuntimeError("Embedding model did not report a positive dimension")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        input_texts = list(texts)
        if not input_texts:
            return []

        encoded = self._model.encode(
            input_texts,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vectors = [[float(value) for value in row] for row in encoded]

        if len(vectors) != len(input_texts):
            raise RuntimeError("Embedding model returned an unexpected vector count")
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError("Embedding model returned an unexpected vector width")
        return vectors
