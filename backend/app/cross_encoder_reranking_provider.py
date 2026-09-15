from collections.abc import Iterable, Sequence
from math import isfinite
from numbers import Real
from typing import Protocol

from app.reranking_provider import RerankerUnavailableError

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class _CrossEncoderModel(Protocol):
    def predict(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        convert_to_tensor: bool,
    ) -> object: ...


class CrossEncoderRerankingProvider:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self._model_name = model_name
        self._model: _CrossEncoderModel | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        input_documents = list(documents)
        if not input_documents:
            return []

        model = self._get_model()
        raw_scores = model.predict(
            [(query, document) for document in input_documents],
            show_progress_bar=False,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )
        return _normalize_scores(raw_scores, expected_count=len(input_documents))

    def _get_model(self) -> _CrossEncoderModel:
        if self._model is None:
            try:
                self._model = _create_cross_encoder(self.model_name)
            except OSError as error:
                raise RerankerUnavailableError(
                    f"Reranking model is unavailable: {self.model_name}"
                ) from error
        return self._model


def _create_cross_encoder(model_name: str) -> _CrossEncoderModel:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, trust_remote_code=False)


def _normalize_scores(raw_scores: object, *, expected_count: int) -> list[float]:
    if isinstance(raw_scores, (str, bytes)) or not isinstance(raw_scores, Iterable):
        raise RuntimeError("Reranker returned malformed scores")
    values = list(raw_scores)
    if len(values) != expected_count:
        raise RuntimeError("Reranker returned an unexpected score count")

    scores: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError("Reranker scores must be real numbers")
        score = float(value)
        if not isfinite(score):
            raise RuntimeError("Reranker scores must be finite")
        scores.append(score)
    return scores
