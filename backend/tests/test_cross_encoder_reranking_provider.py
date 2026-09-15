from collections.abc import Sequence

import pytest

import app.cross_encoder_reranking_provider as provider_module
from app.cross_encoder_reranking_provider import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderRerankingProvider,
)
from app.reranking_provider import RerankerUnavailableError


class FakeCrossEncoder:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[tuple[Sequence[tuple[str, str]], bool, bool, bool]] = []

    def predict(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        convert_to_tensor: bool,
    ) -> object:
        self.calls.append(
            (
                inputs,
                show_progress_bar,
                convert_to_numpy,
                convert_to_tensor,
            )
        )
        return self.output


def test_provider_loads_lazily_batches_pairs_and_reuses_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeCrossEncoder([1, 0.25])
    loaded_names: list[str] = []

    def create_model(model_name: str) -> FakeCrossEncoder:
        loaded_names.append(model_name)
        return model

    monkeypatch.setattr(provider_module, "_create_cross_encoder", create_model)
    provider = CrossEncoderRerankingProvider()

    assert provider.model_name == DEFAULT_RERANKER_MODEL
    assert loaded_names == []
    assert provider.score("query", []) == []
    assert loaded_names == []
    assert provider.score("query", ["first", "second"]) == [1.0, 0.25]
    assert loaded_names == [DEFAULT_RERANKER_MODEL]
    assert model.calls == [
        (
            [("query", "first"), ("query", "second")],
            False,
            True,
            False,
        )
    ]

    model.output = [0.5]
    assert provider.score("another", ["third"]) == [0.5]
    assert loaded_names == [DEFAULT_RERANKER_MODEL]
    assert len(model.calls) == 2


def test_provider_uses_custom_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_names: list[str] = []

    def create_model(model_name: str) -> FakeCrossEncoder:
        loaded_names.append(model_name)
        return FakeCrossEncoder([1.0])

    monkeypatch.setattr(provider_module, "_create_cross_encoder", create_model)
    provider = CrossEncoderRerankingProvider("local/custom-reranker")

    assert provider.model_name == "local/custom-reranker"
    assert provider.score("query", ["document"]) == [1.0]
    assert loaded_names == ["local/custom-reranker"]


def test_cross_encoder_constructor_disables_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeConstructor:
        def __init__(self, model_name: str, *, trust_remote_code: bool) -> None:
            calls.append((model_name, trust_remote_code))

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeConstructor)

    provider_module._create_cross_encoder("configured/model")

    assert calls == [("configured/model", False)]


@pytest.mark.parametrize(
    "output",
    [
        [True],
        ["1.0"],
        [[1.0]],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
        [],
        [1.0, 2.0],
    ],
)
def test_provider_rejects_malformed_scores(
    monkeypatch: pytest.MonkeyPatch,
    output: list[object],
) -> None:
    monkeypatch.setattr(
        provider_module,
        "_create_cross_encoder",
        lambda model_name: FakeCrossEncoder(output),
    )
    provider = CrossEncoderRerankingProvider()

    with pytest.raises(RuntimeError):
        provider.score("query", ["document"])


def test_model_load_oserror_becomes_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(model_name: str) -> FakeCrossEncoder:
        raise OSError("not cached")

    monkeypatch.setattr(provider_module, "_create_cross_encoder", unavailable)
    provider = CrossEncoderRerankingProvider()

    with pytest.raises(RerankerUnavailableError, match=DEFAULT_RERANKER_MODEL):
        provider.score("query", ["document"])


def test_inference_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenCrossEncoder(FakeCrossEncoder):
        def predict(
            self,
            inputs: Sequence[tuple[str, str]],
            *,
            show_progress_bar: bool,
            convert_to_numpy: bool,
            convert_to_tensor: bool,
        ) -> object:
            raise ValueError("inference defect")

    monkeypatch.setattr(
        provider_module,
        "_create_cross_encoder",
        lambda model_name: BrokenCrossEncoder([]),
    )
    provider = CrossEncoderRerankingProvider()

    with pytest.raises(ValueError, match="inference defect"):
        provider.score("query", ["document"])
