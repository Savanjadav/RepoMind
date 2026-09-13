import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import app.retrieval_evaluation as evaluation_module
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.repository import Repository
from app.retrieval_evaluation import (
    K_VALUES,
    EmbeddingMetadata,
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationReport,
    ExpectedTarget,
    QuestionMetrics,
    RankedIdentity,
    RetrievalModeMetrics,
    aggregate_question_metrics,
    evaluate_question,
    load_evaluation_dataset,
    recall_at_k,
    reciprocal_rank,
    run_evaluation,
    serialize_report,
    target_matches_result,
)

DATABASE_URL = os.getenv("DATABASE_URL")


def _identity(path: str, symbol: str | None) -> RankedIdentity:
    return RankedIdentity(path=path, symbol_name=symbol)


def _target(path: str, symbol: str | None) -> ExpectedTarget:
    return ExpectedTarget(path=path, symbol=symbol)


def _question(*targets: ExpectedTarget) -> EvaluationQuestion:
    return EvaluationQuestion(
        id="test-question",
        category="conceptual",
        question="How does the feature work?",
        expected_targets=targets,
    )


@pytest.mark.parametrize(
    ("target", "result", "matches"),
    [
        (_target("src/auth.py", "login"), _identity("src/auth.py", "login"), True),
        (_target("src/auth.py", "login"), _identity("src/api.py", "login"), False),
        (_target("src/auth.py", "login"), _identity("src/auth.py", "logout"), False),
        (_target("README.md", None), _identity("README.md", "Overview"), True),
        (_target("README.md", None), _identity("docs/README.md", None), False),
    ],
)
def test_target_matching_is_exact(
    target: ExpectedTarget,
    result: RankedIdentity,
    matches: bool,
) -> None:
    assert target_matches_result(target, result) is matches


def test_recall_at_k_handles_partial_full_and_truncated_results() -> None:
    expected = (_target("a.py", "a"), _target("b.py", "b"))
    results = (
        _identity("x.py", "x"),
        _identity("a.py", "a"),
        _identity("y.py", "y"),
        _identity("b.py", "b"),
    )

    assert recall_at_k(expected, results, 1) == 0.0
    assert recall_at_k(expected, results, 3) == 0.5
    assert recall_at_k(expected, results, 5) == 1.0
    assert recall_at_k(expected, results, 10) == 1.0


def test_recall_at_k_handles_found_absent_and_duplicate_results() -> None:
    expected = (_target("a.py", "a"), _target("b.py", "b"))

    assert recall_at_k(expected, [_identity("a.py", "a")], 1) == 0.5
    assert recall_at_k(expected, [_identity("x.py", "x")], 1) == 0.0
    assert (
        recall_at_k(
            expected,
            [_identity("a.py", "a"), _identity("a.py", "a")],
            10,
        )
        == 0.5
    )


def test_recall_requires_expected_targets_and_positive_k() -> None:
    with pytest.raises(ValueError, match="Expected targets must not be empty"):
        recall_at_k([], [], 1)
    with pytest.raises(ValueError, match="K must be a positive integer"):
        recall_at_k([_target("a.py", "a")], [], 0)


def test_reciprocal_rank_uses_earliest_relevant_result() -> None:
    expected = (_target("a.py", "a"), _target("b.py", "b"))

    assert reciprocal_rank(expected, [_identity("a.py", "a")]) == 1.0
    assert (
        reciprocal_rank(
            expected,
            [_identity("x.py", "x"), _identity("b.py", "b")],
        )
        == 0.5
    )
    assert reciprocal_rank(expected, [_identity("x.py", "x")]) == 0.0
    assert (
        reciprocal_rank(
            expected,
            [
                _identity("x.py", "x"),
                _identity("b.py", "b"),
                _identity("a.py", "a"),
            ],
        )
        == 0.5
    )


def test_question_metrics_use_one_based_first_relevant_rank() -> None:
    metrics = evaluate_question(
        _question(_target("target.py", "target")),
        [_identity("other.py", "other"), _identity("target.py", "target")],
    )

    assert metrics.first_relevant_rank == 2
    assert metrics.reciprocal_rank == 0.5
    assert metrics.recall_at_k == {1: 0.0, 3: 1.0, 5: 1.0, 10: 1.0}


def test_aggregation_macro_averages_questions() -> None:
    first = QuestionMetrics(
        id="first",
        recall_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        reciprocal_rank=1.0,
        first_relevant_rank=1,
        results=(),
    )
    second = QuestionMetrics(
        id="second",
        recall_at_k={1: 0.0, 3: 0.5, 5: 0.5, 10: 1.0},
        reciprocal_rank=0.25,
        first_relevant_rank=4,
        results=(),
    )

    aggregate = aggregate_question_metrics([first, second])

    assert aggregate.recall_at_k == {1: 0.5, 3: 0.75, 5: 0.75, 10: 1.0}
    assert aggregate.mrr == 0.625


def test_committed_dataset_loads_defensively() -> None:
    dataset = load_evaluation_dataset()

    assert dataset.version == 1
    assert dataset.name == "atlas-service"
    assert dataset.corpus_path.is_dir()
    assert len(dataset.questions) == 10
    assert all(question.expected_targets for question in dataset.questions)


def _sample_report() -> EvaluationReport:
    dataset = EvaluationDataset(
        version=1,
        name="atlas-service",
        corpus_relative_path="backend/tests/fixtures/retrieval_evaluation/corpus",
        corpus_path=Path("corpus"),
        questions=(_question(_target("target.py", "target")),),
    )
    question = QuestionMetrics(
        id="test-question",
        recall_at_k={1: 1 / 3, 3: 2 / 3, 5: 1.0, 10: 1.0},
        reciprocal_rank=1 / 3,
        first_relevant_rank=3,
        results=(_identity("target.py", "target"),),
    )
    mode = RetrievalModeMetrics(
        recall_at_k=dict(question.recall_at_k),
        mrr=question.reciprocal_rank,
        questions=(question,),
    )
    return EvaluationReport(
        dataset=dataset,
        embedding=EmbeddingMetadata(
            provider="FakeEmbeddingProvider",
            model="fake-model",
            dimension=EMBEDDING_DIMENSION,
            sentence_transformers_version="test-version",
        ),
        code_unit_count=15,
        semantic=mode,
        hybrid=mode,
    )


def test_report_serialization_is_deterministic_and_rounded() -> None:
    first = serialize_report(_sample_report())
    second = serialize_report(_sample_report())
    document = json.loads(first)

    assert first == second
    assert first.endswith("\n")
    assert document["modes"]["semantic"]["mrr"] == 0.333333
    assert document["modes"]["semantic"]["recall_at_k"] == {
        "1": 0.333333,
        "3": 0.666667,
        "5": 1.0,
        "10": 1.0,
    }


def test_normal_cli_run_does_not_write_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[EvaluationReport] = []
    monkeypatch.setattr(evaluation_module, "_run_real_evaluation", _sample_report)
    monkeypatch.setattr(evaluation_module, "_print_summary", lambda report: None)
    monkeypatch.setattr(evaluation_module, "_write_baseline", writes.append)

    assert evaluation_module.main([]) == 0
    assert writes == []


def test_write_flag_is_the_only_cli_path_that_writes_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(evaluation_module, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(evaluation_module, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluation_module, "_run_real_evaluation", _sample_report)
    monkeypatch.setattr(evaluation_module, "_print_summary", lambda report: None)

    assert evaluation_module.main(["--write-baseline"]) == 0
    assert baseline_path.read_text(encoding="utf-8") == serialize_report(
        _sample_report()
    )


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        batch = tuple(texts)
        self.batches.append(batch)
        return [_vector_for_text(text) for text in batch]


def _vector_for_text(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [
        (digest[index % len(digest)] + 1) / 256 for index in range(EMBEDDING_DIMENSION)
    ]


def _database_counts(database_url: str) -> tuple[int, int, int]:
    from app.database import create_database_engine

    engine = create_database_engine(database_url)
    try:
        with Session(engine) as session:
            repository_count = session.scalar(
                select(func.count()).select_from(Repository)
            )
            file_count = session.scalar(select(func.count()).select_from(File))
            code_unit_count = session.scalar(select(func.count()).select_from(CodeUnit))
            assert repository_count is not None
            assert file_count is not None
            assert code_unit_count is not None
            return repository_count, file_count, code_unit_count
    finally:
        engine.dispose()


@pytest.mark.skipif(DATABASE_URL is None, reason="DATABASE_URL is not configured")
def test_real_staging_and_retrieval_are_isolated() -> None:
    assert DATABASE_URL is not None
    provider = FakeEmbeddingProvider()
    metadata = EmbeddingMetadata(
        provider=type(provider).__name__,
        model="deterministic-test-vectors",
        dimension=provider.dimension,
        sentence_transformers_version="not-used",
    )
    counts_before = _database_counts(DATABASE_URL)
    baseline_before = (
        evaluation_module.BASELINE_PATH.read_bytes()
        if evaluation_module.BASELINE_PATH.exists()
        else None
    )

    report = run_evaluation(
        embedding_provider=provider,
        embedding_metadata=metadata,
        database_url=DATABASE_URL,
    )

    assert report.code_unit_count == 15
    assert len(report.dataset.questions) == 10
    assert len(report.semantic.questions) == 10
    assert len(report.hybrid.questions) == 10
    assert set(K_VALUES) == set(report.semantic.recall_at_k)
    assert set(K_VALUES) == set(report.hybrid.recall_at_k)
    assert [len(batch) for batch in provider.batches] == [15, 10]
    assert _database_counts(DATABASE_URL) == counts_before
    baseline_after = (
        evaluation_module.BASELINE_PATH.read_bytes()
        if evaluation_module.BASELINE_PATH.exists()
        else None
    )
    assert baseline_after == baseline_before
