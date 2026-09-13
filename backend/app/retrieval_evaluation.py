import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import create_database_engine
from app.embedding_provider import EmbeddingProvider
from app.hybrid_search import (
    CANDIDATE_MULTIPLIER,
    RRF_K,
    search_code_units_hybrid,
)
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.repository import Repository
from app.repository_clone import ClonedRepository
from app.repository_indexing import _index_cloned_repository
from app.semantic_search import search_code_units_semantically
from app.sentence_transformer_embedding_provider import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbeddingProvider,
)

K_VALUES = (1, 3, 5, 10)
RESULT_LIMIT = 10
EXPECTED_FILE_COUNT = 7
EXPECTED_CODE_UNIT_COUNT = 15

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _BACKEND_ROOT.parent
EVALUATION_ROOT = _BACKEND_ROOT / "tests" / "fixtures" / "retrieval_evaluation"
EVALUATION_DATASET_PATH = EVALUATION_ROOT / "evaluation.json"
BASELINE_PATH = EVALUATION_ROOT / "baseline.json"
_EXPECTED_CORPUS_PATH = "backend/tests/fixtures/retrieval_evaluation/corpus"
_EXPECTED_QUESTION_COUNT = 10
_DATASET_VERSION = 1
_REPORT_VERSION = 1


class ResultIdentity(Protocol):
    @property
    def path(self) -> str: ...

    @property
    def symbol_name(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ExpectedTarget:
    path: str
    symbol: str | None


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    id: str
    category: str
    question: str
    expected_targets: tuple[ExpectedTarget, ...]


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    version: int
    name: str
    corpus_relative_path: str
    corpus_path: Path
    questions: tuple[EvaluationQuestion, ...]


@dataclass(frozen=True, slots=True)
class RankedIdentity:
    path: str
    symbol_name: str | None


@dataclass(frozen=True, slots=True)
class QuestionMetrics:
    id: str
    recall_at_k: dict[int, float]
    reciprocal_rank: float
    first_relevant_rank: int | None
    results: tuple[RankedIdentity, ...]


@dataclass(frozen=True, slots=True)
class RetrievalModeMetrics:
    recall_at_k: dict[int, float]
    mrr: float
    questions: tuple[QuestionMetrics, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingMetadata:
    provider: str
    model: str
    dimension: int
    sentence_transformers_version: str


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset: EvaluationDataset
    embedding: EmbeddingMetadata
    code_unit_count: int
    semantic: RetrievalModeMetrics
    hybrid: RetrievalModeMetrics


def target_matches_result(
    target: ExpectedTarget,
    result: ResultIdentity,
) -> bool:
    if target.path != result.path:
        return False
    return target.symbol is None or target.symbol == result.symbol_name


def recall_at_k(
    expected_targets: Sequence[ExpectedTarget],
    results: Sequence[RankedIdentity],
    k: int,
) -> float:
    if not expected_targets:
        raise ValueError("Expected targets must not be empty")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("K must be a positive integer")

    ranked_slice = results[:k]
    matched_count = sum(
        any(target_matches_result(target, result) for result in ranked_slice)
        for target in expected_targets
    )
    return matched_count / len(expected_targets)


def reciprocal_rank(
    expected_targets: Sequence[ExpectedTarget],
    results: Sequence[RankedIdentity],
) -> float:
    if not expected_targets:
        raise ValueError("Expected targets must not be empty")

    for rank, result in enumerate(results, start=1):
        if any(target_matches_result(target, result) for target in expected_targets):
            return 1.0 / rank
    return 0.0


def evaluate_question(
    question: EvaluationQuestion,
    results: Sequence[RankedIdentity],
) -> QuestionMetrics:
    ranked_results = tuple(results)
    rr = reciprocal_rank(question.expected_targets, ranked_results)
    first_relevant_rank = None if rr == 0.0 else round(1.0 / rr)
    return QuestionMetrics(
        id=question.id,
        recall_at_k={
            k: recall_at_k(question.expected_targets, ranked_results, k)
            for k in K_VALUES
        },
        reciprocal_rank=rr,
        first_relevant_rank=first_relevant_rank,
        results=ranked_results,
    )


def aggregate_question_metrics(
    questions: Sequence[QuestionMetrics],
) -> RetrievalModeMetrics:
    question_metrics = tuple(questions)
    if not question_metrics:
        raise ValueError("Question metrics must not be empty")

    return RetrievalModeMetrics(
        recall_at_k={
            k: sum(question.recall_at_k[k] for question in question_metrics)
            / len(question_metrics)
            for k in K_VALUES
        },
        mrr=sum(question.reciprocal_rank for question in question_metrics)
        / len(question_metrics),
        questions=question_metrics,
    )


def load_evaluation_dataset(
    path: Path = EVALUATION_DATASET_PATH,
) -> EvaluationDataset:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    document = _require_mapping(raw, "Evaluation dataset")
    if document.get("version") != _DATASET_VERSION:
        raise ValueError("Evaluation dataset version must be 1")

    repository = _require_mapping(document.get("repository"), "Repository metadata")
    if repository.get("type") != "committed_fixture":
        raise ValueError("Evaluation repository type must be committed_fixture")
    if repository.get("name") != "atlas-service":
        raise ValueError("Evaluation repository name must be atlas-service")
    corpus_relative_path = _require_string(repository.get("path"), "Corpus path")
    if corpus_relative_path != _EXPECTED_CORPUS_PATH:
        raise ValueError("Evaluation corpus path is unexpected")
    corpus_path = (_PROJECT_ROOT / corpus_relative_path).resolve()
    if not corpus_path.is_dir():
        raise ValueError("Evaluation corpus does not exist")

    question_values = _require_list(document.get("questions"), "Questions")
    if len(question_values) != _EXPECTED_QUESTION_COUNT:
        raise ValueError("Evaluation dataset must contain exactly 10 questions")
    questions = tuple(
        _load_question(value, index=index)
        for index, value in enumerate(question_values, start=1)
    )
    return EvaluationDataset(
        version=_DATASET_VERSION,
        name="atlas-service",
        corpus_relative_path=corpus_relative_path,
        corpus_path=corpus_path,
        questions=questions,
    )


def evaluate_retrieval(
    session: Session,
    *,
    dataset: EvaluationDataset,
    embedding_provider: EmbeddingProvider,
    embedding_metadata: EmbeddingMetadata,
) -> EvaluationReport:
    if embedding_provider.dimension != EMBEDDING_DIMENSION:
        raise RuntimeError("Embedding provider dimension does not match storage")
    if embedding_metadata.dimension != embedding_provider.dimension:
        raise RuntimeError("Embedding metadata dimension does not match provider")

    repository_token = uuid4().hex
    repository = Repository(
        name=f"atlas-service evaluation {repository_token}",
        source=f"fixture://atlas-service/{repository_token}",
    )
    session.add(repository)
    session.flush()

    # The public indexer requires a GitHub clone. Reusing its internal post-clone
    # phase keeps this committed local fixture aligned with production indexing.
    _index_cloned_repository(
        session=session,
        repository_id=repository.id,
        cloned_repository=ClonedRepository(
            source=repository.source,
            path=dataset.corpus_path,
        ),
        embedding_provider=embedding_provider,
    )

    file_count = session.scalar(
        select(func.count())
        .select_from(File)
        .where(File.repository_id == repository.id)
    )
    code_unit_count = session.scalar(
        select(func.count())
        .select_from(CodeUnit)
        .join(File, CodeUnit.file_id == File.id)
        .where(File.repository_id == repository.id)
    )
    if file_count != EXPECTED_FILE_COUNT:
        raise RuntimeError("Evaluation corpus did not produce exactly 7 Files")
    if code_unit_count != EXPECTED_CODE_UNIT_COUNT:
        raise RuntimeError("Evaluation corpus did not produce exactly 15 CodeUnits")

    query_vectors = embedding_provider.embed(
        [question.question for question in dataset.questions]
    )
    if len(query_vectors) != len(dataset.questions):
        raise RuntimeError("Embedding provider returned an unexpected query count")

    semantic_questions: list[QuestionMetrics] = []
    hybrid_questions: list[QuestionMetrics] = []
    for question, query_vector in zip(
        dataset.questions,
        query_vectors,
        strict=True,
    ):
        semantic_results = search_code_units_semantically(
            session,
            repository_id=repository.id,
            query_vector=query_vector,
            limit=RESULT_LIMIT,
            filters=None,
        )
        semantic_questions.append(
            evaluate_question(
                question,
                [
                    RankedIdentity(
                        path=result.path,
                        symbol_name=result.symbol_name,
                    )
                    for result in semantic_results
                ],
            )
        )

        hybrid_results = search_code_units_hybrid(
            session,
            repository_id=repository.id,
            query=question.question,
            query_vector=query_vector,
            limit=RESULT_LIMIT,
            filters=None,
        )
        hybrid_questions.append(
            evaluate_question(
                question,
                [
                    RankedIdentity(
                        path=result.path,
                        symbol_name=result.symbol_name,
                    )
                    for result in hybrid_results
                ],
            )
        )

    return EvaluationReport(
        dataset=dataset,
        embedding=embedding_metadata,
        code_unit_count=code_unit_count,
        semantic=aggregate_question_metrics(semantic_questions),
        hybrid=aggregate_question_metrics(hybrid_questions),
    )


def run_evaluation(
    *,
    embedding_provider: EmbeddingProvider,
    embedding_metadata: EmbeddingMetadata,
    database_url: str | None = None,
    dataset_path: Path = EVALUATION_DATASET_PATH,
) -> EvaluationReport:
    dataset = load_evaluation_dataset(dataset_path)
    engine = create_database_engine(database_url)
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                with Session(connection) as session:
                    report = evaluate_retrieval(
                        session,
                        dataset=dataset,
                        embedding_provider=embedding_provider,
                        embedding_metadata=embedding_metadata,
                    )
            finally:
                transaction.rollback()
    finally:
        engine.dispose()
    return report


def serialize_report(report: EvaluationReport) -> str:
    document = {
        "version": _REPORT_VERSION,
        "dataset": {
            "version": report.dataset.version,
            "name": report.dataset.name,
            "corpus_path": report.dataset.corpus_relative_path,
            "question_count": len(report.dataset.questions),
            "code_unit_count": report.code_unit_count,
        },
        "embedding": {
            "provider": report.embedding.provider,
            "model": report.embedding.model,
            "dimension": report.embedding.dimension,
            "sentence_transformers_version": (
                report.embedding.sentence_transformers_version
            ),
        },
        "evaluation": {
            "k_values": list(K_VALUES),
            "result_limit": RESULT_LIMIT,
            "mrr_depth": RESULT_LIMIT,
        },
        "hybrid": {
            "rrf_k": RRF_K,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
        },
        "modes": {
            "semantic": _serialize_mode(report.semantic),
            "hybrid": _serialize_mode(report.hybrid),
        },
    }
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _serialize_mode(mode: RetrievalModeMetrics) -> dict[str, object]:
    return {
        "recall_at_k": _serialize_recall(mode.recall_at_k),
        "mrr": round(mode.mrr, 6),
        "questions": [
            {
                "id": question.id,
                "recall_at_k": _serialize_recall(question.recall_at_k),
                "reciprocal_rank": round(question.reciprocal_rank, 6),
                "first_relevant_rank": question.first_relevant_rank,
                "results": [
                    {
                        "rank": rank,
                        "path": result.path,
                        "symbol": result.symbol_name,
                    }
                    for rank, result in enumerate(question.results, start=1)
                ],
            }
            for question in mode.questions
        ],
    }


def _serialize_recall(recall: dict[int, float]) -> dict[str, float]:
    return {str(k): round(recall[k], 6) for k in K_VALUES}


def _load_question(value: object, *, index: int) -> EvaluationQuestion:
    question = _require_mapping(value, f"Question {index}")
    question_id = _require_string(question.get("id"), f"Question {index} ID")
    category = _require_string(question.get("category"), f"Question {index} category")
    text = _require_string(question.get("question"), f"Question {index} text")
    targets = _require_list(
        question.get("expected_targets"),
        f"Question {index} expected targets",
    )
    if not targets:
        raise ValueError(f"Question {index} expected targets must not be empty")
    expected_targets = tuple(
        _load_target(target, question_index=index, target_index=target_index)
        for target_index, target in enumerate(targets, start=1)
    )
    return EvaluationQuestion(
        id=question_id,
        category=category,
        question=text,
        expected_targets=expected_targets,
    )


def _load_target(
    value: object,
    *,
    question_index: int,
    target_index: int,
) -> ExpectedTarget:
    target = _require_mapping(
        value,
        f"Question {question_index} target {target_index}",
    )
    path = _require_string(
        target.get("path"),
        f"Question {question_index} target {target_index} path",
    )
    symbol = target.get("symbol")
    if symbol is not None and (not isinstance(symbol, str) or not symbol):
        raise ValueError(
            f"Question {question_index} target {target_index} symbol is invalid"
        )
    return ExpectedTarget(path=path, symbol=symbol)


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return {str(key): nested for key, nested in value.items()}


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return list(value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _real_embedding_metadata() -> EmbeddingMetadata:
    return EmbeddingMetadata(
        provider=SentenceTransformerEmbeddingProvider.__name__,
        model=DEFAULT_MODEL_NAME,
        dimension=EMBEDDING_DIMENSION,
        sentence_transformers_version=package_version("sentence-transformers"),
    )


def _run_real_evaluation() -> EvaluationReport:
    provider = SentenceTransformerEmbeddingProvider()
    return run_evaluation(
        embedding_provider=provider,
        embedding_metadata=_real_embedding_metadata(),
    )


def _write_baseline(report: EvaluationReport) -> None:
    BASELINE_PATH.write_text(serialize_report(report), encoding="utf-8")


def _print_summary(report: EvaluationReport) -> None:
    for name, mode in (("semantic", report.semantic), ("hybrid", report.hybrid)):
        recall = " ".join(f"Recall@{k}={mode.recall_at_k[k]:.6f}" for k in K_VALUES)
        print(f"{name}: {recall} MRR={mode.mrr:.6f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RepoMind retrieval")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="write the deterministic retrieval baseline JSON",
    )
    arguments = parser.parse_args(argv)

    report = _run_real_evaluation()
    _print_summary(report)
    if arguments.write_baseline:
        _write_baseline(report)
        print(f"Wrote baseline: {BASELINE_PATH.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
