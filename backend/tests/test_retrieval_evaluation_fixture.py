import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal, TypedDict, cast

from app.code_parser import CodeParser, CodeUnitKind, ParsedCodeUnit
from app.javascript_typescript_code_parser import TypeScriptCodeParser
from app.python_code_parser import PythonCodeParser
from app.text_code_parser import (
    ConfigurationTextParser,
    DocumentationTextParser,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "retrieval_evaluation"
CORPUS_ROOT = FIXTURE_ROOT / "corpus"
EVALUATION_PATH = FIXTURE_ROOT / "evaluation.json"
PROJECT_ROOT = Path(__file__).parents[2]

EXPECTED_CORPUS_FILES = {
    "README.md",
    "config/settings.yaml",
    "src/api/routes.py",
    "src/auth/service.py",
    "src/auth/tokens.py",
    "src/database/session.py",
    "src/workers/retry.ts",
}
EXPECTED_SYMBOLS = {
    "README.md": ["Atlas Service", "Authentication", "Background jobs"],
    "config/settings.yaml": [None],
    "src/api/routes.py": [None, "login_endpoint", "register_routes"],
    "src/auth/service.py": [None, "authenticate_user", "login_user"],
    "src/auth/tokens.py": ["create_access_token", "decode_access_token"],
    "src/database/session.py": ["create_database_session"],
    "src/workers/retry.ts": ["RetryPolicy", "retryFailedJob"],
}
QUESTION_FIELDS = {"id", "category", "question", "expected_targets"}
TARGET_FIELDS = {"path", "symbol"}
ALLOWED_CATEGORIES = {"conceptual", "exact", "structural"}
FORBIDDEN_KEYS = {
    "code_unit_id",
    "file_id",
    "embedding",
    "vector",
    "cosine_distance",
    "lexical_score",
    "hybrid_score",
    "semantic_rank",
    "lexical_rank",
    "expected_rank",
    "expected_score",
    "recall",
    "mrr",
    "filters",
}


class ExpectedTarget(TypedDict):
    path: str
    symbol: str | None


class EvaluationQuestion(TypedDict):
    id: str
    category: Literal["conceptual", "exact", "structural"]
    question: str
    expected_targets: list[ExpectedTarget]


class RepositoryMetadata(TypedDict):
    type: str
    name: str
    path: str


class EvaluationFixture(TypedDict):
    version: int
    repository: RepositoryMetadata
    questions: list[EvaluationQuestion]


def _load_fixture() -> EvaluationFixture:
    return cast(
        EvaluationFixture,
        json.loads(EVALUATION_PATH.read_text(encoding="utf-8")),
    )


def _parser_for(relative_path: str) -> CodeParser:
    suffix = PurePosixPath(relative_path).suffix
    if suffix == ".py":
        return PythonCodeParser()
    if suffix == ".ts":
        return TypeScriptCodeParser()
    if suffix == ".md":
        return DocumentationTextParser()
    if suffix == ".yaml":
        return ConfigurationTextParser()
    raise AssertionError(f"No parser configured for fixture path: {relative_path}")


def _parse_corpus() -> dict[str, list[ParsedCodeUnit]]:
    parsed: dict[str, list[ParsedCodeUnit]] = {}
    for path in sorted(CORPUS_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(CORPUS_ROOT).as_posix()
        parsed[relative_path] = _parser_for(relative_path).parse(
            content=path.read_text(encoding="utf-8"),
            relative_path=relative_path,
        )
    return parsed


def _all_json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {key for key in value if isinstance(key, str)}
        for nested in value.values():
            keys.update(_all_json_keys(nested))
        return keys
    if isinstance(value, list):
        list_keys: set[str] = set()
        for nested in value:
            list_keys.update(_all_json_keys(nested))
        return list_keys
    return set()


def test_evaluation_fixture_structure() -> None:
    assert EVALUATION_PATH.is_file()
    fixture = _load_fixture()

    assert set(fixture) == {"version", "repository", "questions"}
    assert fixture["version"] == 1
    assert fixture["repository"] == {
        "type": "committed_fixture",
        "name": "atlas-service",
        "path": "backend/tests/fixtures/retrieval_evaluation/corpus",
    }
    assert (PROJECT_ROOT / fixture["repository"]["path"]).resolve() == (
        CORPUS_ROOT.resolve()
    )

    questions = fixture["questions"]
    assert len(questions) == 10
    ids = [question["id"] for question in questions]
    assert all(ids)
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", item) for item in ids)
    assert Counter(question["category"] for question in questions) == {
        "conceptual": 4,
        "exact": 3,
        "structural": 3,
    }

    for question in questions:
        assert set(question) == QUESTION_FIELDS
        assert question["category"] in ALLOWED_CATEGORIES
        assert question["question"].strip()
        assert question["expected_targets"]


def test_expected_targets_are_valid_unique_corpus_paths() -> None:
    fixture = _load_fixture()
    corpus_files = {
        path.relative_to(CORPUS_ROOT).as_posix()
        for path in CORPUS_ROOT.rglob("*")
        if path.is_file()
    }
    assert corpus_files == EXPECTED_CORPUS_FILES

    for question in fixture["questions"]:
        targets = question["expected_targets"]
        target_pairs: list[tuple[str, str | None]] = []
        for target in targets:
            assert set(target) == TARGET_FIELDS
            path = target["path"]
            symbol = target["symbol"]
            assert isinstance(path, str) and path
            assert not path.startswith("/")
            assert "\\" not in path
            assert all(part not in {"", ".", ".."} for part in path.split("/"))
            assert PurePosixPath(path).as_posix() == path
            assert path in corpus_files
            assert symbol is None or (isinstance(symbol, str) and symbol.strip())
            target_pairs.append((path, symbol))
        assert len(target_pairs) == len(set(target_pairs))


def test_fixture_contains_no_evaluation_output_fields() -> None:
    raw_fixture: object = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))

    assert _all_json_keys(raw_fixture).isdisjoint(FORBIDDEN_KEYS)


def test_expected_targets_match_current_parser_output() -> None:
    fixture = _load_fixture()
    parsed = _parse_corpus()

    assert {
        path: [unit.symbol_name for unit in units] for path, units in parsed.items()
    } == EXPECTED_SYMBOLS

    emitted_targets = {
        (path, unit.symbol_name) for path, units in parsed.items() for unit in units
    }
    for question in fixture["questions"]:
        for target in question["expected_targets"]:
            if target["symbol"] is not None:
                assert (target["path"], target["symbol"]) in emitted_targets


def test_symbol_less_config_target_is_unambiguous() -> None:
    fixture = _load_fixture()
    units = _parse_corpus()["config/settings.yaml"]

    assert len(units) == 1
    assert units[0].kind is CodeUnitKind.CONFIG
    assert units[0].symbol_name is None
    assert units[0].relative_path == "config/settings.yaml"
    assert [
        target
        for question in fixture["questions"]
        for target in question["expected_targets"]
        if target["symbol"] is None
    ] == [{"path": "config/settings.yaml", "symbol": None}]
