import os
import runpy
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from app import (
    ask_api,
    cross_encoder_reranking_provider,
    ollama_llm_provider,
    search_api,
)
from app.database import create_database_engine, get_db_session
from app.llm_provider import (
    LLMMessage,
    LLMProviderError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.main import app
from app.mock_llm_provider import MockLLMProvider
from app.models.code_unit import EMBEDDING_DIMENSION, CodeUnit
from app.models.file import File
from app.models.repository import Repository
from app.rag_context import ContextEvidence
from app.reranking_provider import RerankerUnavailableError

DATABASE_URL = os.getenv("DATABASE_URL")


def _vector() -> list[float]:
    return [1.0, *([0.0] * (EMBEDDING_DIMENSION - 1))]


class FakeEmbeddingProvider:
    dimension = EMBEDDING_DIMENSION

    def __init__(self) -> None:
        self.output = [_vector()]
        self.failure: Exception | None = None
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        if self.failure is not None:
            raise self.failure
        return self.output


class FakeReranker:
    model_name = "fake"

    def __init__(self) -> None:
        self.failure: Exception | None = None
        self.output: list[float] | None = None
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, tuple(documents)))
        if self.failure is not None:
            raise self.failure
        if self.output is not None:
            return self.output
        return [float(index) for index in range(len(documents))]


@dataclass
class ApiContext:
    client: TestClient
    session: Session
    embedding: FakeEmbeddingProvider
    reranker: FakeReranker
    llm: MockLLMProvider


@pytest.fixture
def api_context(monkeypatch: pytest.MonkeyPatch) -> Iterator[ApiContext]:
    if DATABASE_URL is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(DATABASE_URL)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(connection) as session:
                embedding = FakeEmbeddingProvider()
                reranker = FakeReranker()
                llm = MockLLMProvider(response="  Answer ✓ [Evidence 1]\n")

                def override_session() -> Iterator[Session]:
                    yield session

                app.dependency_overrides[get_db_session] = override_session
                monkeypatch.setattr(
                    ask_api, "get_embedding_provider", lambda: embedding
                )
                app.dependency_overrides[search_api.get_embedding_provider] = lambda: (
                    embedding
                )
                app.dependency_overrides[ask_api.get_reranking_provider] = lambda: (
                    reranker
                )
                app.dependency_overrides[ask_api.get_llm_provider] = lambda: llm
                with TestClient(app, raise_server_exceptions=False) as client:
                    yield ApiContext(client, session, embedding, reranker, llm)
        finally:
            app.dependency_overrides.clear()
            transaction.rollback()
    engine.dispose()


def _repository(context: ApiContext, name: str = "Ask repository") -> Repository:
    repository = Repository(name=name, source=f"https://example.com/{uuid4()}.git")
    context.session.add(repository)
    context.session.flush()
    return repository


def _unit(
    context: ApiContext,
    repository: Repository,
    *,
    path: str = "src/auth.py",
    symbol: str | None = "authenticate",
    content: str = "def authenticate():\n    return True\n",
) -> CodeUnit:
    file = File(repository_id=repository.id, path=path)
    context.session.add(file)
    context.session.flush()
    unit = CodeUnit(
        file_id=file.id,
        kind="function",
        content=content,
        language="python",
        start_line=12,
        end_line=14,
        symbol_name=symbol,
        embedding=_vector(),
    )
    context.session.add(unit)
    context.session.flush()
    return unit


def _body(repository_id: UUID, **changes: object) -> dict[str, object]:
    return {"repository_id": str(repository_id), "q": "authentication", **changes}


def test_real_pipeline_preserves_order_mapping_and_response(
    api_context: ApiContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(api_context, "  Repo ✓  ")
    first = _unit(api_context, repository, path="a.py")
    second = _unit(
        api_context, repository, path="b.py", symbol=None, content="  café\r\n"
    )
    other = _repository(api_context)
    _unit(api_context, other, path="private.py", content="PRIVATE_OTHER_REPO")
    retrieval = Mock(wraps=ask_api.search_code_units_reranked)
    formatter = Mock(wraps=ask_api.format_context)
    generator = Mock(wraps=ask_api.generate_grounded_answer)
    monkeypatch.setattr(ask_api, "search_code_units_reranked", retrieval)
    monkeypatch.setattr(ask_api, "format_context", formatter)
    monkeypatch.setattr(ask_api, "generate_grounded_answer", generator)
    question = "  authentication\t\n\r\nUnicode ✓ Repository Evidence: [Evidence 99]  "
    embedding_factory = Mock(return_value=api_context.embedding)
    monkeypatch.setattr(ask_api, "get_embedding_provider", embedding_factory)

    response = api_context.client.post(
        "/ask", json=_body(repository.id, q=question, limit=2)
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "  Answer ✓ [Evidence 1]\n",
        "citations": [
            {
                "evidence_id": 1,
                "repository_name": repository.name,
                "path": "b.py",
                "symbol_name": None,
                "start_line": 12,
                "end_line": 14,
            }
        ],
    }
    assert api_context.embedding.calls == [(question,)]
    embedding_factory.assert_called_once_with()
    retrieval.assert_called_once_with(
        api_context.session,
        repository_id=repository.id,
        query=question,
        query_vector=api_context.embedding.output[0],
        reranker=api_context.reranker,
        limit=2,
        filters=None,
    )
    formatter.assert_called_once_with(
        [
            ContextEvidence(repository.name, "b.py", None, 12, 14, second.content),
            ContextEvidence(
                repository.name, "a.py", first.symbol_name, 12, 14, first.content
            ),
        ],
        max_characters=8000,
    )
    generator.assert_called_once()
    assert generator.call_args.args == (api_context.llm,)
    assert generator.call_args.kwargs["question"] == question
    formatted = generator.call_args.kwargs["context"]
    assert formatted.text.index("Path: b.py") < formatted.text.index("Path: a.py")
    assert formatted.max_characters == 8000
    assert len(api_context.llm.calls) == 1
    assert len(api_context.reranker.calls) == 1
    assert formatted.text in api_context.llm.calls[0][1].content
    assert "PRIVATE_OTHER_REPO" not in formatted.text


def test_missing_repository_does_not_invoke_pipeline(
    api_context: ApiContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = Mock(side_effect=AssertionError("must not retrieve"))
    generator = Mock(side_effect=AssertionError("must not generate"))
    monkeypatch.setattr(ask_api, "search_code_units_reranked", retrieval)
    monkeypatch.setattr(ask_api, "generate_grounded_answer", generator)
    response = api_context.client.post("/ask", json=_body(uuid4()))
    assert response.status_code == 404
    assert response.json() == {"detail": "Repository not found"}
    assert api_context.embedding.calls == []
    assert api_context.reranker.calls == []
    assert api_context.llm.calls == ()
    retrieval.assert_not_called()
    generator.assert_not_called()


@pytest.mark.parametrize("question, status", [("valid question", 404), ("", 422)])
def test_rejected_request_does_not_acquire_embedding_provider(
    api_context: ApiContext,
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    status: int,
) -> None:
    factory = Mock(side_effect=AssertionError("EMBEDDING_MUST_NOT_BE_ACQUIRED"))
    monkeypatch.setattr(ask_api, "get_embedding_provider", factory)
    response = api_context.client.post("/ask", json=_body(uuid4(), q=question))
    assert response.status_code == status
    if status == 404:
        assert response.json() == {"detail": "Repository not found"}
    factory.assert_not_called()
    assert api_context.embedding.calls == []


def test_empty_retrieval_uses_formatter_and_generator_without_llm(
    api_context: ApiContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(api_context)
    formatter = Mock(wraps=ask_api.format_context)
    generator = Mock(wraps=ask_api.generate_grounded_answer)
    monkeypatch.setattr(ask_api, "format_context", formatter)
    monkeypatch.setattr(ask_api, "generate_grounded_answer", generator)
    response = api_context.client.post("/ask", json=_body(repository.id))
    assert response.status_code == 200
    assert response.json() == {
        "answer": "The available evidence is insufficient to answer the question.",
        "citations": [],
    }
    formatter.assert_called_once_with([], max_characters=8000)
    generator.assert_called_once()
    assert api_context.llm.calls == ()
    assert api_context.reranker.calls == []


@pytest.mark.parametrize("answer", ["", "   "])
def test_empty_llm_answer_is_not_reinterpreted(
    api_context: ApiContext, answer: str
) -> None:
    repository = _repository(api_context)
    _unit(api_context, repository)
    provider = MockLLMProvider(response=answer)
    app.dependency_overrides[ask_api.get_llm_provider] = lambda: provider
    response = api_context.client.post("/ask", json=_body(repository.id))
    assert response.status_code == 200
    assert response.json() == {"answer": answer, "citations": []}
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    "q",
    [
        None,
        1,
        True,
        [],
        {},
        "",
        " \t\r\n",
        "x" * 2001,
        *[f"x{chr(code)}y" for code in range(32) if code not in (9, 10, 13)],
        "x\x7fy",
    ],
)
def test_invalid_questions_fail_without_inference(
    api_context: ApiContext, q: object
) -> None:
    response = api_context.client.post("/ask", json=_body(uuid4(), q=q))
    assert response.status_code == 422
    assert api_context.embedding.calls == []
    assert api_context.reranker.calls == []
    assert api_context.llm.calls == ()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"q": "valid"},
        {"repository_id": str(uuid4())},
        {"repository_id": "bad", "q": "valid"},
    ],
)
def test_missing_fields_and_malformed_uuid(
    api_context: ApiContext, body: dict[str, object]
) -> None:
    assert api_context.client.post("/ask", json=body).status_code == 422
    assert api_context.embedding.calls == []


@pytest.mark.parametrize("limit", [True, False, 1.0, "1", 0, -1, 101, None])
def test_invalid_limit(api_context: ApiContext, limit: object) -> None:
    assert (
        api_context.client.post("/ask", json=_body(uuid4(), limit=limit)).status_code
        == 422
    )
    assert api_context.embedding.calls == []


@pytest.mark.parametrize("limit", [None, 1, 3, 100])
def test_default_explicit_and_boundary_limits(
    api_context: ApiContext,
    monkeypatch: pytest.MonkeyPatch,
    limit: int | None,
) -> None:
    repository = _repository(api_context)
    retrieval = Mock(wraps=ask_api.search_code_units_reranked)
    monkeypatch.setattr(ask_api, "search_code_units_reranked", retrieval)
    body = _body(repository.id)
    if limit is not None:
        body["limit"] = limit
    assert api_context.client.post("/ask", json=body).status_code == 200
    assert retrieval.call_args.kwargs["limit"] == (10 if limit is None else limit)
    assert retrieval.call_args.kwargs["filters"] is None


@pytest.mark.parametrize(
    "field",
    [
        "filters",
        "model",
        "provider",
        "base_url",
        "timeout",
        "context",
        "evidence",
        "prompt",
        "path",
        "source",
        "citations",
    ],
)
def test_extra_fields_forbidden(api_context: ApiContext, field: str) -> None:
    body = _body(uuid4())
    body[field] = "untrusted"
    assert api_context.client.post("/ask", json=body).status_code == 422
    assert api_context.embedding.calls == []


@pytest.mark.parametrize(
    "output",
    [[], [_vector(), _vector()], [[1.0]], [[0.0] * 384], [[float("nan")] * 384]],
)
def test_bad_embedding_output_is_500(
    api_context: ApiContext, output: list[list[float]]
) -> None:
    repository = _repository(api_context)
    api_context.embedding.output = output
    response = api_context.client.post("/ask", json=_body(repository.id))
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert api_context.llm.calls == ()


def test_embedding_dimension_mismatch_before_embed(api_context: ApiContext) -> None:
    repository = _repository(api_context)
    api_context.embedding.dimension = 1
    response = api_context.client.post("/ask", json=_body(repository.id))
    assert response.status_code == 500
    assert api_context.embedding.calls == []


@pytest.mark.parametrize("construction", [False, True])
def test_embedding_failure_is_sanitized(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch, construction: bool
) -> None:
    repository = _repository(api_context)
    failure = RuntimeError("SECRET_EMBEDDING")
    if construction:

        def fail() -> FakeEmbeddingProvider:
            raise failure

        monkeypatch.setattr(ask_api, "get_embedding_provider", fail)
    else:
        api_context.embedding.failure = failure
    response = api_context.client.post("/ask", json=_body(repository.id))
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert "SECRET_EMBEDDING" not in response.text


def test_existing_reranker_fallback_preserves_hybrid_order(
    api_context: ApiContext,
) -> None:
    repository = _repository(api_context)
    _unit(api_context, repository, path="a.py")
    _unit(api_context, repository, path="b.py")
    api_context.reranker.failure = RerankerUnavailableError("not installed")
    response = api_context.client.post("/ask", json=_body(repository.id, limit=2))
    assert response.status_code == 200
    prompt = api_context.llm.calls[0][1].content
    assert prompt.index("Path: a.py") < prompt.index("Path: b.py")
    assert len(api_context.reranker.calls) == 1


@pytest.mark.parametrize("malformed", [False, True])
def test_reranker_defects_do_not_fallback(
    api_context: ApiContext, malformed: bool
) -> None:
    repository = _repository(api_context)
    _unit(api_context, repository)
    if malformed:
        api_context.reranker.output = []
    else:
        api_context.reranker.failure = RuntimeError("SECRET_RERANKER")
    response = api_context.client.post("/ask", json=_body(repository.id))
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert api_context.llm.calls == ()


@pytest.mark.parametrize(
    ("error_type", "status", "detail"),
    [
        (LLMUnavailableError, 503, "Language model service unavailable"),
        (LLMTimeoutError, 504, "Language model request timed out"),
        (LLMResponseError, 502, "Language model returned an invalid response"),
        (LLMProviderError, 502, "Language model request failed"),
        (RuntimeError, 500, None),
    ],
)
def test_llm_error_mapping_is_sanitized(
    api_context: ApiContext,
    error_type: type[Exception],
    status: int,
    detail: str | None,
) -> None:
    class FailingProvider:
        model_name = "fake"

        def chat(self, messages: Sequence[LLMMessage]) -> LLMResponse:
            raise error_type(
                "SECRET_EXCEPTION http://private-model/ SECRET_CONTEXT SECRET_QUESTION"
            )

    repository = _repository(api_context)
    _unit(api_context, repository, content="SECRET_CONTEXT")
    app.dependency_overrides[ask_api.get_llm_provider] = FailingProvider
    response = api_context.client.post(
        "/ask", json=_body(repository.id, q="SECRET_QUESTION")
    )
    assert response.status_code == status
    if detail is None:
        assert response.text == "Internal Server Error"
    else:
        assert response.json() == {"detail": detail}
    for secret in (
        "SECRET_EXCEPTION",
        "SECRET_CONTEXT",
        "SECRET_QUESTION",
        "private-model",
    ):
        assert secret not in response.text


def test_request_does_not_write_or_commit(api_context: ApiContext) -> None:
    repository = _repository(api_context)
    unit = _unit(api_context, repository)
    original = unit.content

    def reject_write(session: Session, *args: object) -> None:
        raise AssertionError("Endpoint must not flush or commit")

    event.listen(api_context.session, "before_flush", reject_write)
    event.listen(api_context.session, "before_commit", reject_write)
    try:
        response = api_context.client.post("/ask", json=_body(repository.id))
        assert response.status_code == 200
        assert not api_context.session.new
        assert not api_context.session.dirty
        assert not api_context.session.deleted
        assert unit.content == original
    finally:
        event.remove(api_context.session, "before_flush", reject_write)
        event.remove(api_context.session, "before_commit", reject_write)


def test_context_is_bounded(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(api_context)
    _unit(api_context, repository, content="x" * 12000)
    generator = Mock(wraps=ask_api.generate_grounded_answer)
    monkeypatch.setattr(ask_api, "generate_grounded_answer", generator)
    assert api_context.client.post("/ask", json=_body(repository.id)).status_code == 200
    context = generator.call_args.kwargs["context"]
    assert len(context.text) <= 8000
    assert context.truncated is True


def test_factories_cached_without_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = Mock(return_value=FakeReranker())
    llm = Mock(return_value=MockLLMProvider(response="fake"))
    monkeypatch.setattr(
        cross_encoder_reranking_provider, "CrossEncoderRerankingProvider", reranker
    )
    monkeypatch.setattr(ollama_llm_provider, "OllamaLLMProvider", llm)
    ask_api.get_reranking_provider.cache_clear()
    ask_api.get_llm_provider.cache_clear()
    try:
        assert ask_api.get_reranking_provider() is ask_api.get_reranking_provider()
        assert ask_api.get_llm_provider() is ask_api.get_llm_provider()
        reranker.assert_called_once_with()
        llm.assert_called_once_with(model_name="qwen2.5-coder:3b")
        assert ask_api.get_embedding_provider is search_api.get_embedding_provider
    finally:
        ask_api.get_reranking_provider.cache_clear()
        ask_api.get_llm_provider.cache_clear()


def test_import_does_not_construct_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = Mock(side_effect=AssertionError("model construction at import"))
    llm = Mock(side_effect=AssertionError("provider construction at import"))
    embedding = Mock(side_effect=AssertionError("embedding construction at import"))
    monkeypatch.setattr(
        cross_encoder_reranking_provider, "CrossEncoderRerankingProvider", reranker
    )
    monkeypatch.setattr(ollama_llm_provider, "OllamaLLMProvider", llm)
    monkeypatch.setattr(search_api, "get_embedding_provider", embedding)
    runpy.run_path(str(Path(ask_api.__file__)))
    reranker.assert_not_called()
    llm.assert_not_called()
    embedding.assert_not_called()


def test_search_still_semantic_only(api_context: ApiContext) -> None:
    repository = _repository(api_context)
    _unit(api_context, repository)
    response = api_context.client.get(
        "/search", params={"repository_id": str(repository.id), "q": "auth"}
    )
    assert response.status_code == 200
    assert set(response.json()) == {"items", "limit"}
    assert "cosine_distance" in response.json()["items"][0]
    assert api_context.reranker.calls == []
    assert api_context.llm.calls == ()


@pytest.mark.parametrize(
    ("answer", "ids"),
    [
        ("  [Evidence 2][Evidence 1] [Evidence 2]\n", [2, 1]),
        ("[Evidence 99]", []),
        ("[Evidence 99] [Evidence 1]", [1]),
        ("no markers", []),
        ("\n", []),
        ("invented.py lines 900-999 [Evidence 1]", [1]),
        ("[Evidence 01]", []),
    ],
)
def test_api_citation_mapping(
    api_context: ApiContext, answer: str, ids: list[int]
) -> None:
    repository = _repository(api_context)
    _unit(api_context, repository, path="a.py")
    _unit(api_context, repository, path="b.py", symbol=None)
    provider = MockLLMProvider(response=answer)
    app.dependency_overrides[ask_api.get_llm_provider] = lambda: provider
    response = api_context.client.post("/ask", json=_body(repository.id, limit=2))
    assert response.status_code == 200
    assert response.json() == {
        "answer": answer,
        "citations": [
            {
                "evidence_id": index,
                "repository_name": repository.name,
                "path": "b.py" if index == 1 else "a.py",
                "symbol_name": None if index == 1 else "authenticate",
                "start_line": 12,
                "end_line": 14,
            }
            for index in ids
        ],
    }
    assert len(provider.calls) == 1
    assert len(api_context.embedding.calls) == 1
    assert len(api_context.reranker.calls) == 1


def test_api_only_maps_included_truncated_prefix(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(api_context)
    _unit(api_context, repository, path="a.py")
    _unit(api_context, repository, path="b.py", content="x" * 12000)
    provider = MockLLMProvider(response="[Evidence 2] [Evidence 1]")
    app.dependency_overrides[ask_api.get_llm_provider] = lambda: provider
    formatter = Mock(wraps=ask_api.format_context)
    generator = Mock(wraps=ask_api.generate_grounded_answer)
    monkeypatch.setattr(ask_api, "format_context", formatter)
    monkeypatch.setattr(ask_api, "generate_grounded_answer", generator)
    response = api_context.client.post("/ask", json=_body(repository.id, limit=2))
    assert response.status_code == 200
    context = generator.call_args.kwargs["context"]
    assert context.truncated is True
    assert context.included_evidence_count == 1
    assert "[TRUNCATED]" in context.text
    assert response.json() == {
        "answer": "[Evidence 2] [Evidence 1]",
        "citations": [
            {
                "evidence_id": 1,
                "repository_name": repository.name,
                "path": "b.py",
                "symbol_name": "authenticate",
                "start_line": 12,
                "end_line": 14,
            }
        ],
    }
    formatter.assert_called_once()


def test_citation_failure_is_not_translated_as_provider_failure(
    api_context: ApiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(api_context)
    _unit(api_context, repository)
    monkeypatch.setattr(
        ask_api, "extract_citations", Mock(side_effect=LLMProviderError("defect"))
    )
    response = api_context.client.post("/ask", json=_body(repository.id))
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
