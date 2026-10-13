import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_client.parser import text_string_to_metric_families
from starlette.types import Message

from app import ask_api, indexing_api, indexing_jobs, search_api
from app import metrics as telemetry
from app.grounded_answer import generate_grounded_answer
from app.llm_provider import (
    LLMProviderError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.mock_llm_provider import MockLLMProvider
from app.rag_context import FormattedContext


@pytest.fixture
def metrics() -> Iterator[telemetry.Metrics]:
    value = telemetry.Metrics()
    with telemetry.use_metrics(value):
        yield value


def sample(metrics: telemetry.Metrics, name: str, **labels: str) -> float | None:
    return metrics.registry.get_sample_value("repomind_" + name, labels)


def test_scrape_is_isolated_idempotent_and_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import app as production_app

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    first, second = FastAPI(), FastAPI()
    one, two = telemetry.install_metrics(first), telemetry.install_metrics(second)
    assert telemetry.install_metrics(first) is one
    assert len([r for r in first.routes if getattr(r, "path", None) == "/metrics"]) == 1
    assert one.registry is not two.registry
    for app in (first, second, production_app):
        with TestClient(app) as client:
            response = client.get("/metrics")
            assert response.status_code == 200
            assert response.headers["content-type"] == CONTENT_TYPE_LATEST
            families = list(text_string_to_metric_families(response.text))
            # No global-registry process/GC collectors or external service probes.
            names = {f.name for f in families if not f.name.endswith("_created")}
            assert names == {
                "repomind_http_requests",
                "repomind_http_request_duration_seconds",
                "repomind_indexing_executions",
                "repomind_indexing_duration_seconds",
                "repomind_indexing_files_hashed",
                "repomind_indexing_code_units_parsed",
                "repomind_retrieval_duration_seconds",
                "repomind_llm_request_duration_seconds",
            }
    with TestClient(first) as client:
        client.get("/private-unknown")
        before = generate_latest(one.registry)
        assert client.get("/metrics").content == before
        assert client.get("/metrics").content == before
    assert (
        sample(
            one,
            "http_requests_total",
            method="GET",
            route="unmatched",
            status_class="4xx",
        )
        == 1
    )
    assert (
        sample(
            two,
            "http_requests_total",
            method="GET",
            route="unmatched",
            status_class="4xx",
        )
        is None
    )


@pytest.mark.parametrize("status", [200, 202, 302, 404, 422, 500, 503])
def test_http_status_template_and_duration(status: int) -> None:
    app = FastAPI()

    @app.get("/repositories/{repository_id}")
    def route(repository_id: UUID) -> None:
        raise HTTPException(status, "PRIVATE exception content")

    metrics = telemetry.install_metrics(app)
    rid = uuid4()
    with TestClient(app) as client:
        assert (
            client.get(
                f"/repositories/{rid}?q=PRIVATE", follow_redirects=False
            ).status_code
            == status
        )
    labels = dict(
        method="GET",
        route="/repositories/{repository_id}",
        status_class=f"{status // 100}xx",
    )
    assert sample(metrics, "http_requests_total", **labels) == 1
    assert sample(metrics, "http_request_duration_seconds_count", **labels) == 1
    output = generate_latest(metrics.registry).decode()
    assert str(rid) not in output and "PRIVATE" not in output


@pytest.mark.parametrize(
    ("method", "path", "expected_route", "expected_method"),
    [
        ("GET", "/repositories/not-a-uuid", "/repositories/{repository_id}", "GET"),
        ("POST", "/repositories/secret", "/repositories/{repository_id}", "POST"),
        ("SECRET", "/unknown-sensitive", "unmatched", "OTHER"),
    ],
)
def test_invalid_unmatched_and_method_labels(
    method: str, path: str, expected_route: str, expected_method: str
) -> None:
    app = FastAPI()

    @app.get("/repositories/{repository_id}")
    def route(repository_id: UUID) -> None:
        pass

    metrics = telemetry.install_metrics(app)
    with TestClient(app) as client:
        assert 400 <= client.request(method, path).status_code < 500
    assert (
        sample(
            metrics,
            "http_requests_total",
            method=expected_method,
            route=expected_route,
            status_class="4xx",
        )
        == 1
    )
    output = generate_latest(metrics.registry).decode()
    assert "secret" not in output and "SECRET" not in output


def test_unhandled_exception_preserved_and_counted_once() -> None:
    app = FastAPI()
    error = RuntimeError("private source")

    @app.get("/broken")
    async def broken() -> None:
        raise error

    metrics = telemetry.install_metrics(app)
    with TestClient(app) as client, pytest.raises(RuntimeError) as caught:
        client.get("/broken")
    assert caught.value is error
    assert (
        sample(
            metrics,
            "http_requests_total",
            method="GET",
            route="/broken",
            status_class="5xx",
        )
        == 1
    )


@pytest.mark.parametrize(
    "finish",
    ["cancel", "incomplete", "error", "complete_then_error", "complete_then_cancel"],
)
def test_asgi_interruption_and_streaming_not_buffered(finish: str) -> None:
    owner = FastAPI()
    metrics = telemetry.install_metrics(owner)
    sent: list[Message] = []
    error: BaseException = (
        asyncio.CancelledError() if "cancel" in finish else RuntimeError("private")
    )

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        body = {"type": "http.response.body", "body": b"first", "more_body": True}
        await send(body)
        assert sent[-1] is body  # forwarded immediately, without copying/buffering
        if finish.startswith("complete"):
            await send({"type": "http.response.body", "body": b"last"})
        if finish != "incomplete":
            raise error

    async def send(message: Message) -> None:
        sent.append(message)

    async def receive() -> Message:
        return {"type": "http.request"}

    middleware = telemetry.MetricsMiddleware(app, owner=owner, metrics=metrics)
    operation = middleware({"type": "http", "method": "GET"}, receive, send)
    if finish == "incomplete":
        asyncio.run(operation)
    else:
        with pytest.raises(type(error)) as caught:
            asyncio.run(operation)
        assert caught.value is error
    outcome = (
        "2xx"
        if finish.startswith("complete")
        else "cancelled"
        if finish == "cancel"
        else "incomplete"
    )
    assert (
        sample(
            metrics,
            "http_requests_total",
            method="GET",
            route="unmatched",
            status_class=outcome,
        )
        == 1
    )
    assert (
        sum(
            s.value
            for f in metrics.registry.collect()
            for s in f.samples
            if s.name == "repomind_http_requests_total"
        )
        == 1
    )


def test_registry_propagates_async_sync_and_real_background_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.include_router(indexing_api.router)
    metrics = telemetry.install_metrics(app)
    clock = [0.0]
    monkeypatch.setattr(telemetry, "perf_counter", lambda: clock[0])
    seen: list[str] = []
    rid, jid = uuid4(), uuid4()
    session = Mock()
    session.commit.side_effect = lambda: seen.append("commit")
    app.dependency_overrides[indexing_api.get_indexing_session] = lambda: session
    app.dependency_overrides[indexing_api.get_redis_cache] = lambda: None
    monkeypatch.setattr(
        indexing_api, "reserve_indexing_job", lambda *args: SimpleNamespace(id=jid)
    )

    @contextmanager
    def factory() -> Iterator[Any]:
        yield session

    monkeypatch.setattr(
        indexing_jobs, "get_application_session_factory", lambda: factory
    )

    def claim(*args: Any) -> UUID:
        assert telemetry.get_metrics() is metrics
        assert args[1] == jid
        # The production ASGI response has already sent its final body.
        assert (
            sample(
                metrics,
                "http_requests_total",
                method="POST",
                route="/repositories/{repository_id}/index",
                status_class="2xx",
            )
            == 1
        )
        seen.append("claim")
        clock[0] = 10
        return rid

    monkeypatch.setattr(indexing_jobs, "_claim_job", claim)
    monkeypatch.setattr(
        indexing_jobs, "get_indexing_embedding_provider", lambda: object()
    )

    def execute(*args: Any, **kwargs: Any) -> None:
        assert telemetry.get_metrics() is metrics
        assert kwargs["job_id"] == jid
        assert kwargs["repository_id"] == rid
        telemetry.file_hashed()
        clock[0] = 110
        seen.append("execute")

    monkeypatch.setattr(indexing_jobs, "execute_reserved_indexing_job", execute)

    @app.get("/async")
    async def async_route() -> None:
        assert telemetry.get_metrics() is metrics
        telemetry.units_parsed(2)

    @app.get("/sync")
    def sync_route() -> None:
        assert telemetry.get_metrics() is metrics
        telemetry.units_parsed(3)

    with TestClient(app) as client:
        assert client.get("/async").status_code == 200
        assert client.get("/sync").status_code == 200
        assert client.post(f"/repositories/{rid}/index").status_code == 202
    assert seen == ["commit", "claim", "execute", "commit"]
    assert (
        sample(
            metrics, "indexing_executions_total", mode="background", outcome="success"
        )
        == 1
    )
    assert (
        sample(
            metrics,
            "indexing_duration_seconds_sum",
            mode="background",
            outcome="success",
        )
        == 100
    )
    assert (
        sample(
            metrics,
            "http_request_duration_seconds_sum",
            method="POST",
            route="/repositories/{repository_id}/index",
            status_class="2xx",
        )
        == 0
    )
    assert sample(metrics, "indexing_files_hashed_total") == 1
    assert sample(metrics, "indexing_code_units_parsed_total") == 5


@pytest.mark.parametrize(
    "error",
    [
        None,
        LLMTimeoutError("secret"),
        LLMUnavailableError("secret"),
        LLMResponseError("secret"),
        LLMProviderError("secret"),
        RuntimeError("secret"),
        asyncio.CancelledError(),
    ],
)
def test_llm_actual_call_outcomes_and_identity(
    metrics: telemetry.Metrics, error: BaseException | None
) -> None:
    response = LLMResponse(content="  PRIVATE answer  ")
    provider = Mock()
    provider.chat.return_value = response
    provider.chat.side_effect = error
    context = FormattedContext(
        text="source",
        included_evidence_count=1,
        truncated=False,
        used_characters=6,
        max_characters=100,
    )
    if error is None:
        assert (
            generate_grounded_answer(provider, question="question", context=context)
            is response
        )
        outcome = "success"
    else:
        with pytest.raises(type(error)) as caught:
            generate_grounded_answer(provider, question="question", context=context)
        assert caught.value is error
        outcome = {
            LLMTimeoutError: "timeout",
            LLMUnavailableError: "unavailable",
            LLMResponseError: "response_error",
            asyncio.CancelledError: "cancelled",
        }.get(type(error), "error")
    provider.chat.assert_called_once()
    assert sample(metrics, "llm_request_duration_seconds_count", outcome=outcome) == 1
    assert "PRIVATE" not in generate_latest(metrics.registry).decode()
    assert "secret" not in generate_latest(metrics.registry).decode()


def test_empty_context_never_observes_llm(metrics: telemetry.Metrics) -> None:
    provider = MockLLMProvider(response="unused")
    context = FormattedContext(
        text="",
        included_evidence_count=0,
        truncated=True,
        used_characters=0,
        max_characters=100,
    )
    response = generate_grounded_answer(provider, question="question", context=context)
    assert (
        response.content
        == "The available evidence is insufficient to answer the question."
    )
    assert provider.calls == ()
    assert (
        sample(metrics, "llm_request_duration_seconds_count", outcome="success") is None
    )


@pytest.mark.parametrize("pipeline", ["search", "ask"])
@pytest.mark.parametrize("failure", [None, "provider", "embed", "retrieve", "cancel"])
def test_real_retrieval_boundaries(
    metrics: telemetry.Metrics,
    monkeypatch: pytest.MonkeyPatch,
    pipeline: str,
    failure: str | None,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(telemetry, "perf_counter", lambda: clock[0])
    error: BaseException = (
        asyncio.CancelledError() if failure == "cancel" else RuntimeError("PRIVATE")
    )
    session = Mock()
    session.scalar.return_value = 0
    session.get.return_value = SimpleNamespace(name="private-repo")
    provider = Mock(dimension=384)

    def factory() -> Any:
        clock[0] += 2
        if failure == "provider":
            raise error
        return provider

    def embed(*args: Any) -> list[list[float]]:
        clock[0] += 3
        if failure == "embed":
            raise error
        return [[0.0] * 384]

    provider.embed.side_effect = embed

    def retrieve(*args: Any, **kwargs: Any) -> list[Any]:
        clock[0] += 5
        if failure in {"retrieve", "cancel"}:
            raise error
        return []

    monkeypatch.setattr(search_api, "search_code_units_semantically", retrieve)
    monkeypatch.setattr(ask_api, "search_code_units_with_dependencies", retrieve)
    monkeypatch.setattr(ask_api, "get_embedding_provider", factory)
    llm = MockLLMProvider(response="unused")

    def invoke() -> Any:
        if pipeline == "search":
            return search_api.search(uuid4(), "PRIVATE query", session, factory, None)
        return ask_api.ask(
            ask_api.AskRequest(repository_id=uuid4(), q="PRIVATE question"),
            session,
            Mock(),
            llm,
        )

    if failure is None:
        result = invoke()
        assert result.items == [] if pipeline == "search" else result.citations == []
        assert llm.calls == ()
    else:
        with pytest.raises(type(error)) as caught:
            invoke()
        assert caught.value is error
    outcome = (
        "success"
        if failure is None
        else "cancelled"
        if failure == "cancel"
        else "error"
    )
    name = "search_semantic" if pipeline == "search" else "ask_dependency"
    assert (
        sample(
            metrics, "retrieval_duration_seconds_count", pipeline=name, outcome=outcome
        )
        == 1
    )
    expected = 2 if failure == "provider" else 5 if failure == "embed" else 10
    assert (
        sample(
            metrics, "retrieval_duration_seconds_sum", pipeline=name, outcome=outcome
        )
        == expected
    )


def test_cache_hit_excludes_provider_and_retrieval(
    metrics: telemetry.Metrics, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, cache, factory = (
        Mock(),
        Mock(),
        Mock(side_effect=AssertionError("provider acquired")),
    )
    session.scalar.return_value = 0
    cache.read.return_value = {"items": [], "limit": 10}
    monkeypatch.setattr(
        search_api,
        "search_code_units_semantically",
        Mock(side_effect=AssertionError("retrieved")),
    )
    response = search_api.search(uuid4(), "query", session, factory, cache)
    assert response.items == []
    factory.assert_not_called()
    assert (
        sample(
            metrics,
            "retrieval_duration_seconds_count",
            pipeline="search_semantic",
            outcome="success",
        )
        is None
    )


@pytest.mark.parametrize("fails", [False, True])
def test_telemetry_failures_do_not_change_service_results(
    metrics: telemetry.Metrics, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("collector failed")

    for collector in (
        metrics.indexing_executions,
        metrics.indexing_duration,
        metrics.retrieval_duration,
        metrics.llm_duration,
    ):
        monkeypatch.setattr(collector, "labels", broken)
    monkeypatch.setattr(metrics.files_hashed, "inc", broken)
    monkeypatch.setattr(metrics.units_parsed, "inc", broken)
    expected = RuntimeError("application failure")
    provider = Mock()
    response = LLMResponse(content="exact")
    provider.chat.return_value = response
    if fails:
        provider.chat.side_effect = expected
    context = FormattedContext(
        text="x",
        included_evidence_count=1,
        truncated=False,
        used_characters=1,
        max_characters=10,
    )

    def operation() -> LLMResponse:
        with (
            telemetry.indexing_execution("direct"),
            telemetry.retrieval("ask_dependency"),
        ):
            telemetry.file_hashed()
            telemetry.units_parsed(1)
            return generate_grounded_answer(provider, question="q", context=context)

    if fails:
        with pytest.raises(RuntimeError) as caught:
            operation()
        assert caught.value is expected
    else:
        assert operation() is response


@pytest.mark.parametrize("failure", ["counter", "histogram", "clock"])
def test_telemetry_failures_preserve_http_response(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = FastAPI()

    @app.get("/ok")
    def ok() -> dict[str, str]:
        return {"value": "unchanged"}

    metrics = telemetry.install_metrics(app)
    broken = Mock(side_effect=RuntimeError("collector failed"))
    if failure == "clock":
        monkeypatch.setattr(telemetry, "perf_counter", broken)
    else:
        monkeypatch.setattr(
            metrics.http_requests if failure == "counter" else metrics.http_duration,
            "labels",
            broken,
        )
    with TestClient(app) as client:
        response = client.get("/ok")
    assert response.status_code == 200
    assert response.json() == {"value": "unchanged"}


@pytest.mark.parametrize("claimed", [False, True])
def test_runner_cancellation_or_unclaimed_is_not_success(
    metrics: telemetry.Metrics, monkeypatch: pytest.MonkeyPatch, claimed: bool
) -> None:
    @contextmanager
    def factory() -> Iterator[Any]:
        yield Mock()

    monkeypatch.setattr(
        indexing_jobs, "get_application_session_factory", lambda: factory
    )
    monkeypatch.setattr(
        indexing_jobs, "_claim_job", lambda *args: uuid4() if claimed else None
    )
    error = asyncio.CancelledError()
    provider = Mock(side_effect=error)
    monkeypatch.setattr(indexing_jobs, "get_indexing_embedding_provider", provider)
    if claimed:
        with pytest.raises(asyncio.CancelledError) as caught:
            indexing_jobs.run_indexing_job(uuid4())
        assert caught.value is error
        assert (
            sample(
                metrics,
                "indexing_executions_total",
                mode="background",
                outcome="cancelled",
            )
            == 1
        )
    else:
        indexing_jobs.run_indexing_job(uuid4())
        provider.assert_not_called()
        assert (
            sample(
                metrics,
                "indexing_executions_total",
                mode="background",
                outcome="cancelled",
            )
            is None
        )
    assert (
        sample(
            metrics, "indexing_executions_total", mode="background", outcome="success"
        )
        is None
    )


def test_exact_histogram_buckets(metrics: telemetry.Metrics) -> None:
    cases = [
        (
            metrics.http_duration,
            ("GET", "unmatched", "2xx"),
            [0.005, 0.025, 0.1, 0.5, 1, 5, 15, 60, 180],
        ),
        (
            metrics.retrieval_duration,
            ("search_semantic", "success"),
            [0.005, 0.025, 0.1, 0.5, 1, 5, 15, 60, 180],
        ),
        (
            metrics.indexing_duration,
            ("direct", "success"),
            [0.1, 1, 5, 15, 60, 180, 600, 1800],
        ),
        (metrics.llm_duration, ("success",), [0.1, 0.5, 1, 5, 15, 30, 60, 120, 180]),
    ]
    for histogram, labels, expected in cases:
        histogram.labels(*labels).observe(0)
        bounds = [
            float(s.labels["le"])
            for f in histogram.collect()
            for s in f.samples
            if s.name.endswith("_bucket")
        ]
        assert bounds == [*expected, float("inf")]
