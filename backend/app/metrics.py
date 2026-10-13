"""Process-local telemetry. Observations must never change application outcomes."""

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Literal

from fastapi import FastAPI
from fastapi.routing import APIRoute
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.llm_provider import LLMResponseError, LLMTimeoutError, LLMUnavailableError

_REQUEST_BUCKETS = (0.005, 0.025, 0.1, 0.5, 1, 5, 15, 60, 180)
_INDEX_BUCKETS = (0.1, 1, 5, 15, 60, 180, 600, 1800)
_LLM_BUCKETS = (0.1, 0.5, 1, 5, 15, 30, 60, 120, 180)
_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
)


def _observe(action: Callable[[], object]) -> None:
    # Catch only inside the telemetry boundary, never around application work.
    # Do not log exception strings: collectors could contain sensitive values.
    try:
        action()
    except Exception:
        pass


def _clock() -> float | None:
    try:
        return perf_counter()
    except Exception:
        return None


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "repomind_http_requests_total",
            "Observed HTTP responses or interruptions.",
            ("method", "route", "status_class"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "repomind_http_request_duration_seconds",
            "HTTP time through final body send, excluding background tasks.",
            ("method", "route", "status_class"),
            buckets=_REQUEST_BUCKETS,
            registry=self.registry,
        )
        self.indexing_executions = Counter(
            "repomind_indexing_executions_total",
            "Observed executions, not authoritative persisted job outcomes.",
            ("mode", "outcome"),
            registry=self.registry,
        )
        self.indexing_duration = Histogram(
            "repomind_indexing_duration_seconds",
            "Execution time: no queue wait or direct caller commit.",
            ("mode", "outcome"),
            buckets=_INDEX_BUCKETS,
            registry=self.registry,
        )
        self.files_hashed = Counter(
            "repomind_indexing_files_hashed_total",
            "Files successfully hashed, including work later rolled back.",
            registry=self.registry,
        )
        self.units_parsed = Counter(
            "repomind_indexing_code_units_parsed_total",
            "Newly parsed units, including work later rolled back; excludes reuse.",
            registry=self.registry,
        )
        self.retrieval_duration = Histogram(
            "repomind_retrieval_duration_seconds",
            "Query embedding/provider acquisition through retrieval; no cache hits.",
            ("pipeline", "outcome"),
            buckets=_REQUEST_BUCKETS,
            registry=self.registry,
        )
        self.llm_duration = Histogram(
            "repomind_llm_request_duration_seconds",
            "Actual provider.chat calls, excluding empty-context bypass.",
            ("outcome",),
            buckets=_LLM_BUCKETS,
            registry=self.registry,
        )


_default = Metrics()
_selected: ContextVar[Metrics | None] = ContextVar("repomind_metrics", default=None)


def get_metrics() -> Metrics:
    return _selected.get() or _default


@contextmanager
def use_metrics(metrics: Metrics) -> Iterator[None]:
    token = _selected.set(metrics)
    try:
        yield
    finally:
        _selected.reset(token)


def _duration(
    histogram: Histogram, labels: tuple[str, ...], start: float | None
) -> None:
    end = _clock()
    if start is not None and end is not None:
        _observe(lambda: histogram.labels(*labels).observe(max(0.0, end - start)))


@contextmanager
def indexing_execution(mode: Literal["direct", "background"]) -> Iterator[None]:
    metrics, start, outcome = get_metrics(), _clock(), "success"
    try:
        yield
    except BaseException as error:
        outcome = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
        raise
    finally:
        _observe(lambda: metrics.indexing_executions.labels(mode, outcome).inc())
        _duration(metrics.indexing_duration, (mode, outcome), start)


@contextmanager
def retrieval(pipeline: Literal["search_semantic", "ask_dependency"]) -> Iterator[None]:
    metrics, start, outcome = get_metrics(), _clock(), "success"
    try:
        yield
    except BaseException as error:
        outcome = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
        raise
    finally:
        _duration(metrics.retrieval_duration, (pipeline, outcome), start)


@contextmanager
def llm_request() -> Iterator[None]:
    metrics, start, outcome = get_metrics(), _clock(), "success"
    try:
        yield
    except BaseException as error:
        if isinstance(error, asyncio.CancelledError):
            outcome = "cancelled"
        elif isinstance(error, LLMTimeoutError):
            outcome = "timeout"
        elif isinstance(error, LLMUnavailableError):
            outcome = "unavailable"
        elif isinstance(error, LLMResponseError):
            outcome = "response_error"
        else:
            outcome = "error"
        raise
    finally:
        _duration(metrics.llm_duration, (outcome,), start)


def file_hashed() -> None:
    _observe(lambda: get_metrics().files_hashed.inc())


def units_parsed(count: int) -> None:
    _observe(lambda: get_metrics().units_parsed.inc(count))


class MetricsMiddleware:
    def __init__(self, app: ASGIApp, *, owner: FastAPI, metrics: Metrics) -> None:
        self.app, self.owner, self.metrics = app, owner, metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start, status, recorded = _clock(), None, False

        def record(classification: str) -> None:
            nonlocal recorded
            if recorded:
                return
            recorded = True

            def observe() -> None:
                route = scope.get("route")
                if route is self.owner.state.metrics_route:
                    return
                # FastAPI sets this to the matched registered APIRoute, including
                # included routers that are not flattened into owner.routes.
                # Never derive a template from request path/query data.
                template = route.path if isinstance(route, APIRoute) else "unmatched"
                method = scope.get("method", "")
                method = method if method in _METHODS else "OTHER"
                labels = (method, template, classification)
                _observe(lambda: self.metrics.http_requests.labels(*labels).inc())
                _duration(self.metrics.http_duration, labels, start)

            _observe(observe)

        async def observed_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                record(
                    f"{status // 100}xx"
                    if status is not None and 200 <= status < 600
                    else "incomplete"
                )

        with use_metrics(self.metrics):
            try:
                await self.app(scope, receive, observed_send)
            except BaseException as error:
                record(
                    "cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else "5xx"
                    if status is None
                    else "incomplete"
                )
                raise
            finally:
                record("incomplete")


def install_metrics(app: FastAPI, metrics: Metrics | None = None) -> Metrics:
    """Each app gets its own registry unless explicitly given the process registry."""
    existing = getattr(app.state, "metrics", None)
    if isinstance(existing, Metrics):
        return existing
    selected = metrics if metrics is not None else Metrics()
    app.state.metrics = selected

    async def exposition() -> Response:
        return Response(
            generate_latest(selected.registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    app.add_api_route("/metrics", exposition, methods=["GET"], include_in_schema=False)
    app.state.metrics_route = app.routes[-1]
    app.add_middleware(MetricsMiddleware, owner=app, metrics=selected)
    return selected
