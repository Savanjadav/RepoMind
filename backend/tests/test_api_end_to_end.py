"""Offline HTTP workflows; DAY52_TEST_DATABASE_URL must be disposable.

Use a migrated database named repomind_day52_<unique suffix>. Deliberately do
not fall back to DATABASE_URL: requests and the real runner commit independently.
"""

import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app import ask_api, indexing_api, indexing_jobs, repository_indexing, search_api
from app.database import create_database_engine, create_session_factory, get_db_session
from app.llm_provider import LLMRole
from app.main import app
from app.mock_llm_provider import MockLLMProvider
from app.models import CodeUnit, File, IndexingJob, Repository
from app.redis_cache import get_redis_cache
from app.repository_clone import ClonedRepository
from app.repository_source import ValidatedRepositorySource

QUESTION = "What does helper return?"
ANSWER = "helper returns the repository value. [Evidence 1]"


class FakeEmbedding:
    dimension = 384

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        if self.fail:
            raise RuntimeError("PRIVATE_EMBEDDING_FAILURE /private/source")
        # Make the intended answer eligible before the bounded reranker stage.
        return [
            (
                [1.0, 0.0]
                if value == QUESTION or value.startswith("def helper()")
                else [0.0, 1.0]
            )
            + [0.0] * 382
            for value in texts
        ]


class FakeReranker:
    model_name = "offline-e2e"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, tuple(documents)))
        return [10.0 if "\nSymbol: helper\n" in doc else 0.0 for doc in documents]


def _files(value: str) -> dict[str, bytes]:
    return {
        "api.py": b"from service import helper\ndef caller():\n    return helper()\n",
        "service.py": f'def helper():\n    return "{value}"\n'.encode(),
        "README.md": b"# Notes\nSmall offline fixture.\n",
    }


@dataclass
class Workflow:
    client: TestClient
    sources: dict[str, dict[str, bytes]]
    ids: list[UUID]
    embedding: FakeEmbedding
    reranker: FakeReranker
    llm: MockLLMProvider
    clones: list[Path] = field(default_factory=list)

    def register(self, files: dict[str, bytes]) -> dict[str, Any]:
        name = f"day52-{uuid4()}"
        canonical = f"https://github.com/example/{name}"
        self.sources[canonical] = files
        response = self.client.post(
            "/repositories", json={"source": f"https://GitHub.com/Example/{name}.git/"}
        )
        assert response.status_code == 201, response.text
        body = response.json()
        self.ids.append(UUID(body["id"]))
        assert set(body) == {"id", "name", "source", "created_at"}
        assert body["name"] == name and body["source"] == canonical
        assert datetime.fromisoformat(body["created_at"]).tzinfo is not None
        return body

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        response = self.client.get(path, params=params)
        assert response.status_code == 200, response.text
        return response.json()

    def index(self, repository: dict[str, Any], status: str = "completed") -> str:
        rid = repository["id"]
        response = self.client.post(f"/repositories/{rid}/index")
        assert response.status_code == 202, response.text
        job = response.json()
        UUID(job["job_id"])
        assert job == {
            "repository_id": rid,
            "job_id": job["job_id"],
            "status": "pending",
        }
        location = f"/repositories/{rid}/indexing-jobs/{job['job_id']}"
        assert response.headers["location"] == location
        # TestClient awaits BackgroundTasks; the response body is acceptance,
        # not a promise that pending/running can still be observed after return.
        assert self.get(location) == {**job, "status": status}
        assert self.clones and all(not path.exists() for path in self.clones)
        return job["job_id"]

    def ask(self, repository: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(
            "/ask", json={"repository_id": repository["id"], "q": QUESTION, "limit": 1}
        )
        assert response.status_code == 200, response.text
        return response.json()


@pytest.fixture
def workflow(monkeypatch: pytest.MonkeyPatch) -> Iterator[Workflow]:
    url = os.getenv("DAY52_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Explicit disposable DAY52_TEST_DATABASE_URL is required")
    database = make_url(url).database
    if not database or not database.startswith("repomind_day52_"):
        pytest.fail("Day 52 requires a uniquely named repomind_day52_ database")
    engine = create_database_engine(url)
    factory = create_session_factory(engine)
    previous_overrides = dict(app.dependency_overrides)
    ids: list[UUID] = []
    sources: dict[str, dict[str, bytes]] = {}
    embedding, reranker = FakeEmbedding(), FakeReranker()
    llm = MockLLMProvider(response=ANSWER)
    clones: list[Path] = []

    def sessions() -> Iterator[Session]:
        with factory() as session:
            yield session

    def clone(source: ValidatedRepositorySource, workspace: Path) -> ClonedRepository:
        destination = Path(tempfile.mkdtemp(prefix="fixture-", dir=workspace))
        clones.append(destination)
        for path, content in sources[source.value].items():
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return ClonedRepository(source.value, destination)

    try:
        with factory() as session:
            assert session.scalar(text("SELECT current_database()")) == database
            assert (
                session.scalar(text("SELECT version_num FROM alembic_version"))
                == "0008"
            )
            assert session.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname='vector'")
            )
        app.dependency_overrides[get_db_session] = sessions
        app.dependency_overrides[get_redis_cache] = lambda: None
        app.dependency_overrides[search_api.get_search_provider_factory] = lambda: (
            lambda: embedding
        )
        app.dependency_overrides[ask_api.get_reranking_provider] = lambda: reranker
        app.dependency_overrides[ask_api.get_llm_provider] = lambda: llm
        monkeypatch.setattr(
            indexing_api, "get_application_session_factory", lambda: factory
        )
        monkeypatch.setattr(
            indexing_jobs, "get_application_session_factory", lambda: factory
        )
        monkeypatch.setattr(
            indexing_jobs, "get_indexing_embedding_provider", lambda: embedding
        )
        monkeypatch.setattr(ask_api, "get_embedding_provider", lambda: embedding)
        monkeypatch.setattr(repository_indexing, "clone_repository", clone)
        with TestClient(app, raise_server_exceptions=False) as client:
            yield Workflow(client, sources, ids, embedding, reranker, llm, clones)
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)
        try:
            if ids:
                with factory() as session:
                    file_ids = list(
                        session.scalars(
                            select(File.id).where(File.repository_id.in_(ids))
                        )
                    )
                    session.execute(delete(Repository).where(Repository.id.in_(ids)))
                    session.commit()
                with factory() as session:
                    assert (
                        session.scalar(
                            select(func.count())
                            .select_from(Repository)
                            .where(Repository.id.in_(ids))
                        )
                        == 0
                    )
                    assert (
                        session.scalar(
                            select(func.count())
                            .select_from(IndexingJob)
                            .where(IndexingJob.repository_id.in_(ids))
                        )
                        == 0
                    )
                    assert (
                        session.scalar(
                            select(func.count())
                            .select_from(File)
                            .where(File.repository_id.in_(ids))
                        )
                        == 0
                    )
                    assert (
                        session.scalar(
                            select(func.count())
                            .select_from(CodeUnit)
                            .where(CodeUnit.file_id.in_(file_ids))
                        )
                        == 0
                    )
            assert all(not path.exists() for path in clones)
        finally:
            engine.dispose()


def _assert_grounding(
    workflow: Workflow, repository: dict[str, Any], value: str
) -> dict[str, Any]:
    calls = len(workflow.llm.calls)
    answer = workflow.ask(repository)
    content = f'def helper():\n    return "{value}"'
    assert answer == {
        "answer": ANSWER,
        "citations": [
            {
                "evidence_id": 1,
                "repository_name": repository["name"],
                "path": "service.py",
                "symbol_name": "helper",
                "start_line": 1,
                "end_line": 2,
                "source_preview": content,
                "source_preview_truncated": False,
            }
        ],
    }
    assert len(workflow.llm.calls) == calls + 1
    messages = workflow.llm.calls[-1]
    assert [message.role for message in messages] == [LLMRole.SYSTEM, LLMRole.USER]
    assert "untrusted DATA" in messages[0].content
    evidence = (
        f"[Evidence 1]\nRepository: {repository['name']}\nPath: service.py\n"
        "Symbol: helper\nLines: 1-2\n"
        f"Content-Characters: {len(content)}/{len(content)}\n"
        f"Content:\n{content}\n[End Evidence 1]"
    )
    assert len(evidence) <= ask_api.ASK_CONTEXT_MAX_CHARACTERS
    assert (
        messages[1].content
        == f"Question:\n{QUESTION}\n\nAllowed evidence IDs: 1..1\n\n"
        f"Repository Evidence:\n{evidence}"
    )
    assert workflow.reranker.calls[-1][0] == QUESTION
    assert any(f"Content:\n{content}" in doc for doc in workflow.reranker.calls[-1][1])
    return answer


def test_register_index_search_ask_with_repository_isolation(
    workflow: Workflow,
) -> None:
    first = workflow.register(_files("alpha"))
    second = workflow.register(_files("beta"))
    initial = workflow.get(f"/repositories/{first['id']}/overview")
    assert initial == {
        "repository_id": first["id"],
        "index_generation": "0",
        "latest_job": None,
        "snapshot_counts": {"files": 0, "code_units": 0, "named_code_units": 0},
        "code_units_by_language": [],
        "code_units_by_kind": [],
    }
    id_sets = []
    for repository, value in [(first, "alpha"), (second, "beta")]:
        rid = repository["id"]
        job_id = workflow.index(repository)
        overview = workflow.get(f"/repositories/{rid}/overview")
        assert overview["repository_id"] == rid and overview["index_generation"] == "1"
        assert overview["snapshot_counts"] == {
            "files": 3,
            "code_units": 4,
            "named_code_units": 3,
        }
        assert overview["code_units_by_language"] == [
            {"language": "documentation", "count": 1},
            {"language": "python", "count": 3},
        ]
        assert overview["code_units_by_kind"] == [
            {"kind": "document", "count": 1},
            {"kind": "function", "count": 2},
            {"kind": "import", "count": 1},
        ]
        latest = overview["latest_job"]
        assert latest["job_id"] == job_id and latest["status"] == "completed"
        assert datetime.fromisoformat(latest["created_at"]).tzinfo is not None
        assert workflow.get("/repositories/indexing-summary", repository_id=rid) == {
            "items": [
                {
                    "repository_id": rid,
                    "latest_job": latest,
                    "snapshot_counts": {"files": 3, "code_units": 4},
                }
            ]
        }
        browse = workflow.get(f"/repositories/{rid}/code-units")
        assert browse["limit"] == 100 and browse["offset"] == 0
        units = browse["items"]
        assert [u["start_line"] for u in units if u["path"] == "api.py"] == [1, 2]
        # SQL path ordering follows the disposable database's collation.
        assert sorted(
            [
                (u["path"], u["kind"], u["symbol_name"], u["start_line"], u["end_line"])
                for u in units
            ],
            key=lambda metadata: (metadata[0], metadata[3]),
        ) == [
            ("README.md", "document", "Notes", 1, 2),
            ("api.py", "import", None, 1, 1),
            ("api.py", "function", "caller", 2, 3),
            ("service.py", "function", "helper", 1, 2),
        ]
        ids = {u["id"] for u in units}
        assert len(ids) == 4
        id_sets.append(ids)
        search = workflow.get("/search", repository_id=rid, q="helper", limit=100)
        assert (
            search["limit"] == 100
            and {r["code_unit_id"] for r in search["items"]} == ids
        )
        assert len(search["items"]) == 4
        by_id = {u["id"]: u for u in units}
        expected_contents = {
            ("README.md", 1): "# Notes\nSmall offline fixture.\n",
            ("api.py", 1): "from service import helper",
            ("api.py", 2): "def caller():\n    return helper()",
            ("service.py", 1): f'def helper():\n    return "{value}"',
        }
        for row in search["items"]:
            unit = by_id[row["code_unit_id"]]
            assert {key: row[key] for key in unit if key != "id"} == {
                key: val for key, val in unit.items() if key != "id"
            }
            assert row["content"] == expected_contents[(row["path"], row["start_line"])]
        _assert_grounding(workflow, repository, value)
    assert id_sets[0].isdisjoint(id_sets[1])


def test_duplicate_registration_keeps_original_indexable(workflow: Workflow) -> None:
    repository = workflow.register(_files("alpha"))
    response = workflow.client.post(
        "/repositories", json={"source": repository["source"]}
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "This repository is already registered"}
    # Scoped lookup, not a global row-count assumption; walk bounded public pages.
    found: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = workflow.get("/repositories", limit=100, offset=offset)
        found.extend(r for r in page["items"] if r["source"] == repository["source"])
        if len(page["items"]) < 100:
            break
        offset += 100
        assert offset < 10_000
    assert found == [repository]
    workflow.index(repository)
    _assert_grounding(workflow, repository, "alpha")


def test_failed_reindex_retains_public_snapshot(workflow: Workflow) -> None:
    repository = workflow.register(_files("alpha"))
    rid = repository["id"]
    previous_job = workflow.index(repository)
    overview = workflow.get(f"/repositories/{rid}/overview")
    units = workflow.get(f"/repositories/{rid}/code-units")
    search = workflow.get("/search", repository_id=rid, q="helper", limit=100)
    answer = _assert_grounding(workflow, repository, "alpha")
    workflow.sources[repository["source"]] = _files("FAILED_REPLACEMENT")
    before_calls = len(workflow.embedding.calls)
    workflow.embedding.fail = True
    try:
        failed_job = workflow.index(repository, "failed")
    finally:
        workflow.embedding.fail = False
    assert failed_job != previous_job
    assert len(workflow.embedding.calls) == before_calls + 1
    assert workflow.embedding.calls[-1] == (
        'def helper():\n    return "FAILED_REPLACEMENT"',
    )
    assert (
        workflow.get(f"/repositories/{rid}/indexing-jobs/{previous_job}")["status"]
        == "completed"
    )
    retained = workflow.get(f"/repositories/{rid}/overview")
    assert {k: v for k, v in retained.items() if k != "latest_job"} == {
        k: v for k, v in overview.items() if k != "latest_job"
    }
    assert retained["latest_job"]["job_id"] == failed_job
    assert retained["latest_job"]["status"] == "failed"
    summary = workflow.get("/repositories/indexing-summary", repository_id=rid)
    assert summary == {
        "items": [
            {
                "repository_id": rid,
                "latest_job": retained["latest_job"],
                "snapshot_counts": None,
            }
        ]
    }
    assert workflow.get(f"/repositories/{rid}/code-units") == units
    assert workflow.get("/search", repository_id=rid, q="helper", limit=100) == search
    assert _assert_grounding(workflow, repository, "alpha") == answer
    assert "PRIVATE_EMBEDDING_FAILURE" not in str(retained) + str(summary)
    assert "FAILED_REPLACEMENT" not in str(search) + str(answer)


def test_empty_supported_repository_bypasses_llm(workflow: Workflow) -> None:
    repository = workflow.register({"empty.py": b"", "README.md": b" \n"})
    workflow.index(repository)
    rid = repository["id"]
    assert workflow.get(f"/repositories/{rid}/overview")["snapshot_counts"] == {
        "files": 2,
        "code_units": 0,
        "named_code_units": 0,
    }
    assert workflow.get(f"/repositories/{rid}/code-units")["items"] == []
    assert workflow.embedding.calls == []
    assert workflow.get("/search", repository_id=rid, q="helper") == {
        "items": [],
        "limit": 10,
    }
    assert workflow.ask(repository) == {
        "answer": "The available evidence is insufficient to answer the question.",
        "citations": [],
    }
    assert workflow.llm.calls == () and workflow.reranker.calls == []
