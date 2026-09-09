# RepoMind

RepoMind is an AI-powered codebase intelligence platform under active
development. It is designed to help developers understand unfamiliar Git
repositories through code-aware retrieval and, ultimately, evidence-grounded
answers that can be checked against source files and line ranges.

> **Project status:** Active development. RepoMind currently includes a FastAPI
> backend, PostgreSQL/pgvector persistence, controlled public GitHub cloning and
> safe file filtering, syntax-aware parsing for Python, JavaScript, and
> TypeScript, documentation/configuration chunking, persisted code units, a
> read-only code-unit browser API, and local Sentence Transformers embeddings
> with pgvector storage. End-to-end indexing, retrieval, RAG, and the frontend
> are still in development.

## Currently Implemented

### Backend and Persistence

- FastAPI application with a health endpoint and request-scoped database
  sessions.
- PostgreSQL models and Alembic migrations for repositories, files, indexing
  jobs, and code units.
- Exact source content, language, symbol, path relationship, and inclusive line
  ranges preserved in persisted code units.
- Nullable 384-dimensional CodeUnit embeddings stored through pgvector.

### Safe Repository Ingestion

- Validation for supported HTTPS repository URLs and local-path source values.
- Controlled, non-interactive cloning for validated public
  `https://github.com/...` repositories. Git hooks, credential prompts,
  redirects, submodule recursion, and Git LFS downloads are disabled.
- Deterministic repository scanning that excludes symlinks, special files,
  nested repositories, binary content, oversized files, likely secrets,
  dependency/vendor trees, generated output, lockfiles, and other noisy
  artifacts.
- Safe repository-relative POSIX paths and persisted file path metadata.

The implemented clone operation is intentionally limited to public GitHub HTTPS
repositories. Local source values can be validated, but local-repository
ingestion is not yet connected to a public workflow.

### Syntax-Aware Parsing

- A provider-independent `CodeParser` protocol and validated `ParsedCodeUnit`
  representation.
- Tree-sitter parsing for Python functions, classes, imports, methods, nested
  symbols, and decorated definitions.
- Tree-sitter parsing for JavaScript and TypeScript functions, classes, class
  methods, ES module imports, exports, and supported named function-valued
  declarations.
- Deterministic documentation and configuration text chunking, including
  lightweight Markdown heading and fenced-code handling.
- Exact source slices, Unicode content, repository-relative paths, and 1-based
  inclusive line ranges.

The parsers do not execute analyzed code and do not attempt compiler-grade call
graph or whole-program analysis.

### Code Unit Inspection

Persisted units can be inspected without invoking an LLM:

```http
GET /repositories/{repository_id}/code-units
```

The endpoint returns bounded, deterministic metadata including the file path,
unit kind, language, line range, and optional symbol name. It does not return
the stored source content.

### Local Embeddings

- A provider-independent `EmbeddingProvider` batch contract.
- `SentenceTransformerEmbeddingProvider` using
  `sentence-transformers/all-MiniLM-L6-v2` by default.
- Local batch inference with plain Python vectors at the provider boundary.
- Fixed-width `vector(384)` storage on CodeUnits.
- An HNSW index configured with pgvector's cosine operator class.
- Validated persistence of already-generated embeddings for existing
  CodeUnits.

The vector column and index are storage foundations. Semantic nearest-neighbor
search is not implemented yet.

## The Problem

Understanding an unfamiliar codebase takes time. Developers often need to trace
behavior across files, find the source of a feature, identify relevant tests,
or estimate what a change might affect. Text search is valuable for exact
terms, but it does not always capture intent or explain relationships between
code elements. General-purpose AI tools can also produce convincing answers
without showing whether their evidence is correct.

RepoMind aims to make codebase exploration faster and more trustworthy by
combining syntax-aware code analysis, inspectable retrieval, and answers that
point back to specific source files and line ranges. Retrieval and grounded
answer generation remain planned work; the current implementation establishes
the ingestion, parsing, persistence, browsing, and embedding foundations.

## Who RepoMind Is For

RepoMind is intended for developers who need to:

- become productive in an unfamiliar repository;
- investigate how a feature works across multiple files;
- locate important symbols, configuration, and tests;
- explore likely code relationships and change impact;
- verify AI-assisted explanations against the original source.

The initial MVP will focus on individual developers working with supported
public Git repositories or repositories already available locally.

## Current Architecture

The following building blocks exist today:

```text
Public GitHub HTTPS repository
        |
        v
Source validation
        |
        v
Controlled Git clone
        |
        v
Safe file filtering
        |
        v
File path metadata persistence
        |
        v
Tree-sitter / text parsers
        |
        v
CodeUnit persistence

CodeUnit content
        |
        v
EmbeddingProvider
        |
        v
Sentence Transformers
        |
        v
384-dimensional vector
        |
        v
PostgreSQL + pgvector
```

These components exist today, but the full repository-to-embedding flow is not
yet orchestrated as one end-to-end indexing operation.

## Target v0.1.0 Workflow

The target high-level workflow for the MVP is:

```text
Git repository
    -> safe file filtering
    -> syntax-aware parsing
    -> code, document, and configuration units
    -> local embeddings and searchable metadata
    -> semantic and lexical retrieval
    -> reranking and bounded relationship expansion
    -> local language model
    -> grounded answer with file and line citations
```

This is the target architecture, not the current end-to-end feature set.
Repository contents will continue to be treated as untrusted input and must not
be executed as part of analysis.

## v0.1.0 MVP Scope

The MVP aims to provide:

- ingestion of supported public or local Git repositories;
- safe filtering of irrelevant, generated, binary, oversized, and sensitive
  files;
- syntax-aware extraction of supported functions, classes, imports, and source
  ranges;
- indexing of selected source code, documentation, tests, and configuration
  text;
- local embedding generation and vector search;
- lexical search for exact identifiers, paths, and configuration terms;
- hybrid retrieval that combines semantic and lexical evidence;
- lightweight reranking and bounded dependency-aware retrieval;
- raw search results that can be inspected without invoking an LLM;
- repository question answering through a local LLM by default;
- structured citations containing file, symbol, and line-range information;
- basic Git-aware incremental re-indexing;
- retrieval evaluation using reproducible questions and expected evidence;
- a browser interface for repository onboarding, indexing status, search,
  questions, and citations;
- a reproducible local environment, automated tests, and continuous
  integration.

This section describes intended release scope. Capabilities not listed under
Currently Implemented remain planned.

## Not Included in v0.1.0

The following are intentionally outside the MVP scope:

- production-ready private-repository OAuth or GitHub App integration;
- perfect support for every programming language;
- compiler-grade whole-program analysis or a complete call graph;
- guarantees of exact dependency or change-impact analysis for dynamic
  languages;
- autonomous code modification or execution of analyzed repository code;
- an IDE extension;
- mandatory hosted AI or paid API services;
- a managed commercial vector database;
- large-scale distributed search infrastructure;
- Kubernetes or production cloud deployment;
- multi-agent investigation workflows;
- organization-wide permissions, collaboration, or enterprise administration.

These boundaries keep the first release focused on a small, correct, and
explainable code-intelligence workflow.

## Local-First and Free-First

Local embedding generation is implemented through a provider-independent
interface and Sentence Transformers. The default model is
`sentence-transformers/all-MiniLM-L6-v2`, and the current database schema stores
its 384-dimensional output. Model artifacts may be downloaded into the standard
Sentence Transformers/Hugging Face cache on first use; inference then runs
locally.

Local language-model generation through Ollama remains planned. Hosted AI is
not required for the intended core MVP, and repository content should not be
sent to hosted models unless a hosted mode is intentionally configured in the
future.

## v0.1.0 Success Criteria

The MVP will be considered successful when a developer can:

- submit a supported public or local repository;
- index useful repository content without executing repository code;
- inspect indexing progress and failures;
- run semantic, lexical, and hybrid searches within a repository;
- inspect raw retrieval results separately from generated answers;
- ask a codebase question and receive an evidence-grounded response;
- open citations that resolve to valid files and line ranges;
- perform basic, clearly qualified dependency-aware investigation;
- re-index changes without unnecessarily reprocessing every unchanged file;
- run the core application locally without a paid AI API;
- reproduce the local environment using documented container configuration;
- run automated tests and continuous-integration checks;
- review recorded retrieval-quality measurements from a reproducible evaluation
  set.

Success will be based on working, testable behavior and valid evidence—not on
unsupported claims, invented benchmarks, or the apparent confidence of
generated text.

## Technology Stack

### Implemented

- Python 3.12+
- FastAPI and Uvicorn
- SQLAlchemy and Alembic
- PostgreSQL 17 and pgvector
- Tree-sitter with Python, JavaScript, and TypeScript grammars
- Sentence Transformers
- Docker Compose
- pytest, Ruff, and Mypy

### Planned for v0.1.0

- Ollama
- Redis
- Next.js and React
- GitHub Actions
- Prometheus and Grafana

Technologies are introduced only when their implementation stage requires
them. RepoMind favors small, testable components over premature infrastructure.

## Run the Backend Locally

Requirements:

- Python 3.12 or newer;
- Docker with Docker Compose;
- Git for repository cloning.

From the repository root, create the local environment file and start the
tracked PostgreSQL/pgvector service:

```bash
cp .env.example .env
docker compose up -d database
docker compose ps
```

The tracked environment template maps PostgreSQL to `127.0.0.1:5433`. Wait for
the `database` service to report healthy, then install and migrate the backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --editable '.[dev]'

set -a
source ../.env
set +a

alembic upgrade head
uvicorn app.main:app --reload
```

The API is then available at `http://127.0.0.1:8000`. The application requires
`DATABASE_URL` for database-backed endpoints. No current API endpoint registers,
clones, or indexes a repository end to end; the code-unit browser operates on
already-persisted database state.

## Current API

### Health

```http
GET /health
```

Returns:

```json
{"status": "ok"}
```

### Repository Code Units

```http
GET /repositories/{repository_id}/code-units
```

- `repository_id`: required repository UUID.
- `kind`: optional single-value filter: `class`, `function`, `import`,
  `document`, or `config`.
- `limit`: optional result limit, default `100`, minimum `1`, maximum `500`.
- `offset`: optional non-negative offset, default `0`.

The response contains `items`, `limit`, and `offset`. Each item contains:

```text
id, file_id, path, kind, language, start_line, end_line, symbol_name
```

Results are ordered deterministically and scoped to the requested repository.
The endpoint returns metadata only; it does not return CodeUnit content or
perform search.

## Vector Storage Note

CodeUnit embeddings are nullable `vector(384)` values. PostgreSQL uses an HNSW
index with `vector_cosine_ops`; existing units may remain unembedded. Storage
and indexing infrastructure are present, but semantic nearest-neighbor
retrieval is not implemented yet.

## Safety and Trust Principles

The current ingestion and parsing components enforce these boundaries:

- analyzed repository code is not executed;
- repository source, documentation, configuration, and embedded instructions
  are treated as untrusted data;
- remote cloning is limited to validated public GitHub HTTPS sources and uses a
  controlled, non-interactive Git invocation;
- symlinks, special files, likely secrets, binaries, oversized files, and
  dependency/generated directories are excluded during discovery;
- database operations use SQLAlchemy or fixed migration SQL rather than SQL
  assembled from repository content;
- secrets, cloned repositories, database data, model files, and generated
  caches stay outside version control.

Later retrieval and RAG work is intended to:

- keep retrieved evidence separate from system instructions;
- ground generated answers in repository evidence;
- preserve paths, symbols, and line ranges through retrieval and citations;
- keep retrieval testable independently of an LLM;
- report insufficient evidence rather than fabricate an answer;
- qualify dependency and impact analysis where dynamic-language behavior makes
  certainty impossible.

These later RAG protections are design requirements, not currently completed
features.

## Project Status

RepoMind is under active development. The ingestion, parsing, persistence,
code-unit inspection, local embedding, and pgvector storage foundations are
implemented and tested. The next major milestone is connecting these components
into an end-to-end repository indexing pipeline, followed by semantic and
hybrid retrieval, grounded local-LLM question answering, and the browser
interface.
