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
> with pgvector storage. Repository questions use bounded retrieval, outgoing-call
> expansion, and grounded local generation. The frontend remains in development.

## Currently Implemented

### Backend and Persistence

- FastAPI application with a health endpoint and request-scoped database
  sessions.
- PostgreSQL models and Alembic migrations for repositories, files, indexing
  jobs, and code units.
- Exact source content, language, symbol, path relationship, and inclusive line
  ranges preserved in persisted code units.
- Nullable 384-dimensional CodeUnit embeddings stored through pgvector.
- Repository-owned File → File `imports` relationships, with database-enforced
  same-repository endpoints, deduplication, and cascading deletion.

### Local Import Relationships

Indexing records a lightweight, static import graph after parsing all safely
discovered files. Relationships refer to files, not resolved symbols or runtime
calls. Repeated imports produce one edge; self-imports are omitted.

- Python supports conventional `__init__.py` packages, package-relative imports,
  and repository-root absolute modules. Explicit modules are resolved, not
  imported names; aliases and wildcards do not expand symbol dependencies.
- JavaScript/TypeScript supports static relative ES imports, including side-effect
  and type imports, to `.js`, `.jsx`, or `.ts` files. Extensionless paths also
  consider those extensions and directory `index` files, requiring one match.
- Missing or ambiguous targets, standard-library/builtin Python modules, external
  packages, inferred Python source roots/namespace packages, JS aliases, escaped
  specifiers, dynamic imports, `require`, re-exports, and package/tsconfig resolution
  are not indexed as relationships.

Resolution uses only the current run's discovered file inventory, never executes
repository code, and makes no model/network calls. This is a conservative
structural map, not compiler-grade or runtime dependency analysis. Existing
repositories are not automatically backfilled. File import relationships are
not used for evidence expansion; a file edge alone does not identify relevant
CodeUnits.

### Local Call Hints

Indexing also records separate CodeUnit → CodeUnit `calls` relationships. A hint
means a direct call in an eligible function body has an unambiguous local binding
under conservative static rules—not that it executes or that dynamic dispatch is
understood. Recursion is allowed; repeated caller/callee pairs produce one edge.

- Python supports undecorated module-level functions (including async), direct
  calls, explicit `from` imports/aliases, and module-qualified calls through simple
  imports or explicit module aliases. Conventional local module resolution is
  shared with the import graph; imported re-exports are not followed.
- JS/TS supports module-level function declarations and function-valued `const`
  bindings, named imports/aliases, explicit named default function imports, and
  namespace calls to direct exports. Only the existing relative ES module forms
  are resolved; type-only imports create no runtime call hints.
- Shadowing, rebinding, duplicate definitions, ambiguous endpoints, unsupported
  scopes, wildcard imports, constructors, methods, nested callable bodies,
  assignment aliases, dynamic calls and re-export chains are omitted. Conservative
  invalidation can omit otherwise valid calls; this is not compiler symbol binding.

Analysis uses the already-read full-file source and persisted CodeUnit identities.
It never executes repository code or makes additional embedding/model calls.
Files with more than 10,000 call occurrences have their call analysis discarded;
repository runs retain at most 10,000 distinct edges in stable path/source order.
The graph is intentionally partial. Database constraints enforce repository,
file, and CodeUnit ownership, uniqueness, and cascading deletion. Call hints are
flushed inside the existing indexing savepoint, without committing the caller's
transaction. Migration 0006 adds this separate table; it does not backfill existing
repositories. `/ask` uses the bounded outgoing-call expansion described below.
General graph traversal APIs and visualization are not implemented.

### Dependency-Aware Answer Evidence

`/ask` composes existing semantic/lexical hybrid retrieval and reranking with
one-hop expansion through persisted outgoing CodeUnit `calls` hints. Incoming
callers and File `imports` edges are not expanded. `/search` remains semantic-only.

At most three top direct seeds contribute at most two neighbors each. Graph
additions are capped at `min(4, limit // 3)`; final evidence never exceeds the
requested `limit` (1–100). Only actual accepted additions reserve space: at limit
10, zero additions retain ten direct results, while one addition retains nine.
Graph evidence is appended after the retained direct prefix and may displace only
the lowest-ranked direct tail. Direct ranking remains authoritative; graph units
receive no fabricated scores and are not reranked. Duplicates are removed by
CodeUnit identity, and all results remain repository-scoped.

Expansion reads persisted data only, without new embedding/model calls. The
existing 8,000-character context budget still applies and may omit appended
neighbors. Call hints are intentionally incomplete, not compiler-precise or
proof of runtime behavior. No automatic backfill or reindexing is performed.

### Basic Impact Analysis

`GET /repositories/{repository_id}/impact` returns likely direct dependents from
persisted graph edges. An exact `path` selects file mode: incoming File `imports`
edges identify direct importers. Adding an exact `symbol` selects symbol mode:
incoming CodeUnit `calls` edges identify direct callers. This is one hop only,
not recursive traversal. `/ask`'s outgoing callee expansion remains separate.

Responses contain the persisted target and dependent metadata, `analysis:
"static_hint"`, `limit`, and `truncated`. Caller line ranges describe the caller
CodeUnit, not exact call sites; file-level items have null symbol/line metadata.
No source content, ranking scores, or model-generated answer is returned.

These conservative static hints neither predict breakage nor cover all runtime
dependencies. An empty result means no matching stored direct hints were found,
not that a change is safe. `truncated=false` likewise describes only the stored
direct results, not real-world completeness. No model calls or writes occur.

### Incremental Repository Indexing

Explicit calls to `index_repository()` handle both first and repeated indexing.
Each accepted run creates a new IndexingJob. Pending/running jobs still block;
completed/failed history remains available. Same-repository writers serialize
through a PostgreSQL Repository row lock held until the caller's outer transaction
ends, not merely until the service returns. Different repositories are independent.

Each admitted File records SHA-256 of its exact safely read bytes. An unchanged
path/hash preserves its File, CodeUnit identities, metadata and embeddings.
Changed files retain their File identity but replace all units and embeddings;
new files are indexed normally. Removed or intentionally excluded files lose their
derived data. Renames are delete plus add, without similarity guessing. Legacy
null hashes are unverified and refreshed on the next successful index. Empty and
metadata-only files also receive hashes. Hashes track bytes, not parser/model
versions; switching those does not automatically invalidate unchanged files.

A changed admitted snapshot rebuilds both repository-local graphs against the
complete current inventory, including unchanged callers/importers. This may
reanalyze source for graph hints but never re-embeds unchanged units. An identical
snapshot skips CodeUnit parsing, embeddings and graph rebuilding entirely, keeping
relationship identities. Discovery and hashing still occur. Unexpected discovery
I/O/identity failures abort rather than masquerading as deleted files. Supported
source reread for analysis must match its classified hash.

Snapshot changes share the existing savepoint: failure restores the previous
Files, hashes, units, embeddings and edges, while the new job follows failed-job
semantics. The caller owns commit/rollback and must commit to retain job status.
Owned clones are cleaned on success/failure; cleanup failure also prevents a
successful snapshot update. Repository source is never executed.

Migration 0007 adds nullable `files.content_hash` with lowercase SHA-256 validation,
without a default or backfill. No new dependency/service is required. Indexing is
still explicitly invoked: no source watching, source polling, webhooks, Git
diff/merge-base processing, rename similarity or chunk-level incremental embeddings.

### Background Indexing Jobs

For an existing Repository, `POST /repositories/{repository_id}/index` reserves
one pending job and returns HTTP 202 with `job_id`, `repository_id`, and
`status: "pending"`. Its `Location` header points to
`GET /repositories/{repository_id}/indexing-jobs/{job_id}`. Poll that metadata-only
resource for coarse persisted status: `pending`, `running`, `completed`, or
`failed`. Missing resources return 404; active indexing or a busy repository
returns 409. No repository registration endpoint is added.

The synchronous runner executes through Starlette BackgroundTasks' thread pool
after the response body is sent. It creates independent database Sessions,
commits the running claim before indexing, and uses the same incremental pipeline
described above. Failed indexing preserves the previous snapshot; the runner
persists failed status when the database remains available. Direct synchronous
`index_repository()` callers still own their outer transaction.

Execution is in-process and non-durable. Abrupt termination can leave pending or
running jobs (which block another request) and temporary workspaces. There is no
automatic stale-job/restart recovery, retry, exactly-once delivery, durable queue,
or Redis integration. Thread-pool, database, and model resources are shared process
limits; background model batches are serialized without globally locking entire
indexing jobs. Day 42 does not promise a durable queue.

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

Semantic nearest-neighbor search is available through `/search` and participates
in `/ask` hybrid retrieval before reranking and dependency expansion.

## The Problem

Understanding an unfamiliar codebase takes time. Developers often need to trace
behavior across files, find the source of a feature, identify relevant tests,
or estimate what a change might affect. Text search is valuable for exact
terms, but it does not always capture intent or explain relationships between
code elements. General-purpose AI tools can also produce convincing answers
without showing whether their evidence is correct.

RepoMind aims to make codebase exploration faster and more trustworthy by
combining syntax-aware code analysis, inspectable retrieval, and answers that
point back to specific source files and line ranges. The backend implements
retrieval and grounded answer generation, including bounded connected evidence.

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

The indexing service orchestrates these components, then resolves and persists
local import relationships and call hints inside the same indexing savepoint. Failures roll
back derived data; the caller controls the outer transaction.

For answers, query embedding → hybrid retrieval → reranking → bounded outgoing
call expansion → bounded context → local LLM → answer/citation mapping is the
current `/ask` path. Expansion is read-only and does not alter indexing.

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

Local language-model generation uses Ollama through a provider boundary. Hosted
AI is not required for the intended core MVP, and repository content should not be
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
- Ollama
- Docker Compose
- pytest, Ruff, and Mypy

### Planned for v0.1.0

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

Existing installations must also run `alembic upgrade head` to apply migration
0005 (file imports) and 0006 (CodeUnit call hints and endpoint ownership
constraints), and 0007 (nullable file content hashes). No additional service or
dependency is required. These migrations do not backfill relationships or hashes.

The API is then available at `http://127.0.0.1:8000`. The application requires
`DATABASE_URL` for database-backed endpoints. Repositories must already exist in
the database: the indexing endpoint schedules their ingestion but does not
register them. The code-unit browser operates on persisted database state.

## Current API

### Repository Impact

`GET /repositories/{repository_id}/impact` accepts required `path` (exact relative
POSIX path, 1–2,000 characters), optional `symbol` (exact non-whitespace name,
1–2,000 characters), and `limit` (default 20, range 1–100). Controls and malformed
paths are rejected; accepted spelling is preserved. Missing repositories, files,
or symbols return 404. Multiple exact symbol matches in the file return 409,
without guessing by kind or line range. Invalid input returns 422.

File results are ordered by path then File ID. Caller results are ordered by path,
start line, descending end line, kind, then CodeUnit ID; identities are unique.
Recursive self-calls are excluded. At most `limit` items are returned, with
`truncated=true` when an additional stored direct dependent exists. Valid targets
without incoming hints return 200 with an empty `items` list.

### Repository Questions

`POST /ask` accepts JSON with `repository_id`, `q` (1–2,000 characters, with
non-whitespace content), and optional `limit` (default 10, range 1–100).
It returns `answer` and structured `citations` for evidence actually included in
the formatted context. The fixed evidence budget follows the dependency-expansion
policy above; response and citation formats are unchanged by expansion.

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
index with `vector_cosine_ops`; existing units may remain unembedded. Semantic
retrieval excludes missing/zero vectors; outgoing-call expansion can include a
persisted neighbor without computing a new embedding.

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
implemented and tested. The indexing pipeline also records conservative local
file-import relationships and bounded local call hints. `/ask` now supplements
direct retrieval with bounded, one-hop outgoing-call evidence.
