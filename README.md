# RepoMind

RepoMind is a planned AI-powered codebase intelligence platform designed to help developers understand unfamiliar Git repositories through code-aware search and evidence-grounded answers.

> **Project status:** Development is in progress. RepoMind is currently at the project-definition stage; the MVP capabilities described below are planned and are not yet implemented.

## The Problem

Understanding an unfamiliar codebase takes time. Developers often need to trace behavior across files, find the source of a feature, identify relevant tests, or estimate what a change might affect. Text search is valuable for exact terms, but it does not always capture intent or explain relationships between code elements. General-purpose AI tools can produce convincing answers without showing whether the underlying evidence is correct.

RepoMind is designed to make codebase exploration faster and more trustworthy by combining syntax-aware code analysis, multiple retrieval strategies, and answers that point back to specific source files and line ranges.

## Who RepoMind Is For

RepoMind is intended for developers who need to:

- become productive in an unfamiliar repository;
- investigate how a feature works across multiple files;
- locate important symbols, configuration, and tests;
- explore likely code relationships and change impact;
- verify AI-assisted explanations against the original source.

The initial MVP will focus on individual developers working with supported public Git repositories or repositories already available locally.

## What RepoMind Will Do

The v0.1.0 MVP is designed to accept a repository, build a searchable representation of its useful source and documentation files, retrieve evidence relevant to a developer's question, and use a local language model to produce an answer grounded in that evidence.

RepoMind will preserve source metadata throughout the process so that retrieved results and generated answers can identify the relevant file, symbol, start line, and end line. Retrieval will also remain inspectable independently from answer generation, making it possible to determine whether a weak answer came from missing evidence or from the language model's reasoning.

## Planned Workflow

The target high-level workflow for the MVP is:

```text
Git repository
    -> safe file filtering
    -> syntax-aware parsing
    -> functions, classes, modules, and document units
    -> local embeddings and searchable metadata
    -> semantic and lexical retrieval
    -> reranking and bounded relationship expansion
    -> local language model
    -> grounded answer with file and line citations
```

Repository contents will be treated as untrusted input. RepoMind is designed to analyze repository files without executing code from the repository being inspected.

## v0.1.0 MVP Scope

The MVP will aim to provide:

- ingestion of supported public or local Git repositories;
- safe filtering of irrelevant, generated, binary, oversized, and sensitive files;
- syntax-aware extraction of functions, classes, modules, imports, and source ranges;
- indexing of selected source code, documentation, tests, and configuration text;
- local embedding generation and vector search;
- lexical search for exact identifiers, paths, and configuration terms;
- hybrid retrieval that combines semantic and lexical evidence;
- lightweight reranking and bounded dependency-aware retrieval;
- raw search results that can be inspected without invoking an LLM;
- repository question answering through a local LLM by default;
- structured citations containing file, symbol, and line-range information;
- basic Git-aware incremental re-indexing;
- retrieval evaluation using reproducible questions and expected evidence;
- a browser interface for repository onboarding, indexing status, search, questions, and citations;
- a reproducible local environment, automated tests, and continuous integration.

## Not Included in v0.1.0

The following are intentionally outside the MVP scope:

- production-ready private-repository OAuth or GitHub App integration;
- perfect support for every programming language;
- compiler-grade whole-program analysis or a complete call graph;
- guarantees of exact dependency or change-impact analysis for dynamic languages;
- autonomous code modification or execution of analyzed repository code;
- an IDE extension;
- mandatory hosted AI or paid API services;
- a managed commercial vector database;
- large-scale distributed search infrastructure;
- Kubernetes or production cloud deployment;
- multi-agent investigation workflows;
- organization-wide permissions, collaboration, or enterprise administration.

These boundaries keep the first release focused on a small, correct, explainable code-intelligence workflow.

## Local-First and Free-First

RepoMind must remain usable without paid AI APIs. The planned default configuration will use local Sentence Transformers embeddings and a local language model through Ollama.

Embedding and language-model integrations will be defined behind provider-independent interfaces. This is intended to allow models or providers to change later without coupling the rest of the system to one vendor. A hosted provider may become an optional configuration, but it will not be required for the core MVP.

The core MVP will not require repository content to be sent to a hosted model; hosted providers may be supported later as an optional configuration.

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
- review recorded retrieval-quality measurements from a reproducible evaluation set.

Success will be based on working, testable behavior and valid evidence—not on unsupported claims, invented benchmarks, or the apparent confidence of generated text.

## Technology Direction

The following technologies represent the planned direction for the MVP. Their presence here does not mean they have already been integrated.

- **Python and FastAPI:** backend application and HTTP API.
- **PostgreSQL:** durable storage for repositories, files, code units, jobs, relationships, citations, and evaluation data.
- **pgvector:** vector storage and semantic similarity search within PostgreSQL.
- **Tree-sitter:** syntax-aware parsing and source-range extraction.
- **Sentence Transformers:** local embedding generation.
- **Ollama:** default local language-model runtime.
- **Redis:** later-stage caching and background-job coordination where needed.
- **Next.js and React:** planned browser interface.
- **Docker Compose:** reproducible local services and development environment.
- **GitHub Actions:** automated quality and test checks.
- **Prometheus and Grafana:** later-stage application metrics and local observability.

Technologies will be introduced only when their roadmap stage requires them. The project will prefer the smallest correct implementation over premature infrastructure.

## Safety and Trust Principles

RepoMind is planned around several non-negotiable principles:

- Never execute code from an analyzed repository.
- Treat source files, documentation, and repository instructions as untrusted data.
- Keep retrieved evidence separate from system instructions.
- Ground generated answers in retrieved repository evidence.
- Preserve file paths, symbols, and line ranges throughout indexing and retrieval.
- Make retrieval testable without relying on an LLM.
- Report insufficient evidence rather than fabricate an answer.
- Describe dependency and impact analysis as approximate where language behavior is dynamic.
- Keep secrets, cloned repositories, database data, model files, and generated caches out of version control.

## Project Status

RepoMind is under active development. The current stage defines the product charter, scope, constraints, and success criteria for v0.1.0. Application implementation will proceed incrementally in later development stages, and this README will be updated as capabilities become available and verifiable.
