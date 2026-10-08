import "server-only";

export type HealthResult = {
  state: "connected" | "unavailable" | "configuration" | "response" | "http";
};

function backendOrigin(): string {
  const value = process.env.REPOMIND_API_BASE_URL ?? "http://127.0.0.1:8000";
  // Inspect raw input before URL parsing can normalize controls or path segments.
  if (
    value !== value.trim() ||
    /[\u0000-\u001f\u007f]/u.test(value) ||
    !/^https?:\/\/[^/?#\\]+\/?$/iu.test(value)
  ) {
    throw new TypeError("Invalid backend configuration");
  }
  const url = new URL(value);
  if (url.username || url.password || value.includes("@")) {
    throw new TypeError("Invalid backend configuration");
  }
  return url.origin;
}

function isTransportFailure(error: unknown): boolean {
  return error instanceof TypeError ||
    (error instanceof DOMException && ["TimeoutError", "AbortError"].includes(error.name));
}

export async function getBackendHealth(): Promise<HealthResult> {
  let url: string;
  try {
    url = `${backendOrigin()}/health`;
  } catch (error) {
    if (error instanceof TypeError) return { state: "configuration" };
    throw error;
  }

  let response: Response;
  try {
    response = await fetch(url, {
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
      redirect: "error",
    });
  } catch (error) {
    if (isTransportFailure(error)) return { state: "unavailable" };
    throw error;
  }
  if (!response.ok) return { state: "http" };

  let payload: unknown;
  try {
    payload = await response.json();
  } catch (error) {
    if (error instanceof SyntaxError) return { state: "response" };
    if (isTransportFailure(error)) return { state: "unavailable" };
    throw error;
  }
  if (
    typeof payload !== "object" || payload === null ||
    !("status" in payload) || payload.status !== "ok"
  ) {
    return { state: "response" };
  }
  return { state: "connected" };
}

export type Repository = {
  id: string;
  name: string;
  source: string;
  created_at: string;
};

type RepositoryFailure = {
  ok: false;
  error: "configuration" | "invalid" | "duplicate" | "unavailable" | "uncertain" | "response";
};

function isRepository(value: unknown): value is Repository {
  if (typeof value !== "object" || value === null) return false;
  return "id" in value && typeof value.id === "string" &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu.test(value.id) &&
    "name" in value && typeof value.name === "string" && value.name.length > 0 && value.name.length <= 255 &&
    "source" in value && typeof value.source === "string" && value.source.length > 0 &&
    "created_at" in value && typeof value.created_at === "string" &&
    /^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/u.test(value.created_at) &&
    Number.isFinite(Date.parse(value.created_at));
}

// Only fixed repository endpoints are constructed here; source is payload data.
async function repositoryRequest(
  source: string | undefined, limit: number, offset: number,
): Promise<{ ok: true; data: unknown } | RepositoryFailure> {
  let origin: string;
  try {
    origin = backendOrigin();
  } catch (error) {
    if (error instanceof TypeError) return { ok: false, error: "configuration" };
    throw error;
  }
  const mutation = source !== undefined;
  let response: Response;
  try {
    response = await fetch(
      mutation ? `${origin}/repositories` : `${origin}/repositories?limit=${limit}&offset=${offset}`,
      {
        method: mutation ? "POST" : "GET",
        ...(mutation ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source }) } : {}),
        cache: "no-store",
        redirect: "error",
        signal: AbortSignal.timeout(5_000),
      },
    );
  } catch (error) {
    if (isTransportFailure(error)) return { ok: false, error: mutation ? "uncertain" : "unavailable" };
    throw error;
  }
  if (!response.ok) {
    if (mutation && response.status === 409) return { ok: false, error: "duplicate" };
    if (mutation && response.status === 422) return { ok: false, error: "invalid" };
    return { ok: false, error: mutation ? "uncertain" : "unavailable" };
  }
  if (response.status !== (mutation ? 201 : 200)) {
    return { ok: false, error: mutation ? "uncertain" : "response" };
  }
  try {
    return { ok: true, data: await response.json() };
  } catch (error) {
    if (error instanceof SyntaxError || isTransportFailure(error)) {
      return { ok: false, error: mutation ? "uncertain" : "response" };
    }
    throw error;
  }
}

export async function registerRepository(source: string): Promise<
  { ok: true; repository: Repository } | RepositoryFailure
> {
  const result = await repositoryRequest(source, 100, 0);
  if (!result.ok) return result;
  return isRepository(result.data)
    ? { ok: true, repository: result.data }
    : { ok: false, error: "uncertain" };
}

export async function listRepositories(limit = 100, offset = 0): Promise<
  { ok: true; items: Repository[] } | RepositoryFailure
> {
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 100 ||
      !Number.isSafeInteger(offset) || offset < 0) throw new RangeError("Invalid list bounds");
  const result = await repositoryRequest(undefined, limit, offset);
  if (!result.ok) return result;
  const value = result.data;
  if (typeof value !== "object" || value === null ||
      !("items" in value) || !Array.isArray(value.items) ||
      value.items.length > limit || !value.items.every(isRepository) ||
      !("limit" in value) || value.limit !== limit ||
      !("offset" in value) || value.offset !== offset) {
    return { ok: false, error: "response" };
  }
  return { ok: true, items: value.items };
}

export type IndexingStatus = "pending" | "running" | "completed" | "failed";
export type IndexingSummary = {
  repository_id: string;
  latest_job: { job_id: string; status: IndexingStatus; created_at: string } | null;
  snapshot_counts: { files: number; code_units: number } | null;
};
export type IndexingFailure = {
  ok: false;
  error: "invalid" | "not_found" | "conflict" | "configuration" | "unavailable" | "uncertain" | "response";
};

const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu;
function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function validUuid(value: unknown): value is string {
  return typeof value === "string" && uuid.test(value);
}
function timestamp(value: unknown): value is string {
  return typeof value === "string" &&
    /^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/u.test(value) && Number.isFinite(Date.parse(value));
}
function summary(value: unknown): IndexingSummary | null {
  if (!object(value) || !validUuid(value.repository_id)) return null;
  const job = value.latest_job;
  const counts = value.snapshot_counts;
  if (job === null) {
    return counts === null ? { repository_id: value.repository_id.toLowerCase(), latest_job: null, snapshot_counts: null } : null;
  }
  if (!object(job) || !validUuid(job.job_id) || !timestamp(job.created_at) || typeof job.status !== "string" ||
      !["pending", "running", "completed", "failed"].includes(job.status)) return null;
  let validatedCounts: IndexingSummary["snapshot_counts"] = null;
  if (job.status === "completed") {
    if (!object(counts) || typeof counts.files !== "number" || !Number.isSafeInteger(counts.files) || counts.files < 0 ||
        typeof counts.code_units !== "number" || !Number.isSafeInteger(counts.code_units) || counts.code_units < 0) return null;
    validatedCounts = { files: counts.files, code_units: counts.code_units };
  } else if (counts !== null) return null;
  return {
    repository_id: value.repository_id.toLowerCase(),
    latest_job: { job_id: job.job_id.toLowerCase(), status: job.status as IndexingStatus, created_at: job.created_at },
    snapshot_counts: validatedCounts,
  };
}

export async function startRepositoryIndexing(repositoryId: string): Promise<
  { ok: true; job_id: string; repository_id: string; status: "pending" } | IndexingFailure
> {
  if (!validUuid(repositoryId)) return { ok: false, error: "invalid" };
  const id = repositoryId.toLowerCase();
  let origin: string;
  try { origin = backendOrigin(); }
  catch (error) {
    if (error instanceof TypeError) return { ok: false, error: "configuration" };
    throw error;
  }
  let response: Response;
  try {
    response = await fetch(`${origin}/repositories/${id}/index`, {
      method: "POST", cache: "no-store", redirect: "error", signal: AbortSignal.timeout(5_000),
    });
  } catch (error) {
    if (isTransportFailure(error)) return { ok: false, error: "uncertain" };
    throw error;
  }
  if (response.status === 409) return { ok: false, error: "conflict" };
  if (response.status === 404) return { ok: false, error: "not_found" };
  if (response.status === 422) return { ok: false, error: "invalid" };
  if (response.status !== 202) return { ok: false, error: "uncertain" };
  let value: unknown;
  try { value = await response.json(); }
  catch (error) {
    if (error instanceof SyntaxError || isTransportFailure(error)) return { ok: false, error: "uncertain" };
    throw error;
  }
  if (!object(value) || !validUuid(value.job_id) || !validUuid(value.repository_id) ||
      value.repository_id.toLowerCase() !== id || value.status !== "pending") return { ok: false, error: "uncertain" };
  return { ok: true, job_id: value.job_id.toLowerCase(), repository_id: id, status: "pending" };
}

export async function getRepositoryIndexingSummaries(repositoryIds: string[]): Promise<
  { ok: true; items: IndexingSummary[] } | IndexingFailure
> {
  if (!Array.isArray(repositoryIds) || repositoryIds.length < 1 || repositoryIds.length > 100 ||
      !repositoryIds.every(validUuid)) return { ok: false, error: "invalid" };
  const ids = new Set(repositoryIds.map((id) => id.toLowerCase()));
  const query = new URLSearchParams();
  [...ids].sort().forEach((id) => query.append("repository_id", id));
  let origin: string;
  try { origin = backendOrigin(); }
  catch (error) {
    if (error instanceof TypeError) return { ok: false, error: "configuration" };
    throw error;
  }
  let response: Response;
  try {
    response = await fetch(`${origin}/repositories/indexing-summary?${query}`, {
      cache: "no-store", redirect: "error", signal: AbortSignal.timeout(5_000),
    });
  } catch (error) {
    if (isTransportFailure(error)) return { ok: false, error: "unavailable" };
    throw error;
  }
  if (response.status === 404) return { ok: false, error: "not_found" };
  if (response.status !== 200) return { ok: false, error: "unavailable" };
  let value: unknown;
  try { value = await response.json(); }
  catch (error) {
    if (error instanceof SyntaxError || isTransportFailure(error)) return { ok: false, error: "response" };
    throw error;
  }
  if (!object(value) || !Array.isArray(value.items) || value.items.length !== ids.size) return { ok: false, error: "response" };
  const items: IndexingSummary[] = [];
  for (const raw of value.items) {
    const item = summary(raw);
    if (!item || !ids.delete(item.repository_id)) return { ok: false, error: "response" };
    items.push(item);
  }
  return { ok: true, items };
}

export type AskAnswer = { answer: string; citations: {
  evidence_id: number; repository_name: string; path: string; symbol_name: string | null;
  start_line: number; end_line: number;
}[] };
export type AskError = "invalid" | "not_found" | "unavailable" | "timeout" | "provider" | "network" | "response" | "failed" | "cancelled";

export function validQuestion(value: unknown): value is string {
  // Python str.strip() whitespace differs from JavaScript trim (NEL and BOM).
  return typeof value === "string" && Array.from(value).length >= 1 && Array.from(value).length <= 2000 &&
    /[^\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/u.test(value) &&
    !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/u.test(value);
}

export class JsonBodyError extends Error {}

// Count actual decoded-body bytes, never trust Content-Length or parse a prefix.
// Cancellation also terminates a pending body read; listeners are always removed.
export async function readBoundedJson(body: ReadableStream<Uint8Array> | null, maximum: number, signal?: AbortSignal): Promise<unknown> {
  if (!body) throw new JsonBodyError("Missing JSON body");
  const reader = body.getReader();
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal?.addEventListener("abort", cancel, { once: true });
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    signal?.throwIfAborted();
    while (true) {
      const part = await reader.read();
      signal?.throwIfAborted();
      if (part.done) break;
      size += part.value.byteLength;
      if (size > maximum) { cancel(); throw new JsonBodyError("JSON body exceeds limit"); }
      chunks.push(part.value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    try { return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); }
    catch (error) {
      if (error instanceof SyntaxError || error instanceof TypeError) throw new JsonBodyError("Invalid JSON body");
      throw error;
    }
  } finally {
    if (signal?.aborted) cancel();
    signal?.removeEventListener("abort", cancel);
    reader.releaseLock();
  }
}

function askAnswer(value: unknown): AskAnswer | null {
  if (!object(value) || typeof value.answer !== "string" || !Array.isArray(value.citations) || value.citations.length > 10) return null;
  const seen = new Set<number>();
  const citations: AskAnswer["citations"] = [];
  for (const item of value.citations) {
    if (!object(item) || typeof item.evidence_id !== "number" || !Number.isSafeInteger(item.evidence_id) ||
        item.evidence_id < 1 || item.evidence_id > 10 || seen.has(item.evidence_id) ||
        typeof item.repository_name !== "string" || !item.repository_name.trim() ||
        typeof item.path !== "string" || !item.path.trim() ||
        (item.symbol_name !== null && typeof item.symbol_name !== "string") ||
        typeof item.start_line !== "number" || !Number.isSafeInteger(item.start_line) || item.start_line < 1 ||
        typeof item.end_line !== "number" || !Number.isSafeInteger(item.end_line) || item.end_line < item.start_line) return null;
    seen.add(item.evidence_id);
    citations.push({ evidence_id: item.evidence_id, repository_name: item.repository_name, path: item.path,
      symbol_name: item.symbol_name, start_line: item.start_line, end_line: item.end_line });
  }
  return { answer: value.answer, citations };
}

export async function askRepository(repositoryId: string, question: string, signal?: AbortSignal): Promise<
  { ok: true; data: AskAnswer } | { ok: false; error: AskError }
> {
  if (!validUuid(repositoryId) || !validQuestion(question)) return { ok: false, error: "invalid" };
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(new DOMException("Ask timeout", "TimeoutError")), 180_000);
  const combined = AbortSignal.any(signal ? [signal, timeout.signal] : [timeout.signal]);
  try {
    combined.throwIfAborted();
    const response = await fetch(`${backendOrigin()}/ask`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repository_id: repositoryId.toLowerCase(), q: question }),
      cache: "no-store", redirect: "error", signal: combined,
    });
    if (response.status !== 200) {
      void response.body?.cancel().catch(() => {});
      const mapping: Record<number, AskError> = { 404: "not_found", 422: "invalid", 502: "provider", 503: "unavailable", 504: "timeout" };
      return { ok: false, error: mapping[response.status] ?? "failed" };
    }
    const data = askAnswer(await readBoundedJson(response.body, 1024 * 1024, combined));
    return data ? { ok: true, data } : { ok: false, error: "response" };
  } catch (error) {
    if (signal?.aborted) return { ok: false, error: "cancelled" };
    if (timeout.signal.aborted || (error instanceof DOMException && error.name === "TimeoutError")) return { ok: false, error: "timeout" };
    if (error instanceof JsonBodyError) return { ok: false, error: "response" };
    if (isTransportFailure(error)) return { ok: false, error: "network" };
    throw error;
  } finally { clearTimeout(timer); }
}
