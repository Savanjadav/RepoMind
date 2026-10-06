import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import { afterEach, beforeEach, test } from "node:test";

// Node executes TypeScript directly; tsc resolves the same module without suffix.
const modulePath = "./api.ts";
const api: typeof import("./api") = await import(modulePath);
// Resolve Next's app alias for Node's native runner; still import the real
// action and GET handler, without copying their validation into the tests.
const hook = registerHooks({ resolve(specifier, context, nextResolve) {
  if (specifier === "@/lib/api") return { url: new URL("./api.ts", import.meta.url).href, shortCircuit: true };
  return nextResolve(specifier === "next/cache" ? "next/cache.js" : specifier, context);
} });
const actionPath = "../app/actions.ts";
const routePath = "../app/api/indexing-summary/route.ts";
const actions: typeof import("../app/actions") = await import(actionPath);
const route: typeof import("../app/api/indexing-summary/route") = await import(routePath);
hook.deregister();
const originalFetch = globalThis.fetch;
const originalUrl = process.env.REPOMIND_API_BASE_URL;
const repository = {
  id: "c3ea5adb-9342-4f93-b3be-94ef4c99a100", name: "repo",
  source: "https://github.com/owner/repo", created_at: "2026-10-03T12:00:00Z",
};
let calls: { url: string; options?: RequestInit }[];

beforeEach(() => {
  calls = [];
  process.env.REPOMIND_API_BASE_URL = "http://127.0.0.1:8000";
});
afterEach(() => {
  globalThis.fetch = originalFetch;
  if (originalUrl === undefined) delete process.env.REPOMIND_API_BASE_URL;
  else process.env.REPOMIND_API_BASE_URL = originalUrl;
});
function respond(body: unknown, status = 200) {
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), options });
    return Response.json(body, { status });
  };
}

test("registration sends source as data to a fixed endpoint, once", async () => {
  respond(repository, 201);
  const source = "https://attacker.invalid/do-not-fetch";
  assert.deepEqual(await api.registerRepository(source), { ok: true, repository });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://127.0.0.1:8000/repositories");
  assert.equal(calls[0].options?.method, "POST");
  assert.deepEqual(JSON.parse(String(calls[0].options?.body)), { source });
  assert.deepEqual(calls[0].options?.headers, { "Content-Type": "application/json" });
  assert.equal(calls[0].options?.cache, "no-store");
  assert.equal(calls[0].options?.redirect, "error");
  assert(calls[0].options?.signal instanceof AbortSignal);
});
test("list uses bounded query and validates metadata", async () => {
  respond({ items: [repository], limit: 2, offset: 3 });
  assert.deepEqual(await api.listRepositories(2, 3), { ok: true, items: [repository] });
  assert.equal(calls[0].url, "http://127.0.0.1:8000/repositories?limit=2&offset=3");
  assert.equal(calls[0].options?.method, "GET");
  assert.equal(calls[0].options?.body, undefined);
});
test("empty list is successful, not an unavailable service", async () => {
  respond({ items: [], limit: 100, offset: 0 });
  assert.deepEqual(await api.listRepositories(), { ok: true, items: [] });
});
for (const body of [null, [], {}, { ...repository, id: "bad" },
  { ...repository, name: 5 }, { ...repository, source: null },
  { ...repository, created_at: "yesterday" }]) {
  test(`invalid registration DTO ${JSON.stringify(body)}`, async () => {
    respond(body, 201);
    assert.deepEqual(await api.registerRepository(repository.source), { ok: false, error: "uncertain" });
  });
}
for (const body of [null, [], {}, { items: [null], limit: 100, offset: 0 },
  { items: [], limit: 99, offset: 0 }, { items: [], limit: 100, offset: 1 }]) {
  test(`invalid list DTO ${JSON.stringify(body)}`, async () => {
    respond(body);
    assert.deepEqual(await api.listRepositories(), { ok: false, error: "response" });
  });
}
for (const [status, error] of [[409, "duplicate"], [422, "invalid"], [500, "uncertain"], [503, "uncertain"], [200, "uncertain"]] as const) {
  test(`registration HTTP ${status} is bounded`, async () => {
    respond({ detail: "SECRET DATABASE" }, status);
    assert.deepEqual(await api.registerRepository(repository.source), { ok: false, error });
    assert.equal(calls.length, 1);
  });
}
test("malformed JSON cannot claim mutation failure or expose body", async () => {
  globalThis.fetch = async () => new Response("SECRET", { status: 201 });
  assert.deepEqual(await api.registerRepository(repository.source), { ok: false, error: "uncertain" });
});
for (const error of [new TypeError("SECRET"), new DOMException("SECRET", "TimeoutError")]) {
  test(`POST ${error.name} is uncertain and never retried`, async () => {
    let count = 0;
    globalThis.fetch = async () => { count++; throw error; };
    assert.deepEqual(await api.registerRepository(repository.source), { ok: false, error: "uncertain" });
    assert.equal(count, 1);
    assert.deepEqual(await api.listRepositories(), { ok: false, error: "unavailable" });
    assert.equal(count, 2);
  });
}
test("programming errors propagate", async () => {
  const error = new Error("unexpected");
  globalThis.fetch = async () => { throw error; };
  await assert.rejects(api.listRepositories, (actual) => actual === error);
});
test("invalid configuration makes no request", async () => {
  process.env.REPOMIND_API_BASE_URL = "http://user:SECRET@example.com";
  respond(repository, 201);
  assert.deepEqual(await api.registerRepository(repository.source), { ok: false, error: "configuration" });
  assert.equal(calls.length, 0);
});
test("invalid list bounds make no request", async () => {
  respond({});
  await assert.rejects(() => api.listRepositories(101), RangeError);
  await assert.rejects(() => api.listRepositories(1, -1), RangeError);
  assert.equal(calls.length, 0);
});
test("health behavior remains server-side and uncached", async () => {
  respond({ status: "ok" });
  assert.deepEqual(await api.getBackendHealth(), { state: "connected" });
  assert.equal(calls[0].url, "http://127.0.0.1:8000/health");
  assert.equal(calls[0].options?.cache, "no-store");
});

const rid = repository.id;
const otherId = "e3ea5adb-9342-4f93-b3be-94ef4c99a101";
const jobId = "a3ea5adb-9342-4f93-b3be-94ef4c99a102";
const accepted = { repository_id: rid, job_id: jobId, status: "pending" };
const completed = {
  repository_id: rid,
  latest_job: { job_id: jobId, status: "completed", created_at: repository.created_at },
  snapshot_counts: { files: 1, code_units: 5 },
};

test("index start uses fixed bodyless POST and validated accepted job", async () => {
  respond({ ...accepted, secret: "not forwarded" }, 202);
  assert.deepEqual(await api.startRepositoryIndexing(rid.toUpperCase()), { ok: true, ...accepted });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `http://127.0.0.1:8000/repositories/${rid}/index`);
  assert.equal(calls[0].options?.method, "POST");
  assert.equal(calls[0].options?.body, undefined);
  assert.equal(calls[0].options?.cache, "no-store");
  assert.equal(calls[0].options?.redirect, "error");
  assert(calls[0].options?.signal instanceof AbortSignal);
});
for (const id of ["", "../health", "?x=1", "#fragment", "not-a-uuid", `${rid}/index`]) {
  test(`invalid indexing UUID ${id} never fetches`, async () => {
    respond({});
    assert.deepEqual(await api.startRepositoryIndexing(id), { ok: false, error: "invalid" });
    assert.deepEqual(await api.getRepositoryIndexingSummaries([id]), { ok: false, error: "invalid" });
    assert.equal(calls.length, 0);
  });
}
for (const [status, error] of [[409, "conflict"], [404, "not_found"], [422, "invalid"], [500, "uncertain"], [200, "uncertain"]] as const) {
  test(`index start HTTP ${status}`, async () => {
    respond({ detail: "SECRET" }, status);
    assert.deepEqual(await api.startRepositoryIndexing(rid), { ok: false, error });
    assert.equal(calls.length, 1);
  });
}
for (const body of [null, {}, { ...accepted, repository_id: otherId }, { ...accepted, job_id: "bad" }, { ...accepted, status: "running" }]) {
  test(`invalid accepted job ${JSON.stringify(body)}`, async () => {
    respond(body, 202);
    assert.deepEqual(await api.startRepositoryIndexing(rid), { ok: false, error: "uncertain" });
    assert.equal(calls.length, 1);
  });
}
for (const error of [new TypeError("SECRET"), new DOMException("SECRET", "TimeoutError")]) {
  test(`indexing ${error.name} is uncertain for mutation, unavailable for read`, async () => {
    let count = 0;
    globalThis.fetch = async () => { count++; throw error; };
    assert.deepEqual(await api.startRepositoryIndexing(rid), { ok: false, error: "uncertain" });
    assert.equal(count, 1);
    assert.deepEqual(await api.getRepositoryIndexingSummaries([rid]), { ok: false, error: "unavailable" });
    assert.equal(count, 2);
  });
}
test("summary request deduplicates bounded UUIDs and projects metadata", async () => {
  respond({ items: [{ ...completed, secret: "not forwarded" }] });
  assert.deepEqual(await api.getRepositoryIndexingSummaries([rid.toUpperCase(), rid]), { ok: true, items: [completed] });
  assert.equal(calls[0].url, `http://127.0.0.1:8000/repositories/indexing-summary?repository_id=${rid}`);
  assert.equal(calls[0].options?.cache, "no-store");
  assert.equal(calls[0].options?.redirect, "error");
  assert(calls[0].options?.signal instanceof AbortSignal);
});
for (const status of [null, "pending", "running", "failed", "completed"]) {
  test(`summary accepts ${status} with correct counts semantics`, async () => {
    const item = {
      repository_id: rid,
      latest_job: status === null ? null : { ...completed.latest_job, status },
      snapshot_counts: status === "completed" ? { files: 0, code_units: 0 } : null,
    };
    respond({ items: [item] });
    assert.deepEqual(await api.getRepositoryIndexingSummaries([rid]), { ok: true, items: [item] });
  });
}
const invalidSummaries = [
  null, {}, { items: [] }, { items: [null] },
  { items: [{ ...completed, repository_id: otherId }] },
  ...["unknown", ["running"], null].map((status) => ({ items: [{ ...completed, latest_job: { ...completed.latest_job, status } }] })),
  ...[-1, 1.5, "1", true].map((files) => ({ items: [{ ...completed, snapshot_counts: { files, code_units: 5 } }] })),
  { items: [{ ...completed, snapshot_counts: { files: 1, code_units: -1 } }] },
  { items: [{ ...completed, snapshot_counts: null }] },
  { items: [{ ...completed, latest_job: null }] },
  { items: [{ ...completed, latest_job: { ...completed.latest_job, created_at: "yesterday" } }] },
  { items: [{ ...completed, latest_job: { ...completed.latest_job, status: "running" } }] },
];
invalidSummaries.forEach((body, index) => {
  test(`reject malformed summary ${index}`, async () => {
    respond(body);
    assert.deepEqual(await api.getRepositoryIndexingSummaries([rid]), { ok: false, error: "response" });
  });
});
test("summary rejects duplicate and missing requested associations", async () => {
  respond({ items: [completed, completed] });
  assert.deepEqual(await api.getRepositoryIndexingSummaries([rid, otherId]), { ok: false, error: "response" });
  respond({ items: [completed] });
  assert.deepEqual(await api.getRepositoryIndexingSummaries([rid, otherId]), { ok: false, error: "response" });
});
test("summary input bound applies before deduplication", async () => {
  respond({});
  for (const ids of [[], Array(101).fill(rid)]) {
    assert.deepEqual(await api.getRepositoryIndexingSummaries(ids), { ok: false, error: "invalid" });
  }
  assert.equal(calls.length, 0);
});
test("summary missing repository remains a bounded not-found result", async () => {
  respond({ detail: "SECRET" }, 404);
  assert.deepEqual(await api.getRepositoryIndexingSummaries([rid]), { ok: false, error: "not_found" });
});
test("indexing programming exceptions are not hidden", async () => {
  const error = new Error("unexpected");
  globalThis.fetch = async () => { throw error; };
  await assert.rejects(() => api.startRepositoryIndexing(rid), (value) => value === error);
  await assert.rejects(() => api.getRepositoryIndexingSummaries([rid]), (value) => value === error);
});

for (const mode of ["missing", "duplicate", "file", "extra", "path"]) {
  test(`start action rejects ${mode} FormData before fetch`, async () => {
    const form = new FormData();
    if (mode !== "missing") form.append("repository_id", mode === "file" ? new File([rid], "id.txt") : mode === "path" ? "../health" : rid);
    if (mode === "duplicate") form.append("repository_id", rid);
    if (mode === "extra") form.append("source", "https://attacker.invalid");
    respond({});
    assert.deepEqual(await actions.startIndexing(form), { status: "error", message: "Invalid repository identifier." });
    assert.equal(calls.length, 0);
  });
}
test("real start action returns bounded accepted and uncertain states", async () => {
  const form = new FormData(); form.append("repository_id", rid);
  respond(accepted, 202);
  assert.deepEqual(await actions.startIndexing(form), { status: "accepted", jobId, message: "Indexing request accepted." });
  globalThis.fetch = async () => { throw new TypeError("SECRET"); };
  const result = await actions.startIndexing(form);
  assert.equal(result.status, "uncertain");
  assert.match(result.message, /outcome is uncertain/u);
  assert(!JSON.stringify(result).includes("SECRET"));
});
for (const query of ["", "repository_id=../health", "repository_id=%23fragment", `repository_id=${rid}&url=https://attacker.invalid`, Array(101).fill(`repository_id=${rid}`).join("&")]) {
  test(`GET handler rejects query ${query.slice(0, 60)}`, async () => {
    respond({});
    const response = await route.GET(new Request(`http://localhost/api/indexing-summary?${query}`));
    assert.equal(response.status, 400);
    assert.equal(response.headers.get("Cache-Control"), "no-store");
    assert.equal(calls.length, 0);
  });
}
test("real GET handler projects validated data and sanitizes failures", async () => {
  const request = new Request(`http://localhost/api/indexing-summary?repository_id=${rid}`);
  respond({ items: [{ ...completed, secret: "SECRET" }] });
  const response = await route.GET(request);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.deepEqual(await response.json(), { items: [completed] });
  respond({ detail: "SECRET" }, 404);
  const missing = await route.GET(request);
  assert.equal(missing.status, 404);
  assert.deepEqual(await missing.json(), { error: "not_found" });
});
