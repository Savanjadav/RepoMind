import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

// Node executes TypeScript directly; tsc resolves the same module without suffix.
const modulePath = "./api.ts";
const api: typeof import("./api") = await import(modulePath);
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
