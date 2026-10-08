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
const askRoutePath = "../app/api/ask/route.ts";
const askRoute: typeof import("../app/api/ask/route") = await import(askRoutePath);
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

const citation = { evidence_id: 1, repository_name: "repo", path: "a.py", symbol_name: null, start_line: 1, end_line: 2 };
const answer = { answer: "  Answer [Evidence 1]\n", citations: [citation] };
test("ask uses exact question and fixed POST without retrieval controls", async () => {
  const q = "  Unicode 😀\t\r\nquestion  ";
  respond({ ...answer, secret: "SECRET" });
  assert.deepEqual(await api.askRepository(rid, q), { ok: true, data: answer });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://127.0.0.1:8000/ask");
  assert.equal(calls[0].options?.method, "POST");
  assert.deepEqual(JSON.parse(String(calls[0].options?.body)), { repository_id: rid, q });
  assert.equal(calls[0].options?.cache, "no-store");
  assert.equal(calls[0].options?.redirect, "error");
  assert.deepEqual(calls[0].options?.headers, { "Content-Type": "application/json" });
});
for (const q of ["", " \t\r\n", "😀".repeat(2001), ...Array.from({ length: 32 }, (_, i) => i).filter(i => ![9,10,13].includes(i)).map(i => `a${String.fromCharCode(i)}b`), "a\x7fb"]) {
  test(`ask rejects question ${JSON.stringify(q).slice(0,40)}`, async () => {
    respond(answer);
    assert.deepEqual(await api.askRepository(rid, q), { ok: false, error: "invalid" });
    assert.equal(calls.length, 0);
  });
}
test("ask accepts 2000 Unicode code points and rejects injected UUID", async () => {
  respond(answer);
  assert.equal((await api.askRepository(rid, "😀".repeat(2000))).ok, true);
  assert.deepEqual(await api.askRepository(`${rid}/ask`, "q"), { ok: false, error: "invalid" });
  assert.equal(calls.length, 1);
});
test("ask whitespace validation matches Python NEL/BOM behavior", async () => {
  respond(answer);
  assert.deepEqual(await api.askRepository(rid, "\u0085"), { ok: false, error: "invalid" });
  assert.equal((await api.askRepository(rid, "\ufeff")).ok, true);
  assert.equal(calls.length, 1);
});
for (const text of ["", "   ", "The available evidence is insufficient to answer the question.", "<script>alert(1)</script>"]) {
  test(`ask preserves answer ${text}`, async () => {
    respond({ answer: text, citations: [] });
    assert.deepEqual(await api.askRepository(rid, "q"), { ok: true, data: { answer: text, citations: [] } });
  });
}
for (const malformed of [null, [], {}, { ...answer, answer: 1 }, { ...answer, citations: null },
  ...[{ ...citation, evidence_id: 0 }, { ...citation, evidence_id: 11 }, { ...citation, evidence_id: 1.5 },
    { ...citation, repository_name: "" }, { ...citation, path: "" }, { ...citation, symbol_name: 1 },
    { ...citation, start_line: 0 }, { ...citation, end_line: 0 }, { ...citation, end_line: 1.2 }].map(c => ({ ...answer, citations: [c] })),
  { ...answer, citations: [citation, citation] }]) {
  test(`ask rejects malformed response ${JSON.stringify(malformed)}`, async () => {
    respond(malformed);
    assert.deepEqual(await api.askRepository(rid, "q"), { ok: false, error: "response" });
  });
}
test("ask preserves citation order and projects only approved metadata", async () => {
  const citations = [{ ...citation, evidence_id: 2 }, citation];
  respond({ answer: "q", citations: citations.map(c => ({ ...c, secret: "SECRET" })) });
  assert.deepEqual(await api.askRepository(rid, "q"), { ok: true, data: { answer: "q", citations } });
});
for (const [status, error] of [[404,"not_found"],[422,"invalid"],[502,"provider"],[503,"unavailable"],[504,"timeout"],[500,"failed"]] as const) {
  test(`ask maps ${status} without retry or raw error`, async () => {
    respond({ detail: "SECRET" }, status);
    assert.deepEqual(await api.askRepository(rid, "q"), { ok: false, error });
    assert.equal(calls.length, 1);
  });
}
test("ask caller abort works before fetch and during body consumption", async () => {
  respond(answer);
  const controller = new AbortController(); controller.abort();
  assert.deepEqual(await api.askRepository(rid, "q", controller.signal), { ok: false, error: "cancelled" });
  assert.equal(calls.length, 0);
  const active = new AbortController(); let cancelled = false;
  globalThis.fetch = async () => new Response(new ReadableStream({ start() { queueMicrotask(() => active.abort()); }, cancel() { cancelled = true; } }));
  assert.deepEqual(await api.askRepository(rid, "q", active.signal), { ok: false, error: "cancelled" });
  assert.equal(active.signal.aborted, true);
  assert.equal(cancelled, true);
});
test("bounded JSON reader cancels a pending read and releases its lock", async () => {
  const controller = new AbortController(); let cancelled = false;
  const stream = new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } });
  const reading = api.readBoundedJson(stream, 1024, controller.signal);
  controller.abort();
  await assert.rejects(reading, (error: unknown) => error === controller.signal.reason);
  assert.equal(cancelled, true); assert.equal(stream.locked, false);
});
test("bounded JSON reader counts streamed chunks and rejects rather than truncates", async () => {
  let cancelled = false;
  const stream = new ReadableStream<Uint8Array>({ start(controller) {
    controller.enqueue(new TextEncoder().encode('{"x":'));
    controller.enqueue(new TextEncoder().encode('"123456"}'));
  }, cancel() { cancelled = true; } });
  await assert.rejects(api.readBoundedJson(stream, 10), api.JsonBodyError);
  assert.equal(cancelled, true); assert.equal(stream.locked, false);
});
test("ask downstream timeout is 180 seconds and timer is cleared", async () => {
  const originalSet = globalThis.setTimeout, originalClear = globalThis.clearTimeout;
  let expire: (() => void) | undefined, delay: number | undefined, cleared = false;
  globalThis.setTimeout = ((fn: () => void, ms: number) => { expire = fn; delay = ms; return 1; }) as unknown as typeof setTimeout;
  globalThis.clearTimeout = (() => { cleared = true; }) as typeof clearTimeout;
  globalThis.fetch = async (_url, options) => { expire?.(); options?.signal?.throwIfAborted(); return Response.json(answer); };
  try {
    assert.deepEqual(await api.askRepository(rid, "q"), { ok: false, error: "timeout" });
    assert.equal(delay, 180000); assert(cleared);
  } finally { globalThis.setTimeout = originalSet; globalThis.clearTimeout = originalClear; }
});
test("ask bounds actual response bytes and rejects malformed JSON", async () => {
  for (const body of ["x".repeat(1024*1024+1), "not JSON"]) {
    globalThis.fetch = async () => new Response(body);
    assert.deepEqual(await api.askRepository(rid, "q"), { ok: false, error: "response" });
  }
  globalThis.fetch = async () => { throw new TypeError("SECRET"); };
  assert.deepEqual(await api.askRepository(rid, "q"), { ok: false, error: "network" });
});

function askRequest(body: string, headers: Record<string,string> = {}) {
  return new Request("http://localhost/api/ask", { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body });
}
for (const body of ["bad JSON", "null", "[]", ...[{}, { q:"q" }, { repository_id:rid }, { repository_id:"../ask",q:"q" },
  ...["", " ", "😀".repeat(2001), "x\0y"].map(q=>({ repository_id:rid,q })),
  { repository_id:rid,q:"q",url:"https://attacker.invalid" }].map(v=>JSON.stringify(v))]) {
  test(`ask route rejects ${body.slice(0,60)}`, async () => {
    respond(answer);
    const response=await askRoute.POST(askRequest(body));
    assert.equal(response.status,400); assert.equal(calls.length,0);
    assert.equal(response.headers.get("Cache-Control"),"no-store");
  });
}
for (const headers of [{}, { "Content-Length":"1" }, { "Content-Length":"invalid" }] as Record<string,string>[]) {
  test(`ask route enforces actual body bound ${JSON.stringify(headers)}`, async () => {
    respond(answer);
    assert.equal((await askRoute.POST(askRequest(" ".repeat(32769),headers))).status,400);
    assert.equal(calls.length,0);
  });
}
test("ask route content type, origin, projection, errors and no-store", async () => {
  const body=JSON.stringify({repository_id:rid,q:"q"});
  respond(answer);
  assert.equal((await askRoute.POST(askRequest(body,{"Content-Type":"text/plain"}))).status,415);
  assert.equal((await askRoute.POST(askRequest(body,{Origin:"http://attacker.invalid","Sec-Fetch-Site":"cross-site"}))).status,403);
  assert.equal((await askRoute.POST(askRequest(body,{"Sec-Fetch-Site":"same-site"}))).status,403);
  assert.equal((await askRoute.POST(askRequest(body,{"Content-Length":"32769"}))).status,413);
  assert.equal(calls.length,0);
  for (const headers of [{}, { Origin:"http://localhost", "Content-Type":"application/json; charset=utf-8" }] as Record<string,string>[]) {
    const response=await askRoute.POST(askRequest(body,headers));
    assert.equal(response.status,200); assert.deepEqual(await response.json(),answer);
    assert.equal(response.headers.get("Cache-Control"),"no-store");
  }
  respond({ detail:"SECRET" },503);
  const response=await askRoute.POST(askRequest(body));
  assert.equal(response.status,503); assert.deepEqual(await response.json(),{error:"unavailable"});
});
