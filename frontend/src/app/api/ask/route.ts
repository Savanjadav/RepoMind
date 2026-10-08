import { askRepository, JsonBodyError, readBoundedJson, validQuestion } from "@/lib/api";

export async function POST(request: Request): Promise<Response> {
  const headers = { "Cache-Control": "no-store" };
  const fail = (error: string, status: number) => Response.json({ error }, { status, headers });
  // Next can synthesize request.url from its listening hostname, not the public
  // origin. Avoid a brittle Origin comparison or trusting forwarded hosts.
  // Browser fetch metadata rejects cross-origin requests; JSON + no CORS is the
  // fallback browser boundary when metadata is absent (not authentication).
  const site = request.headers.get("sec-fetch-site");
  if (site !== null && site !== "same-origin" && site !== "none") return fail("invalid", 403);
  if (request.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase() !== "application/json") return fail("invalid", 415);
  const length = request.headers.get("content-length");
  if (length !== null && /^\d+$/u.test(length) && Number(length) > 32 * 1024) return fail("invalid", 413);
  let body: unknown;
  try { body = await readBoundedJson(request.body, 32 * 1024, request.signal); }
  catch (error) {
    if (request.signal.aborted) return fail("cancelled", 499);
    if (error instanceof JsonBodyError) return fail("invalid", 400);
    throw error;
  }
  if (typeof body !== "object" || body === null || Array.isArray(body) ||
      Object.keys(body).length !== 2 || !("repository_id" in body) || !("q" in body) ||
      typeof body.repository_id !== "string" || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu.test(body.repository_id) ||
      !validQuestion(body.q)) return fail("invalid", 400);
  const result = await askRepository(body.repository_id, body.q, request.signal);
  if (result.ok) return Response.json(result.data, { headers });
  const statuses = { invalid: 400, not_found: 404, unavailable: 503, timeout: 504, provider: 502,
    network: 502, response: 502, failed: 500, cancelled: 499 };
  return fail(result.error, statuses[result.error]);
}
