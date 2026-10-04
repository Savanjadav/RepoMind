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
