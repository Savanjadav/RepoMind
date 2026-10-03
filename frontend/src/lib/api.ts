import "server-only";

export type HealthResult = {
  state: "connected" | "unavailable" | "configuration" | "response" | "http";
};

function healthUrl(): string {
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
  return `${url.origin}/health`;
}

function isTransportFailure(error: unknown): boolean {
  return error instanceof TypeError ||
    (error instanceof DOMException && ["TimeoutError", "AbortError"].includes(error.name));
}

export async function getBackendHealth(): Promise<HealthResult> {
  let url: string;
  try {
    url = healthUrl();
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
