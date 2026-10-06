import { getRepositoryIndexingSummaries } from "@/lib/api";

export async function GET(request: Request): Promise<Response> {
  const query = new URL(request.url).searchParams;
  const ids = query.getAll("repository_id");
  const headers = { "Cache-Control": "no-store" };
  if ([...query.keys()].some((key) => key !== "repository_id") || ids.length < 1 || ids.length > 100 ||
      ids.some((id) => !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu.test(id))) {
    return Response.json({ error: "invalid" }, { status: 400, headers });
  }
  const result = await getRepositoryIndexingSummaries(ids);
  if (!result.ok) {
    return Response.json({ error: result.error === "not_found" ? "not_found" : "unavailable" }, {
      status: result.error === "not_found" ? 404 : 503, headers,
    });
  }
  return Response.json({ items: result.items }, { headers });
}
