// Same-origin bridge for the persistent FastAPI conversation service. Keep this
// separate from /api/*: that namespace is the live retriever/reranker trace API.

const UPSTREAM = process.env.MEMORY_API_URL ?? "http://127.0.0.1:8000";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type Ctx = { params: Promise<{ path: string[] }> };

async function proxy(req: Request, { params }: Ctx) {
  const { path } = await params;
  const url = `${UPSTREAM}/${path.map(encodeURIComponent).join("/")}`;
  try {
    const upstream = await fetch(url, {
      method: req.method,
      headers: { "Content-Type": req.headers.get("content-type") ?? "application/json" },
      body: req.method === "GET" || req.method === "HEAD" ? undefined : await req.text(),
      cache: "no-store",
      signal: req.signal,
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "application/json",
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    return Response.json(
      { error: `memory API unreachable at ${UPSTREAM}. Start Docker with the api profile.` },
      { status: 502 },
    );
  }
}

export { proxy as GET, proxy as POST };
