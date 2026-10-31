// Proxies /api/* to `pipeline serve`, passing the NDJSON stream of /api/ask through unbuffered.

const UPSTREAM = process.env.PIPELINE_API_URL ?? "http://127.0.0.1:8765";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type Ctx = { params: Promise<{ path: string[] }> };

async function proxy(req: Request, { params }: Ctx) {
  const { path } = await params;
  const url = `${UPSTREAM}/api/${path.map(encodeURIComponent).join("/")}`;
  try {
    const upstream = await fetch(url, {
      method: req.method,
      headers: { "Content-Type": req.headers.get("content-type") ?? "application/json" },
      body: req.method === "POST" ? await req.text() : undefined,
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
      { error: `pipeline API unreachable at ${UPSTREAM}. Start it with: uv run --extra retrieval pipeline serve` },
      { status: 502 },
    );
  }
}

export { proxy as GET, proxy as POST };
