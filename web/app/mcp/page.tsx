export default function McpGuide() {
  return (
    <main className="docs-shell">
      <a className="docs-back" href="/">
        ← Back to Recall
      </a>
      <p className="docs-kicker">MODEL CONTEXT PROTOCOL</p>
      <h1>Connect an AI client to your memory pipeline.</h1>
      <p className="docs-lede">
        The MCP server exposes saved conversations and Reactor-backed dataset answers as standard tools. It returns
        validated session IDs with every answer, so a client can show exactly which memory was used.
      </p>

      <section className="docs-card">
        <h2>Local endpoint</h2>
        <code>http://localhost:8001/mcp</code>
        <p>Start it alongside the application:</p>
        <pre>{`docker compose --profile api --profile mcp up -d --build`}</pre>
      </section>

      <section className="docs-card">
        <h2>Available tools</h2>
        <ul>
          <li><code>memory_health</code> — check the backend.</li>
          <li><code>list_conversations</code> and <code>create_conversation</code> — find or start resumable chats.</li>
          <li><code>get_conversation_answers</code> — inspect saved answers and citations.</li>
          <li><code>answer_dataset_question</code> — ask Reactor a configured dataset question.</li>
        </ul>
      </section>

      <section className="docs-card">
        <h2>Remote endpoint</h2>
        <p>
          For a named Cloudflare Tunnel, add a separate public hostname for MCP and route it to
          <code>http://memory-mcp:8001</code>. Your client then connects to <code>https://mcp.your-domain.example/mcp</code>.
        </p>
        <pre>{`{
  "mcpServers": {
    "rxai-memory": {
      "url": "https://mcp.your-domain.example/mcp"
    }
  }
}`}</pre>
      </section>

      <aside className="docs-warning">
        <strong>Protect the MCP hostname.</strong> The answer tool can read saved conversation data and trigger paid
        provider calls. Put Cloudflare Access or equivalent authentication in front of the MCP hostname before sharing it.
      </aside>
    </main>
  );
}
