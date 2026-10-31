"""MCP bridge for the persistent RxAI Memory FastAPI service.

This is deliberately a thin client: conversation state and Reactor calls remain
owned by ``pipeline api-serve``.  The bridge exposes that capability through
the standard Streamable HTTP MCP transport at ``/mcp``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server import MCPServer

API_URL = os.environ.get("MEMORY_API_URL", "http://127.0.0.1:8000").rstrip("/")
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8001"))

mcp = MCPServer(
    "RxAI Memory",
    instructions=(
        "Use these tools to inspect or resume benchmark-memory conversations. "
        "Only call answer_dataset_question when the user explicitly asks for an answer: "
        "it invokes Reactor and may take time."
    ),
)


async def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(360.0, connect=15.0)) as client:
            response = await client.request(method, f"{API_URL}{path}", json=payload)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Memory backend unavailable at {API_URL}: {exc}") from exc
    if response.is_error:
        detail = response.json().get("detail", response.text) if response.content else response.reason_phrase
        raise RuntimeError(f"Memory backend returned {response.status_code}: {detail}")
    return response.json()


@mcp.tool()
async def memory_health() -> dict[str, Any]:
    """Return health and configuration of the connected memory backend."""
    return await _request("GET", "/health")


@mcp.tool()
async def list_conversations() -> list[dict[str, Any]]:
    """List saved resumable Reactor conversations, newest first."""
    return await _request("GET", "/v1/conversations")


@mcp.tool()
async def create_conversation(title: str = "") -> dict[str, Any]:
    """Create a persistent Reactor conversation. Use its ID to resume later."""
    return await _request("POST", "/v1/conversations", {"title": title})


@mcp.tool()
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    """Get metadata for one saved conversation."""
    return await _request("GET", f"/v1/conversations/{conversation_id}")


@mcp.tool()
async def get_conversation_answers(conversation_id: str) -> list[dict[str, Any]]:
    """Return all saved answers and validated citation session IDs for a conversation."""
    return await _request("GET", f"/v1/conversations/{conversation_id}/answers")


@mcp.tool()
async def answer_dataset_question(question_id: str, conversation_id: str | None = None) -> dict[str, Any]:
    """Answer one configured dataset question with Reactor.

    With ``conversation_id``, resumes that saved conversation; without it,
    creates an isolated one-off answer. The returned citations are session IDs.
    """
    if conversation_id:
        path = f"/v1/conversations/{conversation_id}/dataset/answers/{question_id}"
    else:
        path = f"/v1/dataset/answers/{question_id}"
    return await _request("POST", path)


def main() -> None:
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)


if __name__ == "__main__":
    main()
