"""Client for the Backend FastMCP server.

Per 05-ai-agent-and-mcp.md, the AI Agent connects to Backend FastMCP over the internal
Docker network (or localhost in dev) to access controlled read-only capabilities:
- Ranger state inspection (`inspect_ranger_state`)
- Bounded read-only Trino diagnostics (`query_trino_readonly`)
"""
from __future__ import annotations

import itertools
from typing import Any

import httpx

ALLOWED_BACKEND_TOOLS = frozenset(
    {
        "inspect_ranger_state",
        "query_trino_readonly",
    }
)


class BackendMCPClient:
    """Client for Backend FastMCP server."""

    def __init__(
        self,
        *,
        endpoint: str = "http://localhost:8000/mcp",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.client = client or httpx.Client(timeout=timeout)
        self._ids = itertools.count(1)

    def close(self) -> None:
        self.client.close()

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if name not in ALLOWED_BACKEND_TOOLS:
            raise ValueError(f"Backend MCP tool is not allowed: {name}")

        payload = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        }
        response = self.client.post(self.endpoint, json=payload)
        response.raise_for_status()
        body = response.json()
        if body.get("error"):
            raise RuntimeError(f"Backend MCP error: {body['error']}")
        return body.get("result")

    def inspect_ranger_state(self, service_name: str | None = None) -> dict[str, Any]:
        return self.call_tool("inspect_ranger_state", {"service_name": service_name})

    def query_trino_readonly(
        self, query: str, username: str = "alice", limit: int = 50
    ) -> dict[str, Any]:
        return self.call_tool(
            "query_trino_readonly",
            {"query": query, "username": username, "limit": limit},
        )
