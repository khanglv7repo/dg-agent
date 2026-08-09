"""Governance Gateway for Backend MCP capabilities.

Per R6-A requirements:
- Exposes verified Backend MCP read capabilities (`inspect_ranger_state`, `query_trino_readonly`).
- MUST NOT invent or call nonexistent Backend R5 policy tools.
- Agent never communicates directly with Ranger or Trino administrative ports or credentials.
"""
from __future__ import annotations

import logging
from typing import Any

from app.clients.backend_mcp import BackendMCPClient

logger = logging.getLogger(__name__)


class GovernanceGateway:
    """Gateway facing the Backend FastMCP server for governance inspection and diagnostics."""

    def __init__(
        self,
        *,
        endpoint: str = "http://localhost:8000/mcp",
        timeout: float = 30.0,
        client: BackendMCPClient | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.client = client or BackendMCPClient(endpoint=endpoint, timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def inspect_ranger_state(self, service_name: str | None = None) -> dict[str, Any]:
        """Inspect Ranger state via Backend MCP read capability."""
        return self.client.inspect_ranger_state(service_name=service_name)

    def query_trino_readonly(
        self,
        query: str,
        username: str = "alice",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Execute bounded read-only Trino diagnostic query via Backend MCP capability."""
        return self.client.query_trino_readonly(
            query=query,
            username=username,
            limit=limit,
        )
