from __future__ import annotations

import pytest
from app.clients.backend_mcp import BackendMCPClient


def test_backend_mcp_client_restricts_unknown_tools() -> None:
    client = BackendMCPClient()
    with pytest.raises(ValueError, match="Backend MCP tool is not allowed"):
        client.call_tool("unauthorized_mutation_tool")
