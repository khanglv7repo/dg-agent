from __future__ import annotations

import httpx
import pytest
from app.clients.mcp import OpenMetadataMCPClient


def test_mcp_read_only_tool_restriction() -> None:
    client = OpenMetadataMCPClient(endpoint="http://localhost:8585/mcp", token="test-token")
    with pytest.raises(ValueError, match="MCP tool is not allowed"):
        client.call_tool("patch_entity", {"fqn": "test"})


def test_mcp_headers_contain_bearer_token() -> None:
    client = OpenMetadataMCPClient(endpoint="http://localhost:8585/mcp", token="agent-bot-token")
    assert client.headers.get("Authorization") == "Bearer agent-bot-token"
