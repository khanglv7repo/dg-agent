from __future__ import annotations

import pytest
from app.clients.backend_mcp import ALLOWED_BACKEND_TOOLS, BackendMCPClient
from app.gateways.governance import GovernanceGateway


def test_backend_mcp_client_restricts_unknown_tools() -> None:
    client = BackendMCPClient()
    with pytest.raises(ValueError, match="Backend MCP tool is not allowed"):
        client.call_tool("unauthorized_mutation_tool")


def test_backend_mcp_client_does_not_invent_r5_policy_mutation_tools() -> None:
    client = BackendMCPClient()
    # R5 tools such as create_policy_version, activate_policy_version must not be allowed in current client boundary
    assert "create_policy_version" not in ALLOWED_BACKEND_TOOLS
    assert "activate_policy_version" not in ALLOWED_BACKEND_TOOLS

    with pytest.raises(ValueError, match="Backend MCP tool is not allowed"):
        client.call_tool("create_policy_version", {"proposal": {}})


def test_governance_gateway_wraps_verified_backend_mcp_read_tools(monkeypatch) -> None:
    mock_client = pytest.importorskip("unittest.mock").MagicMock(spec=BackendMCPClient)
    mock_client.inspect_ranger_state.return_value = {"status": "ok", "services": ["trino"]}
    mock_client.query_trino_readonly.return_value = {"rows": [["test_col"]]}

    gateway = GovernanceGateway(client=mock_client)

    ranger_res = gateway.inspect_ranger_state(service_name="trino")
    assert ranger_res == {"status": "ok", "services": ["trino"]}
    mock_client.inspect_ranger_state.assert_called_once_with(service_name="trino")

    trino_res = gateway.query_trino_readonly(query="SELECT 1", username="alice", limit=10)
    assert trino_res == {"rows": [["test_col"]]}
    mock_client.query_trino_readonly.assert_called_once_with(
        query="SELECT 1", username="alice", limit=10
    )
