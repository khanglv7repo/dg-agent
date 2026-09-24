from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastmcp import FastMCP

from app.clients.backend_mcp import (
    EXPECTED_BACKEND_TOOLS,
    BackendMCPClient,
)
from app.gateways.governance import GovernanceGateway


def _server() -> FastMCP:
    mcp = FastMCP("Bounded Agent Contract Test")

    @mcp.tool
    def get_policy(policy_key: str, version: int | None = None) -> dict:
        return {"policy_key": policy_key, "version": version or 1}

    @mcp.tool
    def list_policy_versions(policy_key: str) -> list[dict]:
        return [{"policy_key": policy_key, "version": 1}]

    @mcp.tool
    def preview_policy_change(policy_key: str, logical_policy: dict) -> dict:
        return {"policy_key": policy_key, "logical_policy": logical_policy}

    @mcp.tool
    def check_policy_conflict(policy_key: str, logical_policy: dict) -> dict:
        return {"conflict": False}

    @mcp.tool
    def resolve_resource_mapping(om_service_name: str, environment: str) -> dict:
        return {"om_service_name": om_service_name, "environment": environment}

    @mcp.tool
    def get_ranger_sync_status(policy_key: str, version: int | None = None) -> dict:
        return {"policy_key": policy_key}

    @mcp.tool
    def get_audit_summary(limit: int = 20) -> dict:
        return {"limit": limit}

    @mcp.tool
    def inspect_ranger_state(
        kind: str,
        name: str | None = None,
        policy_key: str | None = None,
    ) -> dict:
        return {"kind": kind, "name": name, "policy_key": policy_key}

    @mcp.tool
    def query_trino_readonly(sql: str) -> dict:
        return {"sql": sql, "rows": [[1]]}

    @mcp.tool
    def create_policy_version(
        policy_key: str,
        logical_policy: dict,
        reason: str | None = None,
    ) -> dict:
        return {
            "policy_key": policy_key,
            "status": "DRAFT",
            "authority_changed": False,
            "dispatched": False,
        }

    @mcp.tool
    def get_tag_sync_observability(entity_type: str, entity_fqn: str) -> dict:
        return {"entity_type": entity_type, "entity_fqn": entity_fqn}

    return mcp


def test_actual_fastmcp_client_validates_bounded_contract() -> None:
    client = BackendMCPClient(source=_server())
    probe = client.validate_contract()
    assert probe["server"]["name"] == "Bounded Agent Contract Test"
    assert [tool["name"] for tool in probe["tools"]] == list(EXPECTED_BACKEND_TOOLS)
    assert client.call_tool("query_trino_readonly", {"sql": "SELECT 1"})["rows"] == [[1]]


def test_authority_mutation_tools_are_rejected_before_network_call() -> None:
    client = BackendMCPClient(source=_server())
    for forbidden in (
        "activate_policy_version",
        "rollback_policy",
        "update_service_mapping",
        "request_ranger_sync",
        "complete_classification_execution",
        "get_workflow_status",
    ):
        with pytest.raises(ValueError, match="not allowed"):
            client.call_tool(forbidden, {})


def test_unknown_tool_is_rejected() -> None:
    client = BackendMCPClient(source=_server())
    with pytest.raises(ValueError, match="not allowed"):
        client.call_tool("invented_tool", {})


def test_gateway_exposes_bounded_signatures() -> None:
    mock = MagicMock(spec=BackendMCPClient)
    mock.call_tool.side_effect = lambda name, args: {"tool": name, "args": args}
    gateway = GovernanceGateway(client=mock)
    logical = {
        "subjects": [{"type": "USER", "name": "alice"}],
        "resource": {"catalog": "financial", "schema": "crm", "table": "customers"},
        "access": {"select": "ALLOW"},
        "masks": {},
        "row_filter": None,
    }

    assert gateway.get_policy("p")["tool"] == "get_policy"
    assert gateway.inspect_ranger_state(kind="health")["tool"] == "inspect_ranger_state"
    assert gateway.query_trino_readonly(sql="SELECT 1")["tool"] == "query_trino_readonly"
    assert gateway.preview_policy_change(
        policy_key="p", logical_policy=logical
    )["tool"] == "preview_policy_change"
    assert gateway.check_policy_conflict(
        policy_key="p", logical_policy=logical
    )["tool"] == "check_policy_conflict"
    assert gateway.create_policy_version(
        policy_key="p", logical_policy=logical
    )["tool"] == "create_policy_version"

    for forbidden in (
        "activate_policy_version",
        "rollback_policy",
        "update_service_mapping",
        "request_ranger_sync",
        "complete_classification_execution",
        "get_workflow_status",
    ):
        assert not hasattr(gateway, forbidden)


def test_important_input_schemas_remain_bounded() -> None:
    probe = BackendMCPClient(source=_server()).validate_contract()
    schemas = {item["name"]: item["input_schema"] for item in probe["tools"]}

    ranger_props = schemas["inspect_ranger_state"]["properties"]
    assert "kind" in ranger_props
    assert "service_name" not in ranger_props

    trino_props = schemas["query_trino_readonly"]["properties"]
    assert set(trino_props) == {"sql"}
