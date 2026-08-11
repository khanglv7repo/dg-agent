from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from app.clients.backend_mcp import (
    EXPECTED_BACKEND_TOOLS,
    BackendMCPClient,
    BackendMCPError,
)
from app.gateways.governance import GovernanceGateway


def _server() -> FastMCP:
    mcp = FastMCP("R5 Contract Test")

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
    def get_workflow_status(execution_id: str) -> dict:
        return {"id": execution_id, "status": "WAITING_AI"}

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
    def activate_policy_version(
        policy_key: str,
        version: int,
        confirmed: bool = False,
        approval_reason: str | None = None,
    ) -> dict:
        if not confirmed:
            raise ToolError(json.dumps({
                "ok": False,
                "error": {
                    "code": "CONFIRMATION_REQUIRED",
                    "message": "confirmation required",
                    "retryable": False,
                    "details": {},
                },
            }))
        return {"policy_key": policy_key, "version": version, "status": "ACTIVE"}

    @mcp.tool
    def rollback_policy(
        policy_key: str,
        target_version: int,
        confirmed: bool = False,
        reason: str | None = None,
    ) -> dict:
        return {"policy_key": policy_key, "version": target_version}

    @mcp.tool
    def update_service_mapping(
        om_service_name: str,
        trino_catalog: str,
        ranger_service_name: str,
        environment: str,
        confirmed: bool = False,
        ranger_tag_service_name: str | None = None,
        enabled: bool = True,
        reason: str | None = None,
    ) -> dict:
        return {"om_service_name": om_service_name}

    @mcp.tool
    def request_ranger_sync(policy_key: str) -> dict:
        return {"policy_key": policy_key, "authority_changed": False}

    return mcp


def test_actual_fastmcp_client_initializes_lists_and_calls() -> None:
    client = BackendMCPClient(source=_server())
    probe = client.validate_frozen_contract()
    assert probe["server"]["name"] == "R5 Contract Test"
    assert [tool["name"] for tool in probe["tools"]] == list(EXPECTED_BACKEND_TOOLS)
    assert client.call_tool("query_trino_readonly", {"sql": "SELECT 1"})["rows"] == [[1]]


def test_structured_backend_tool_error_is_translated() -> None:
    client = BackendMCPClient(source=_server())
    with pytest.raises(BackendMCPError) as caught:
        client.call_tool(
            "activate_policy_version",
            {"policy_key": "p", "version": 1, "confirmed": False},
        )
    assert caught.value.code == "CONFIRMATION_REQUIRED"
    assert caught.value.retryable is False


def test_unknown_tool_is_rejected() -> None:
    client = BackendMCPClient(source=_server())
    with pytest.raises(ValueError, match="not allowed"):
        client.call_tool("invented_tool", {})


def test_gateway_uses_frozen_signatures() -> None:
    mock = MagicMock(spec=BackendMCPClient)
    mock.call_tool.side_effect = lambda name, args: {"name": name, "args": args}
    gateway = GovernanceGateway(client=mock)

    gateway.inspect_ranger_state(kind="health")
    mock.call_tool.assert_called_with("inspect_ranger_state", {"kind": "health"})

    gateway.query_trino_readonly(sql="SELECT 1")
    mock.call_tool.assert_called_with("query_trino_readonly", {"sql": "SELECT 1"})

    gateway.activate_policy_version(policy_key="p", version=1)
    assert mock.call_tool.call_args.args[1]["confirmed"] is False

    gateway.rollback_policy(policy_key="p", target_version=1)
    assert mock.call_tool.call_args.args[1]["confirmed"] is False


def test_required_typed_gateway_wrappers() -> None:
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
    assert gateway.get_workflow_status("exec")["tool"] == "get_workflow_status"
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
    assert gateway.request_ranger_sync(policy_key="p")["tool"] == "request_ranger_sync"


def test_important_frozen_input_schemas_reject_r6a_assumptions() -> None:
    probe = BackendMCPClient(source=_server()).validate_frozen_contract()
    schemas = {item["name"]: item["input_schema"] for item in probe["tools"]}

    ranger_props = schemas["inspect_ranger_state"]["properties"]
    assert "kind" in ranger_props
    assert "service_name" not in ranger_props

    trino_props = schemas["query_trino_readonly"]["properties"]
    assert set(trino_props) == {"sql"}
    assert "query" not in trino_props
    assert "username" not in trino_props
    assert "limit" not in trino_props
