"""Typed deterministic gateway over Backend FastMCP contracts."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from app.clients.backend_mcp import BackendMCPClient, BackendMCPError


class GovernanceGateway:
    """Application-facing Backend gateway.

    Workflows invoke only explicit typed methods. The LLM never chooses an
    arbitrary MCP tool name. R6-B deliberately adds the bounded classification
    completion continuation to the frozen R5 capability set.
    """

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:8001/mcp",
        timeout: float = 30.0,
        client: BackendMCPClient | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.client = client or BackendMCPClient(endpoint=endpoint, timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def validate_contract(self) -> dict[str, Any]:
        return self.client.validate_r6b_contract()

    def get_policy(self, policy_key: str, version: int | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {"policy_key": policy_key}
        if version is not None:
            args["version"] = version
        return self.client.call_tool("get_policy", args)

    def list_policy_versions(self, policy_key: str) -> list[dict[str, Any]]:
        return self.client.call_tool("list_policy_versions", {"policy_key": policy_key})

    def preview_policy_change(self, *, policy_key: str, logical_policy: dict[str, Any]) -> dict[str, Any]:
        return self.client.call_tool("preview_policy_change", {"policy_key": policy_key, "logical_policy": logical_policy})

    def check_policy_conflict(self, *, policy_key: str, logical_policy: dict[str, Any]) -> dict[str, Any]:
        return self.client.call_tool("check_policy_conflict", {"policy_key": policy_key, "logical_policy": logical_policy})

    def resolve_resource_mapping(self, *, om_service_name: str, environment: str) -> dict[str, Any]:
        return self.client.call_tool("resolve_resource_mapping", {"om_service_name": om_service_name, "environment": environment})

    def get_ranger_sync_status(self, *, policy_key: str, version: int | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {"policy_key": policy_key}
        if version is not None:
            args["version"] = version
        return self.client.call_tool("get_ranger_sync_status", args)

    def get_workflow_status(self, execution_id: str) -> dict[str, Any]:
        return self.client.call_tool("get_workflow_status", {"execution_id": execution_id})

    def get_audit_summary(
        self,
        *,
        object_type: str | None = None,
        object_id: str | None = None,
        policy_key: str | None = None,
        action: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"limit": limit}
        optional = {"object_type": object_type, "object_id": object_id, "policy_key": policy_key, "action": action, "since": since, "until": until}
        args.update({key: value for key, value in optional.items() if value is not None})
        return self.client.call_tool("get_audit_summary", args)

    def inspect_ranger_state(
        self,
        *,
        kind: Literal["health", "policy", "policy_key", "user", "group"],
        name: str | None = None,
        policy_key: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"kind": kind}
        if name is not None:
            args["name"] = name
        if policy_key is not None:
            args["policy_key"] = policy_key
        return self.client.call_tool("inspect_ranger_state", args)

    def query_trino_readonly(self, *, sql: str) -> dict[str, Any]:
        return self.client.call_tool("query_trino_readonly", {"sql": sql})

    def create_policy_version(self, *, policy_key: str, logical_policy: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {"policy_key": policy_key, "logical_policy": logical_policy}
        if reason:
            args["reason"] = reason
        return self.client.call_tool("create_policy_version", args)

    def activate_policy_version(
        self,
        *,
        policy_key: str,
        version: int,
        confirmed: bool = False,
        approval_reason: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"policy_key": policy_key, "version": version, "confirmed": confirmed}
        if approval_reason:
            args["approval_reason"] = approval_reason
        return self.client.call_tool("activate_policy_version", args)

    def rollback_policy(
        self,
        *,
        policy_key: str,
        target_version: int,
        confirmed: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"policy_key": policy_key, "target_version": target_version, "confirmed": confirmed}
        if reason:
            args["reason"] = reason
        return self.client.call_tool("rollback_policy", args)

    def update_service_mapping(
        self,
        *,
        om_service_name: str,
        trino_catalog: str,
        ranger_service_name: str,
        environment: str,
        confirmed: bool = False,
        ranger_tag_service_name: str | None = None,
        enabled: bool = True,
        reason: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "om_service_name": om_service_name,
            "trino_catalog": trino_catalog,
            "ranger_service_name": ranger_service_name,
            "environment": environment,
            "confirmed": confirmed,
            "enabled": enabled,
        }
        if ranger_tag_service_name is not None:
            args["ranger_tag_service_name"] = ranger_tag_service_name
        if reason:
            args["reason"] = reason
        return self.client.call_tool("update_service_mapping", args)

    def request_ranger_sync(self, *, policy_key: str) -> dict[str, Any]:
        return self.client.call_tool("request_ranger_sync", {"policy_key": policy_key})

    def complete_classification_execution(
        self,
        *,
        execution_id: str,
        generation: int,
        status: Literal["COMPLETED", "NO_PROPOSAL"],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Complete the same already-dispatched classification generation."""
        return self.client.call_tool(
            "complete_classification_execution",
            {"execution_id": execution_id, "generation": generation, "status": status, "result": result},
        )


__all__ = ["GovernanceGateway", "BackendMCPError"]
