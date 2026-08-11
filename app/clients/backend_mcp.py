"""Typed transport client for the frozen R5 Backend FastMCP server."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastmcp import Client
from fastmcp.exceptions import ToolError

EXPECTED_BACKEND_TOOLS = (
    "get_policy",
    "list_policy_versions",
    "preview_policy_change",
    "check_policy_conflict",
    "resolve_resource_mapping",
    "get_ranger_sync_status",
    "get_workflow_status",
    "get_audit_summary",
    "inspect_ranger_state",
    "query_trino_readonly",
    "create_policy_version",
    "activate_policy_version",
    "rollback_policy",
    "update_service_mapping",
    "request_ranger_sync",
)
ALLOWED_BACKEND_TOOLS = frozenset(EXPECTED_BACKEND_TOOLS)


class BackendMCPError(RuntimeError):
    """Stable Agent-side representation of a Backend MCP semantic error."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        system: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.system = system
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }
        if self.system:
            payload["system"] = self.system
        return payload


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        if set(value) == {"root"}:
            return _plain(value["root"])
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _structured_result(result: Any) -> Any:
    """Prefer standard MCP structured JSON instead of FastMCP hydrated Root wrappers."""
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured

    data = getattr(result, "data", None)
    if data is not None:
        return _plain(data)

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


def _translate_tool_error(exc: ToolError) -> BackendMCPError:
    raw = str(exc)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return BackendMCPError(
            code="BACKEND_MCP_ERROR",
            message=raw[:2000] or "Backend MCP tool failed",
            retryable=False,
        )

    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return BackendMCPError(
            code="BACKEND_MCP_ERROR",
            message=raw[:2000] or "Backend MCP tool failed",
            retryable=False,
        )
    return BackendMCPError(
        code=str(error.get("code") or "BACKEND_MCP_ERROR"),
        message=str(error.get("message") or "Backend MCP tool failed"),
        retryable=bool(error.get("retryable", False)),
        system=str(error["system"]) if error.get("system") else None,
        details=error.get("details") if isinstance(error.get("details"), dict) else {},
    )


class BackendMCPClient:
    """Deterministic FastMCP client. It is not exposed as a generic LLM tool loop."""

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:8001/mcp",
        timeout: float = 30.0,
        source: Any | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._source = source if source is not None else endpoint

    def close(self) -> None:
        """Connections are scoped per call through `async with Client(...)`."""

    @staticmethod
    def _run(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "BackendMCPClient synchronous facade cannot be called from an active event loop"
        )

    async def _probe_async(self) -> dict[str, Any]:
        async with Client(self._source, timeout=self.timeout) as client:
            tools = await client.list_tools()
            init = client.initialize_result
            server_info = getattr(init, "serverInfo", None) if init is not None else None
            return {
                "server": {
                    "name": getattr(server_info, "name", None),
                    "version": getattr(server_info, "version", None),
                },
                "tools": [
                    {
                        "name": tool.name,
                        "input_schema": _plain(getattr(tool, "inputSchema", {})),
                    }
                    for tool in tools
                ],
            }

    def probe(self) -> dict[str, Any]:
        return self._run(self._probe_async())

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.probe()["tools"])

    def validate_frozen_contract(self) -> dict[str, Any]:
        probe = self.probe()
        actual = [item["name"] for item in probe["tools"]]
        if actual != list(EXPECTED_BACKEND_TOOLS):
            raise BackendMCPError(
                code="BACKEND_MCP_CONTRACT_MISMATCH",
                message="Backend MCP tool inventory does not match frozen R5 contract",
                retryable=False,
                details={
                    "expected": list(EXPECTED_BACKEND_TOOLS),
                    "actual": actual,
                },
            )
        return probe

    async def _call_tool_async(
        self,
        name: str,
        arguments: dict[str, Any] | None,
    ) -> Any:
        if name not in ALLOWED_BACKEND_TOOLS:
            raise ValueError(f"Backend MCP tool is not allowed: {name}")
        try:
            async with Client(self._source, timeout=self.timeout) as client:
                result = await client.call_tool(name, arguments or {})
                return _structured_result(result)
        except ToolError as exc:
            raise _translate_tool_error(exc) from None

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return self._run(self._call_tool_async(name, arguments))
