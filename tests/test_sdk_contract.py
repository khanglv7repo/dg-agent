from __future__ import annotations

from unittest.mock import MagicMock

import ai_sdk
from ai_sdk import AISdk
from ai_sdk.auth import TokenAuth
from ai_sdk.client import MCPClient

from app.gateways.openmetadata import OpenMetadataGateway


def test_official_openmetadata_ai_sdk_imports() -> None:
    assert ai_sdk is not None
    assert AISdk is not None
    assert MCPClient is not None
    assert TokenAuth is not None


def test_official_mcp_client_methods_contract() -> None:
    sdk = AISdk(host="http://localhost:8585", token="test-token")
    assert hasattr(sdk.mcp, "list_tools")
    assert hasattr(sdk.mcp, "call_tool")


def test_gateway_can_capture_live_patch_entity_schema_without_private_imports() -> None:
    gateway = OpenMetadataGateway(
        endpoint="http://localhost:8585/mcp",
        token="test-token",
        fallback_mcp=MagicMock(),
    )
    tool = MagicMock()
    tool.name = "patch_entity"
    tool.inputSchema = {
        "type": "object",
        "required": ["entityType", "fqn", "patch"],
        "properties": {
            "entityType": {"type": "string"},
            "fqn": {"type": "string"},
            "patch": {"type": "string"},
        },
    }
    gateway._sdk = MagicMock()
    gateway._sdk.mcp.list_tools.return_value = [tool]
    contract = gateway.patch_entity_contract()
    assert contract["name"] == "patch_entity"
    assert gateway._patch_entity_schema_supported(contract) is True
