from __future__ import annotations

from unittest.mock import MagicMock
import pytest

# Direct imports — must NOT skip if missing
import ai_sdk
from ai_sdk import AISdk
from ai_sdk.client import MCPClient
from ai_sdk.auth import TokenAuth

from app.gateways.openmetadata import OpenMetadataGateway


def test_official_openmetadata_ai_sdk_imports() -> None:
    assert ai_sdk is not None
    assert AISdk is not None
    assert MCPClient is not None
    assert TokenAuth is not None


def test_official_mcp_client_methods_contract() -> None:
    sdk = AISdk(host="http://localhost:8585", token="test-token")
    client = sdk.mcp

    # Verify public API methods exist on MCPClient via AISdk.mcp
    assert hasattr(client, "list_tools")
    assert hasattr(client, "call_tool")
    assert hasattr(client, "as_langchain_tools")
    assert hasattr(client, "as_openai_tools")

    # Verify invented method names DO NOT exist
    with pytest.raises(AttributeError):
        _ = client.nonexistent_invented_mcp_method()


def test_gateway_uses_official_sdk_as_primary_path() -> None:
    gateway = OpenMetadataGateway(endpoint="http://localhost:8585/mcp", token="test-token")

    # Mock the internal AISdk instance mcp.call_tool
    mock_mcp = MagicMock()
    mock_mcp.call_tool.return_value = {"success": True, "data": {"name": "customer_table"}}
    gateway._sdk = MagicMock()
    gateway._sdk.mcp = mock_mcp

    res = gateway.get_entity_context(entity_type="table", entity_fqn="service.db.customer_table", include_lineage=False)
    assert res == {"details": {"name": "customer_table"}}
    assert gateway.active_transport == "official_sdk"
    mock_mcp.call_tool.assert_called_once_with(
        "get_entity_details", {"entity_type": "table", "fqn": "service.db.customer_table"}
    )


def test_gateway_falls_back_when_official_sdk_fails() -> None:
    mock_fallback = MagicMock()
    mock_fallback.entity_context.return_value = {"details": {"name": "fallback_table"}}

    gateway = OpenMetadataGateway(
        endpoint="http://localhost:8585/mcp",
        token="test-token",
        fallback_mcp=mock_fallback,
    )

    # Simulate official SDK tool call raising an exception
    gateway._sdk = MagicMock()
    gateway._sdk.mcp.call_tool.side_effect = RuntimeError("Official SDK connection failed")

    res = gateway.get_entity_context(entity_type="table", entity_fqn="service.db.fallback_table", include_lineage=False)
    assert res == {"details": {"name": "fallback_table"}}
    assert gateway.active_transport == "fallback"
    mock_fallback.entity_context.assert_called_once_with(
        entity_type="table", entity_fqn="service.db.fallback_table", include_lineage=False
    )


def test_official_sdk_tag_mutation_is_unverified() -> None:
    gateway = OpenMetadataGateway(endpoint="http://localhost:8585/mcp", token="test-token")
    with pytest.raises(NotImplementedError, match="UNVERIFIED"):
        gateway.apply_tag_authoritative(
            entity_type="table",
            entity_fqn="db.schema.table",
            tag_fqn="PII.Email",
        )
