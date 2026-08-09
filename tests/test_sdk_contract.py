from __future__ import annotations

import pytest

# Verify official OpenMetadata AI SDK import
ai_sdk = pytest.importorskip("ai_sdk")
from ai_sdk.client import AISdk, MCPClient
from ai_sdk.auth import TokenAuth
from ai_sdk._http import HTTPClient


def test_official_openmetadata_ai_sdk_imports() -> None:
    assert AISdk is not None
    assert MCPClient is not None
    assert TokenAuth is not None


def test_official_mcp_client_methods_contract() -> None:
    auth = TokenAuth("test-token")
    http = HTTPClient(base_url="http://localhost:8585", auth=auth)
    client = MCPClient(host="http://localhost:8585", auth=auth, http=http)

    # Verify actual methods exist on MCPClient
    assert hasattr(client, "list_tools")
    assert hasattr(client, "call_tool")
    assert hasattr(client, "as_langchain_tools")
    assert hasattr(client, "as_openai_tools")

    # Verify invented method names DO NOT exist (must raise AttributeError)
    with pytest.raises(AttributeError):
        _ = client.nonexistent_invented_mcp_method()


def test_official_sdk_tag_mutation_is_unverified() -> None:
    from app.gateways.openmetadata import OpenMetadataGateway

    gateway = OpenMetadataGateway(endpoint="http://localhost:8585/mcp", token="test-token")
    with pytest.raises(NotImplementedError, match="UNVERIFIED"):
        gateway.apply_tag_authoritative(
            entity_type="table",
            entity_fqn="db.schema.table",
            tag_fqn="PII.Email",
        )
