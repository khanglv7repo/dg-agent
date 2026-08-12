from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.gateways.openmetadata import OpenMetadataMutationError
from app.gateways.openmetadata_context import OpenMetadataGateway


FQN = "financial_postgres.financial_db.analytics.customer_360"
ENTITY = {
    "id": "entity-1",
    "name": "customer_360",
    "fullyQualifiedName": FQN,
    "columns": [
        {
            "name": "customer_name",
            "fullyQualifiedName": f"{FQN}.customer_name",
            "dataType": "VARCHAR",
        }
    ],
    "tags": [],
}


def _gateway(*, fallback=None) -> OpenMetadataGateway:
    gateway = OpenMetadataGateway(
        endpoint="http://localhost:8585/mcp",
        token="token",
        fallback_mcp=fallback or MagicMock(),
    )
    gateway._sdk = None
    return gateway


def _tool_error_payload() -> dict:
    return {
        "isError": True,
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "error": "Error executing tool: resource is marked non-null but is null",
                        "statusCode": 500,
                    }
                ),
            }
        ],
    }


def test_invalid_fallback_payload_uses_native_rest_and_never_returns_error_context() -> None:
    fallback = MagicMock()
    fallback.call_tool.return_value = _tool_error_payload()
    gateway = _gateway(fallback=fallback)
    gateway.get_entity_native = MagicMock(return_value=ENTITY)

    context = gateway.get_entity_context(
        entity_type="table",
        entity_fqn=FQN,
        include_lineage=True,
    )

    assert context["details"]["fullyQualifiedName"] == FQN
    assert context["details"]["columns"][0]["name"] == "customer_name"
    assert gateway.active_transport == "native_api"
    assert "error executing tool" not in json.dumps(context).lower()
    gateway.get_entity_native.assert_called_once_with(
        entity_type="table",
        entity_fqn=FQN,
        fields="tags,columns,service,description",
    )


def test_valid_fallback_mcp_details_are_unwrapped_without_native_rest() -> None:
    fallback = MagicMock()

    def call_tool(name, _args):
        if name == "get_entity_details":
            return {
                "content": [
                    {"type": "text", "text": json.dumps(ENTITY)}
                ]
            }
        return _tool_error_payload()

    fallback.call_tool.side_effect = call_tool
    gateway = _gateway(fallback=fallback)
    gateway.get_entity_native = MagicMock()

    context = gateway.get_entity_context(
        entity_type="table",
        entity_fqn=FQN,
        include_lineage=True,
    )

    assert context == {"details": ENTITY}
    assert gateway.active_transport == "fallback_mcp"
    gateway.get_entity_native.assert_not_called()


def test_sdk_failure_then_invalid_fallback_uses_native_rest() -> None:
    fallback = MagicMock()
    fallback.call_tool.return_value = _tool_error_payload()
    gateway = _gateway(fallback=fallback)
    gateway._sdk = MagicMock()
    gateway._call_sdk_mcp_tool = MagicMock(side_effect=RuntimeError("sdk failure"))
    gateway.get_entity_native = MagicMock(return_value=ENTITY)

    context = gateway.get_entity_context(
        entity_type="table",
        entity_fqn=FQN,
        include_lineage=False,
    )

    assert context["details"]["fullyQualifiedName"] == FQN
    assert gateway.active_transport == "native_api"


def test_all_context_transports_invalid_fail_closed() -> None:
    fallback = MagicMock()
    fallback.call_tool.return_value = _tool_error_payload()
    gateway = _gateway(fallback=fallback)
    gateway.get_entity_native = MagicMock(
        side_effect=OpenMetadataMutationError("native unavailable")
    )

    with pytest.raises(OpenMetadataMutationError, match="entity context unavailable"):
        gateway.get_entity_context(
            entity_type="table",
            entity_fqn=FQN,
            include_lineage=True,
        )
