from __future__ import annotations

import itertools
import json
from typing import Any

import httpx

READ_ONLY_TOOLS = frozenset(
    {
        "search_metadata",
        "semantic_search",
        "get_entity_details",
        "get_entity_lineage",
        "get_test_definitions",
    }
)


class OpenMetadataMCPClient:
    """Minimal JSON-RPC client for the OpenMetadata MCP endpoint.

    The classification agent is read-only by construction. Mutation tools such
    as `patch_entity` are deliberately not exposed to this client.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        token: str | None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.headers = {"Accept": "application/json, text/event-stream"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.client = client or httpx.Client(timeout=timeout)
        self.endpoint = endpoint
        self._ids = itertools.count(1)

    def close(self) -> None:
        self.client.close()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in READ_ONLY_TOOLS:
            raise ValueError(f"MCP tool is not allowed for this agent: {name}")
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        response = self.client.post(self.endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        body = self._decode_response(response)
        if body.get("error"):
            raise RuntimeError(f"OpenMetadata MCP error: {body['error']}")
        return body.get("result")

    @staticmethod
    def _decode_response(response: httpx.Response) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            return response.json()
        frames = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
        if not frames:
            raise RuntimeError("OpenMetadata MCP SSE response contained no data frame")
        return json.loads(frames[-1])

    def entity_context(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        include_lineage: bool,
    ) -> dict[str, Any]:
        context = {
            "details": self.call_tool(
                "get_entity_details",
                {"entity_type": entity_type, "fqn": entity_fqn},
            )
        }
        if include_lineage:
            context["lineage"] = self.call_tool(
                "get_entity_lineage",
                {"entity_type": entity_type, "fqn": entity_fqn},
            )
        return context
