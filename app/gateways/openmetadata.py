"""OpenMetadata Gateway using official OpenMetadata AI SDK / MCP capabilities as primary transport.

Per R6-A final correctness requirements:
- Uses public supported AISdk API as the primary MCP path.
- Avoids depending on private module APIs.
- Handwritten OpenMetadataMCPClient serves as an explicit fallback.
- Implements get_taxonomies() using actual OpenMetadata search/metadata capabilities.
- Authoritative classification-tag mutation via official SDK is marked UNVERIFIED for R6-B.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    from ai_sdk import AISdk
except ImportError:
    AISdk = None

from app.clients.mcp import OpenMetadataMCPClient

logger = logging.getLogger(__name__)


class OpenMetadataGateway:
    """Gateway interfacing with OpenMetadata via official AI SDK / MCP with explicit fallback."""

    def __init__(
        self,
        *,
        endpoint: str,
        token: str | None,
        timeout: float = 30.0,
        fallback_mcp: OpenMetadataMCPClient | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.token = token
        self.timeout = timeout
        self._fallback_mcp = fallback_mcp or OpenMetadataMCPClient(
            endpoint=endpoint,
            token=token,
            timeout=timeout,
        )
        self._sdk = None
        self.active_transport: str = "fallback"

        if AISdk is not None and token:
            try:
                base_url = endpoint.rsplit("/mcp", 1)[0]
                self._sdk = AISdk(host=base_url, token=token, timeout=timeout)
                self.active_transport = "official_sdk"
                logger.info("Initialized primary official OpenMetadata AI SDK")
            except Exception as exc:
                logger.warning(f"Could not initialize official OpenMetadata AI SDK: {exc}")

    def close(self) -> None:
        if self._fallback_mcp:
            self._fallback_mcp.close()

    def get_entity_context(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        include_lineage: bool = True,
    ) -> dict[str, Any]:
        """Fetch entity metadata and optional lineage using official SDK as primary path."""
        if self._sdk is not None:
            try:
                self.active_transport = "official_sdk"
                details_res = self._sdk.mcp.call_tool(
                    "get_entity_details",
                    {"entity_type": entity_type, "fqn": entity_fqn},
                )
                details = (
                    details_res.data if hasattr(details_res, "data") and details_res.data is not None
                    else (details_res.get("data") if isinstance(details_res, dict) and "data" in details_res else details_res)
                )

                context = {"details": details}
                if include_lineage:
                    lineage_res = self._sdk.mcp.call_tool(
                        "get_entity_lineage",
                        {"entity_type": entity_type, "fqn": entity_fqn},
                    )
                    lineage = (
                        lineage_res.data if hasattr(lineage_res, "data") and lineage_res.data is not None
                        else (lineage_res.get("data") if isinstance(lineage_res, dict) and "data" in lineage_res else lineage_res)
                    )
                    context["lineage"] = lineage
                return context
            except Exception as exc:
                logger.warning(f"Official SDK MCP call failed, falling back to OpenMetadataMCPClient: {exc}")
                self.active_transport = "fallback"

        # Fallback path
        self.active_transport = "fallback"
        return self._fallback_mcp.entity_context(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            include_lineage=include_lineage,
        )

    def get_taxonomies(self) -> list[str]:
        """Fetch actual existing classification tag FQNs from OpenMetadata via MCP tool search."""
        raw_data = None
        if self._sdk is not None:
            try:
                self.active_transport = "official_sdk"
                res = self._sdk.mcp.call_tool("search_metadata", {"query": "*", "entity_type": "tag"})
                raw_data = res.data if hasattr(res, "data") and res.data is not None else res
            except Exception as exc:
                logger.warning(f"Official SDK get_taxonomies call failed: {exc}")
                self.active_transport = "fallback"

        if raw_data is None:
            try:
                self.active_transport = "fallback"
                raw_data = self._fallback_mcp.call_tool("search_metadata", {"query": "*", "entity_type": "tag"})
            except Exception as exc:
                logger.warning(f"Fallback MCP get_taxonomies call failed: {exc}")
                return []

        tags: list[str] = []
        if isinstance(raw_data, dict):
            hits = raw_data.get("hits", []) or raw_data.get("results", []) or raw_data.get("data", [])
            for hit in hits:
                if isinstance(hit, dict):
                    fqn = hit.get("fullyQualifiedName") or hit.get("fqn") or hit.get("name")
                    if fqn:
                        tags.append(str(fqn))
                elif isinstance(hit, str):
                    tags.append(hit)
        elif isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    fqn = item.get("fullyQualifiedName") or item.get("fqn") or item.get("name")
                    if fqn:
                        tags.append(str(fqn))
                elif isinstance(item, str):
                    tags.append(item)

        return tags

    def apply_tag_authoritative(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        tag_fqn: str,
    ) -> dict[str, Any]:
        """Explicit boundary for authoritative tag mutation in OpenMetadata.

        UNVERIFIED in R6-A: Official OpenMetadata AI SDK tag mutation API contract
        is not yet verified. R6-B will implement this after Backend R5 stabilization.
        """
        raise NotImplementedError(
            "UNVERIFIED: Authoritative tag mutation via official OpenMetadata AI SDK "
            "is not verified in R6-A and is deferred to R6-B."
        )
