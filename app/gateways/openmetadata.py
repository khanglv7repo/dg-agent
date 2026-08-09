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


class ToolNameWrapper:
    """Wrapper providing .value attribute for AISdk tool calls without private imports."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.value == other
        return getattr(other, "value", str(other)) == self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return repr(self.value)


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

    def _call_sdk_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return self._sdk.mcp.call_tool(ToolNameWrapper(tool_name), arguments)

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
                args = {"entity_type": entity_type, "fqn": entity_fqn}
                details_res = self._call_sdk_mcp_tool("get_entity_details", args)
                details = (
                    details_res.data if hasattr(details_res, "data") and details_res.data is not None
                    else (details_res.get("data") if isinstance(details_res, dict) and "data" in details_res else details_res)
                )

                context = {"details": details}
                if include_lineage:
                    lineage_res = self._call_sdk_mcp_tool("get_entity_lineage", args)
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

    def _fetch_tags_from_url(
        self,
        base_url: str,
        headers: dict[str, str],
        http_client: Any,
    ) -> list[str] | None:
        """Fetch all tags strictly parsing data[].fullyQualifiedName with pagination."""
        tags: list[str] = []
        after_cursor: str | None = None
        while True:
            url = f"{base_url}/api/v1/tags?limit=100"
            if after_cursor:
                url += f"&after={after_cursor}"
            resp = http_client.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            try:
                raw_data = resp.json()
            except Exception:
                return None
            if not isinstance(raw_data, dict):
                return None
            items = raw_data.get("data")
            if not isinstance(items, list):
                return None
            for item in items:
                if isinstance(item, dict):
                    fqn = item.get("fullyQualifiedName")
                    if isinstance(fqn, str) and fqn.strip():
                        tags.append(fqn.strip())
                # Strictly ignore string items, 'name' fallbacks, or 'fqn' fallbacks

            paging = raw_data.get("paging")
            if isinstance(paging, dict) and isinstance(paging.get("after"), str) and paging.get("after"):
                after_cursor = paging["after"]
            else:
                after_cursor = None

            if not after_cursor:
                break
        return tags

    def get_taxonomies(self) -> list[str]:
        """Fetch actual existing classification tag FQNs from OpenMetadata via native Tags API.

        Per verified OpenMetadata contract:
        - Primary path uses native Tags API (/api/v1/tags) with paging.after pagination.
        - Strictly parses data[].fullyQualifiedName (ignores name, fqn, or string items).
        - Fallback path uses fallback HTTP client with native Tags API.
        - Malformed/unexpected responses or failures fail closed and return [].
        """
        import httpx

        base_url = self.endpoint.rsplit("/mcp", 1)[0].rstrip("/")
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        if self._sdk is not None:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    tags = self._fetch_tags_from_url(base_url, headers, client)
                    if tags is not None:
                        self.active_transport = "native_api"
                        return tags
                    else:
                        logger.warning("Primary native taxonomy request failed or returned non-200")
                        self.active_transport = "fallback"
            except Exception as exc:
                logger.warning(f"Primary native taxonomy call failed: {exc}")
                self.active_transport = "fallback"

        try:
            self.active_transport = "fallback"
            if self._fallback_mcp and hasattr(self._fallback_mcp, "client"):
                tags = self._fetch_tags_from_url(base_url, headers, self._fallback_mcp.client)
                if tags is not None:
                    return tags
        except Exception as exc:
            logger.warning(f"Fallback native taxonomy call failed: {exc}")

        return []

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
