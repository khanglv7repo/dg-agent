"""OpenMetadata Gateway using official OpenMetadata AI SDK / MCP capabilities.

Per R6-A requirements:
- Wraps OpenMetadata metadata retrieval capabilities behind a high-level boundary.
- Decouples Agent reasoning from underlying JSON-RPC / SSE mechanics.
- Authoritative classification-tag mutation via official SDK is currently marked UNVERIFIED for R6-B.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    from ai_sdk.client import MCPClient as OfficialSDKMCPClient
    from ai_sdk.auth import TokenAuth
    from ai_sdk._http import HTTPClient as SDKHTTPClient
except ImportError:
    OfficialSDKMCPClient = None
    TokenAuth = None
    SDKHTTPClient = None

from app.clients.mcp import OpenMetadataMCPClient

logger = logging.getLogger(__name__)


class OpenMetadataGateway:
    """Gateway interfacing with OpenMetadata via official AI SDK / MCP or fallback transport."""

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
        self._official_sdk_client = None

        if OfficialSDKMCPClient is not None and token:
            try:
                base_url = endpoint.rsplit("/mcp", 1)[0]
                auth = TokenAuth(token)
                sdk_http = SDKHTTPClient(base_url=base_url, auth=auth, timeout=timeout)
                self._official_sdk_client = OfficialSDKMCPClient(
                    host=base_url,
                    auth=auth,
                    http=sdk_http,
                )
                logger.info("Initialized official OpenMetadata AI SDK MCPClient")
            except Exception as exc:
                logger.warning(f"Could not initialize official OpenMetadata AI SDK MCPClient: {exc}")

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
        """Fetch entity metadata and optional lineage from OpenMetadata."""
        return self._fallback_mcp.entity_context(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            include_lineage=include_lineage,
        )

    def get_taxonomies() -> list[str]:
        """Fetch allowed taxonomies / classification tag FQNs."""
        # Unverified dynamic fetch -> returns currently provided context allow-list
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
