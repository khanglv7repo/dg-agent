"""Gateways for external integration boundaries (OpenMetadata AI SDK/MCP and Backend Governance MCP)."""
from __future__ import annotations

from app.gateways.openmetadata import OpenMetadataGateway
from app.gateways.governance import GovernanceGateway

__all__ = ["OpenMetadataGateway", "GovernanceGateway"]
