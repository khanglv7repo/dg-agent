"""Reliable OpenMetadata entity context gateway for R6-B reasoning.

The existing OpenMetadataGateway remains the authority-safe mutation implementation.
This subclass hardens only metadata reads used as LLM context:

official SDK MCP -> fallback MCP -> native OpenMetadata REST.

A transport response is accepted only when an actual entity mapping can be recovered.
Tool-error payloads are never forwarded to the LLM as metadata context.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.gateways.openmetadata import (
    OpenMetadataGateway as _BaseOpenMetadataGateway,
    OpenMetadataMutationError,
)

logger = logging.getLogger(__name__)


class OpenMetadataGateway(_BaseOpenMetadataGateway):
    """OpenMetadata gateway with validated, fail-closed entity-context fallback."""

    @staticmethod
    def _plain(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        data = getattr(value, "data", None)
        if data is not None and not isinstance(value, (dict, list, tuple, str, bytes)):
            return data
        return value

    @classmethod
    def _find_entity_mapping(
        cls,
        value: Any,
        *,
        entity_fqn: str,
    ) -> dict[str, Any] | None:
        value = cls._plain(value)

        if isinstance(value, str):
            text = value.strip()
            if not text or text[0:1] not in {"{", "["}:
                return None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            return cls._find_entity_mapping(parsed, entity_fqn=entity_fqn)

        if isinstance(value, dict):
            fqn = str(
                value.get("fullyQualifiedName")
                or value.get("fully_qualified_name")
                or value.get("fqn")
                or ""
            ).strip()
            name = str(value.get("name") or "").strip()
            columns = value.get("columns")
            expected_name = entity_fqn.rsplit(".", 1)[-1]

            if (
                fqn == entity_fqn
                and (name or isinstance(columns, list))
            ) or (
                not fqn
                and name == expected_name
                and isinstance(columns, list)
            ):
                return dict(value)

            priority_keys = (
                "data",
                "result",
                "structuredContent",
                "structured_content",
                "entity",
                "details",
                "value",
                "content",
            )
            for key in priority_keys:
                if key not in value:
                    continue
                found = cls._find_entity_mapping(
                    value[key],
                    entity_fqn=entity_fqn,
                )
                if found is not None:
                    return found

            for key, item in value.items():
                if key in priority_keys:
                    continue
                found = cls._find_entity_mapping(
                    item,
                    entity_fqn=entity_fqn,
                )
                if found is not None:
                    return found
            return None

        if isinstance(value, (list, tuple)):
            for item in value:
                found = cls._find_entity_mapping(
                    item,
                    entity_fqn=entity_fqn,
                )
                if found is not None:
                    return found
        return None

    @classmethod
    def _contains_tool_error(cls, value: Any) -> bool:
        value = cls._plain(value)
        if value is None:
            return False
        if isinstance(value, str):
            lowered = value.lower()
            return any(
                marker in lowered
                for marker in (
                    "error executing tool",
                    "resource is marked non-null but is null",
                    "statuscode\":500",
                    "status code 500",
                )
            )
        if isinstance(value, dict):
            if value.get("isError") is True or value.get("is_error") is True:
                return True
            status = value.get("statusCode", value.get("status_code"))
            try:
                if status is not None and int(status) >= 400:
                    return True
            except (TypeError, ValueError):
                pass
            if value.get("error") not in (None, "", {}, []):
                return True
            return any(cls._contains_tool_error(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(cls._contains_tool_error(item) for item in value)
        return False

    @classmethod
    def _normalized_context(
        cls,
        details: Any,
        *,
        entity_fqn: str,
        lineage: Any | None = None,
    ) -> dict[str, Any] | None:
        entity = cls._find_entity_mapping(details, entity_fqn=entity_fqn)
        if entity is None:
            return None
        context: dict[str, Any] = {"details": entity}
        if lineage is not None and not cls._contains_tool_error(lineage):
            context["lineage"] = cls._plain(lineage)
        return context

    def _sdk_context(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        include_lineage: bool,
    ) -> dict[str, Any] | None:
        if self._sdk is None:
            return None

        args = {"entity_type": entity_type, "fqn": entity_fqn}
        try:
            details = self._call_sdk_mcp_tool("get_entity_details", args)
        except Exception as exc:
            logger.warning("Official SDK MCP entity read failed: %s", exc)
            return None

        lineage = None
        if include_lineage:
            try:
                lineage = self._call_sdk_mcp_tool("get_entity_lineage", args)
            except Exception as exc:
                logger.warning("Official SDK MCP lineage read failed; omitting lineage: %s", exc)

        context = self._normalized_context(
            details,
            entity_fqn=entity_fqn,
            lineage=lineage,
        )
        if context is None:
            logger.warning(
                "Official SDK MCP returned no usable entity metadata for %s",
                entity_fqn,
            )
        return context

    def _fallback_mcp_context(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        include_lineage: bool,
    ) -> dict[str, Any] | None:
        if self._fallback_mcp is None:
            return None

        args = {"entity_type": entity_type, "fqn": entity_fqn}
        try:
            details = self._fallback_mcp.call_tool("get_entity_details", args)
        except Exception as exc:
            logger.warning("Fallback MCP entity read failed: %s", exc)
            return None

        lineage = None
        if include_lineage:
            try:
                lineage = self._fallback_mcp.call_tool("get_entity_lineage", args)
            except Exception as exc:
                logger.warning("Fallback MCP lineage read failed; omitting lineage: %s", exc)

        context = self._normalized_context(
            details,
            entity_fqn=entity_fqn,
            lineage=lineage,
        )
        if context is None:
            logger.warning(
                "Fallback MCP returned no usable entity metadata for %s",
                entity_fqn,
            )
        return context

    def _native_context(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
    ) -> dict[str, Any]:
        entity = self.get_entity_native(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            fields="tags,columns,service,description",
        )
        normalized = self._find_entity_mapping(entity, entity_fqn=entity_fqn)
        if normalized is None:
            raise OpenMetadataMutationError(
                f"native OpenMetadata response contained no usable entity metadata for {entity_fqn}"
            )
        return {"details": normalized}

    def get_entity_context(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        include_lineage: bool = True,
    ) -> dict[str, Any]:
        """Return validated entity metadata; never return a tool-error payload as context."""

        context = self._sdk_context(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            include_lineage=include_lineage,
        )
        if context is not None:
            self.active_transport = "official_sdk"
            return context

        context = self._fallback_mcp_context(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            include_lineage=include_lineage,
        )
        if context is not None:
            self.active_transport = "fallback_mcp"
            return context

        try:
            context = self._native_context(
                entity_type=entity_type,
                entity_fqn=entity_fqn,
            )
        except Exception as exc:
            raise OpenMetadataMutationError(
                "OpenMetadata entity context unavailable from official SDK MCP, "
                "fallback MCP, and native REST"
            ) from exc

        self.active_transport = "native_api"
        return context
