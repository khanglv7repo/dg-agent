"""OpenMetadata gateway: official AI SDK for MCP reads, native verified API for tags."""
from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

import httpx

try:
    from ai_sdk import AISdk
except ImportError:
    AISdk = None

from app.clients.mcp import OpenMetadataMCPClient

logger = logging.getLogger(__name__)


class ToolNameWrapper:
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


class OpenMetadataMutationError(RuntimeError):
    pass


class OpenMetadataGateway:
    def __init__(
        self,
        *,
        endpoint: str,
        token: str | None,
        timeout: float = 30.0,
        fallback_mcp: OpenMetadataMCPClient | None = None,
        native_http_client: Any | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.token = token
        self.timeout = timeout
        self._fallback_mcp = fallback_mcp or OpenMetadataMCPClient(
            endpoint=endpoint,
            token=token,
            timeout=timeout,
        )
        self._native_http_client = native_http_client
        self._sdk = None
        self.active_transport = "fallback"

        if AISdk is not None and token:
            try:
                base_url = endpoint.rsplit("/mcp", 1)[0]
                self._sdk = AISdk(host=base_url, token=token, timeout=timeout)
                self.active_transport = "official_sdk"
            except Exception as exc:
                logger.warning("Could not initialize OpenMetadata AISdk: %s", exc)

    @property
    def base_url(self) -> str:
        return self.endpoint.rsplit("/mcp", 1)[0].rstrip("/")

    def close(self) -> None:
        if self._fallback_mcp:
            self._fallback_mcp.close()
        client = self._native_http_client
        if client is not None and hasattr(client, "close"):
            client.close()

    def _call_sdk_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return self._sdk.mcp.call_tool(ToolNameWrapper(tool_name), arguments)

    @staticmethod
    def _tool_schema(tool: Any) -> dict[str, Any]:
        for name in ("inputSchema", "input_schema"):
            value = getattr(tool, name, None)
            if isinstance(value, dict):
                return value
        if isinstance(tool, dict):
            for name in ("inputSchema", "input_schema"):
                if isinstance(tool.get(name), dict):
                    return tool[name]
        return {}

    @staticmethod
    def _tool_name(tool: Any) -> str:
        if isinstance(tool, dict):
            return str(tool.get("name") or "")
        return str(getattr(tool, "name", "") or "")

    def list_sdk_tools(self) -> list[dict[str, Any]]:
        """Inspect the actual live AISdk MCP contract; no assumed private SDK API."""
        if self._sdk is None:
            raise OpenMetadataMutationError(
            "OpenMetadata AISdk client is not initialized; verify data-ai-sdk import "
            "and OPENMETADATA_AGENT_BOT_TOKEN"
        )
        tools = self._sdk.mcp.list_tools()
        raw = getattr(tools, "data", tools)
        if isinstance(raw, dict) and isinstance(raw.get("tools"), list):
            raw = raw["tools"]
        if not isinstance(raw, list):
            raw = list(raw) if raw is not None else []
        return [
            {"name": self._tool_name(tool), "input_schema": self._tool_schema(tool)}
            for tool in raw
        ]

    def patch_entity_contract(self) -> dict[str, Any] | None:
        for tool in self.list_sdk_tools():
            if tool["name"] == "patch_entity":
                return tool
        return None

    @staticmethod
    def _patch_entity_schema_supported(contract: dict[str, Any] | None) -> bool:
        if not contract:
            return False
        schema = contract.get("input_schema") or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        expected = {"entityType", "fqn", "patch"}
        return expected.issubset(props) and expected.issubset(required)

    def get_entity_context(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        include_lineage: bool = True,
    ) -> dict[str, Any]:
        if self._sdk is not None:
            try:
                self.active_transport = "official_sdk"
                args = {"entity_type": entity_type, "fqn": entity_fqn}
                details_res = self._call_sdk_mcp_tool("get_entity_details", args)
                details = (
                    details_res.data
                    if hasattr(details_res, "data") and details_res.data is not None
                    else (
                        details_res.get("data")
                        if isinstance(details_res, dict) and "data" in details_res
                        else details_res
                    )
                )
                context = {"details": details}
                if include_lineage:
                    lineage_res = self._call_sdk_mcp_tool("get_entity_lineage", args)
                    lineage = (
                        lineage_res.data
                        if hasattr(lineage_res, "data") and lineage_res.data is not None
                        else (
                            lineage_res.get("data")
                            if isinstance(lineage_res, dict) and "data" in lineage_res
                            else lineage_res
                        )
                    )
                    context["lineage"] = lineage
                return context
            except Exception as exc:
                logger.warning("Official SDK MCP read failed; using fallback: %s", exc)
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
            paging = raw_data.get("paging")
            after_cursor = (
                paging.get("after")
                if isinstance(paging, dict) and isinstance(paging.get("after"), str)
                else None
            )
            if not after_cursor:
                break
        return tags

    def get_taxonomies(self) -> list[str]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                tags = self._fetch_tags_from_url(self.base_url, headers, client)
                if tags is not None:
                    self.active_transport = "native_api"
                    return tags
        except Exception as exc:
            logger.warning("Primary native taxonomy call failed: %s", exc)

        try:
            self.active_transport = "fallback"
            if self._fallback_mcp and hasattr(self._fallback_mcp, "client"):
                tags = self._fetch_tags_from_url(
                    self.base_url,
                    headers,
                    self._fallback_mcp.client,
                )
                if tags is not None:
                    return tags
        except Exception as exc:
            logger.warning("Fallback native taxonomy call failed: %s", exc)
        return []

    @staticmethod
    def collection_for(entity_type: str) -> str:
        mapping = {
            "table": "tables",
            "databaseSchema": "databaseSchemas",
            "topic": "topics",
        }
        return mapping.get(
            entity_type,
            entity_type if entity_type.endswith("s") else f"{entity_type}s",
        )

    def _native_headers(self, *, patch: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if patch:
            headers["Content-Type"] = "application/json-patch+json"
        return headers

    def _native_request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = dict(self._native_headers(patch=method.upper() == "PATCH"))
        headers.update(kwargs.pop("headers", {}))
        try:
            if self._native_http_client is not None:
                response = self._native_http_client.request(
                    method, url, headers=headers, **kwargs
                )
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise OpenMetadataMutationError(
                f"OpenMetadata {method} connection failed"
            ) from exc
        if response.status_code == 404:
            raise OpenMetadataMutationError(f"OpenMetadata target not found: {path}")
        if response.status_code >= 400:
            raise OpenMetadataMutationError(
                f"OpenMetadata {method} {path} failed with HTTP {response.status_code}"
            )
        if not response.content:
            return {}
        body = response.json()
        if not isinstance(body, dict):
            raise OpenMetadataMutationError("OpenMetadata response must be a JSON object")
        return body

    def get_entity_native(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        fields: str = "tags,columns,service",
    ) -> dict[str, Any]:
        collection = self.collection_for(entity_type)
        encoded = quote(entity_fqn, safe="")
        return self._native_request(
            "GET",
            f"/api/v1/{collection}/name/{encoded}",
            params={"fields": fields},
        )

    @staticmethod
    def _confirmed_tag_fqns(labels: Any) -> set[str]:
        result: set[str] = set()
        for item in labels or []:
            if not isinstance(item, dict):
                continue
            fqn = str(item.get("tagFQN") or "").strip()
            state = item.get("state")
            state_text = str(state).strip().lower() if state is not None else "confirmed"
            if fqn and state_text == "confirmed":
                result.add(fqn)
        return result

    @staticmethod
    def _tag_label(tag_fqn: str) -> dict[str, str]:
        return {
            "tagFQN": tag_fqn,
            "source": "Classification",
            "labelType": "Automated",
            "state": "Confirmed",
        }

    @classmethod
    def _without_tag(cls, labels: list[Any], tag_fqn: str) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in labels
            if isinstance(item, dict)
            and str(item.get("tagFQN") or "").strip() != tag_fqn
        ]

    @classmethod
    def _merged_confirmed(
        cls,
        labels: list[Any],
        tag_fqn: str,
    ) -> list[dict[str, Any]]:
        current = cls._without_tag(labels, tag_fqn)
        current.append(cls._tag_label(tag_fqn))
        return current

    @staticmethod
    def _resolve_column(
        entity: dict[str, Any],
        *,
        entity_fqn: str,
        field_path: str,
    ) -> tuple[int, dict[str, Any]]:
        columns = entity.get("columns") or []
        expected_name = (
            field_path.split(".", 1)[1]
            if field_path.startswith("columns.")
            else field_path.rsplit(".", 1)[-1]
        )
        for index, column in enumerate(columns):
            if not isinstance(column, dict):
                continue
            column_fqn = str(column.get("fullyQualifiedName") or "")
            name = str(column.get("name") or "")
            if field_path == column_fqn or (
                field_path.startswith("columns.") and name == expected_name
            ):
                if column_fqn and not column_fqn.startswith(f"{entity_fqn}."):
                    continue
                return index, column
        raise OpenMetadataMutationError(
            f"field_path {field_path!r} is not a column of {entity_fqn}"
        )

    def _apply_patch(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        patch: list[dict[str, Any]],
    ) -> str:
        contract = None
        if self._sdk is not None:
            try:
                contract = self.patch_entity_contract()
            except Exception as exc:
                logger.warning("Could not inspect live patch_entity contract: %s", exc)
        if self._sdk is not None and self._patch_entity_schema_supported(contract):
            self._call_sdk_mcp_tool(
                "patch_entity",
                {
                    "entityType": entity_type,
                    "fqn": entity_fqn,
                    "patch": json.dumps(patch, separators=(",", ":")),
                },
            )
            return "SDK_MCP"

        collection = self.collection_for(entity_type)
        entity = self.get_entity_native(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            fields="tags,columns",
        )
        entity_id = entity.get("id")
        if not entity_id:
            raise OpenMetadataMutationError("OpenMetadata entity did not return id")
        self._native_request(
            "PATCH",
            f"/api/v1/{collection}/{entity_id}",
            json=patch,
        )
        return "NATIVE_API"

    def apply_tag_authoritative(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        tag_fqn: str,
        field_path: str | None = None,
    ) -> dict[str, Any]:
        """READ -> APPLY IF MISSING -> READ BACK -> VERIFY."""
        taxonomy = set(self.get_taxonomies())
        if tag_fqn not in taxonomy:
            raise OpenMetadataMutationError(
                f"tag {tag_fqn!r} is not present in current OpenMetadata taxonomy"
            )

        before = self.get_entity_native(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            fields="tags,columns,service",
        )

        if field_path is None:
            current = list(before.get("tags") or [])
            if tag_fqn in self._confirmed_tag_fqns(current):
                return {
                    "status": "NO_CHANGE",
                    "entity_fqn": entity_fqn,
                    "field_path": None,
                    "tag_fqn": tag_fqn,
                    "mutation_count": 0,
                }
            patch_path = "/tags"
            patch_op = "replace" if "tags" in before else "add"
            target_entity_fqn = entity_fqn
        else:
            index, column = self._resolve_column(
                before,
                entity_fqn=entity_fqn,
                field_path=field_path,
            )
            current = list(column.get("tags") or [])
            if tag_fqn in self._confirmed_tag_fqns(current):
                return {
                    "status": "NO_CHANGE",
                    "entity_fqn": entity_fqn,
                    "field_path": field_path,
                    "tag_fqn": tag_fqn,
                    "mutation_count": 0,
                }
            patch_path = f"/columns/{index}/tags"
            patch_op = "replace" if "tags" in column else "add"
            target_entity_fqn = entity_fqn

        mutation_count = 0
        # Suggested -> Confirmed promotion is performed as remove then add/replace,
        # matching the already-proven R3 native OpenMetadata semantics.
        if any(
            isinstance(item, dict)
            and str(item.get("tagFQN") or "").strip() == tag_fqn
            for item in current
        ):
            self._apply_patch(
                entity_type=entity_type,
                entity_fqn=target_entity_fqn,
                patch=[{
                    "op": patch_op,
                    "path": patch_path,
                    "value": self._without_tag(current, tag_fqn),
                }],
            )
            mutation_count += 1
            current = self._without_tag(current, tag_fqn)
            patch_op = "replace"

        transport = self._apply_patch(
            entity_type=entity_type,
            entity_fqn=target_entity_fqn,
            patch=[{
                "op": patch_op,
                "path": patch_path,
                "value": self._merged_confirmed(current, tag_fqn),
            }],
        )
        mutation_count += 1

        after = self.get_entity_native(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            fields="tags,columns,service",
        )
        if field_path is None:
            observed = self._confirmed_tag_fqns(after.get("tags") or [])
        else:
            _, observed_column = self._resolve_column(
                after,
                entity_fqn=entity_fqn,
                field_path=field_path,
            )
            observed = self._confirmed_tag_fqns(observed_column.get("tags") or [])

        if tag_fqn not in observed:
            raise OpenMetadataMutationError(
                "OpenMetadata read-back did not contain the confirmed tag"
            )

        return {
            "status": "APPLIED",
            "entity_fqn": entity_fqn,
            "field_path": field_path,
            "tag_fqn": tag_fqn,
            "mutation_count": mutation_count,
            "transport": transport,
        }
