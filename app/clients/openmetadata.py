from __future__ import annotations

from typing import Any
import httpx


class OpenMetadataAgentClient:
    """Client for OpenMetadata REST API used by Agent Bot to submit native Suggestions."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def create_suggestion(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        tag_fqn: str,
        rationale: str,
    ) -> dict[str, Any]:
        """Create a native Suggestion in OpenMetadata directly from Agent Bot."""
        url = f"{self.base_url}/api/v1/suggestions"
        payload = {
            "type": "SuggestTagLabel",
            "entityType": entity_type,
            "entityFQN": entity_fqn,
            "description": rationale,
            "tag": {
                "tagFQN": tag_fqn,
                "labelType": "Automated",
                "state": "Suggested",
            },
        }
        response = self.client.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
