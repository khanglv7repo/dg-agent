"""Typed REST transport client for Backend's plain FastAPI routes.

Distinct from `app/clients/backend_mcp.py`: Backend's DQ TestCase-creation
endpoint (`POST /api/v1/dq/test-cases`, docs/13_IMPLEMENTATION_SPEC.md
section 4) is a normal REST route on the FastAPI app
(`backend/governance_app/app/main.py`), not one of the 17 `@mcp.tool`
functions on the separate Backend MCP server process. The two run as
separate processes on separate ports (MCP defaults to 8001, the FastAPI app
to uvicorn's default 8000) -- confirmed via `backend/governance_app`'s
`app/core/config.py:mcp_port` and `Makefile`'s plain `uvicorn app.main:app`.

Backend's HTTP identity is the trusted-header pattern
(`app/core/security.py:actor_from_headers`, gated by `TRUSTED_IDENTITY_HEADERS`)
used at the local/proxy trust boundary -- this client sends
`X-Actor-Id`/`X-Actor-Name`/`X-Actor-Roles` rather than a bearer token,
matching what `backend/governance_app/app/api/routes/dq.py` actually checks
(`governance-agent-bot`/`governance-operator`/`governance-admin`).
"""
from __future__ import annotations

from typing import Any

import httpx


class BackendRestError(RuntimeError):
    """Stable Agent-side representation of a Backend REST error response.

    Mirrors the exact `{"error": {"code", "message", "details"}}` body shape
    from `backend/governance_app/app/main.py`'s `governance_error_handler`,
    and the real status-code mapping verified there: 404 NotFoundError,
    409 ConflictError, 403 AuthorizationError, 422 ValidationError,
    502/503 ExternalSystemError (503 iff retryable).
    """

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
        # Matches Backend's own retryable convention: a 503 means Backend
        # itself judged the failure retryable (ExternalSystemError.retryable);
        # every other status is a deterministic outcome for this exact request.
        self.retryable = status_code == 503

    def as_dict(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


def _parse_error(response: httpx.Response) -> BackendRestError:
    try:
        payload = response.json()
    except Exception:
        return BackendRestError(
            status_code=response.status_code,
            code="BACKEND_REST_ERROR",
            message=response.text[:2000] or "Backend REST call failed",
        )
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return BackendRestError(
            status_code=response.status_code,
            code="BACKEND_REST_ERROR",
            message=str(payload)[:2000] or "Backend REST call failed",
        )
    return BackendRestError(
        status_code=response.status_code,
        code=str(error.get("code") or "BACKEND_REST_ERROR"),
        message=str(error.get("message") or "Backend REST call failed"),
        details=error.get("details") if isinstance(error.get("details"), dict) else {},
    )


class BackendRestClient:
    """Deterministic, bounded REST client. Only explicit typed methods are
    exposed -- no generic "call this path" escape hatch, matching the same
    LLM-cannot-choose-a-tool boundary as `BackendMCPClient`.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/api/v1",
        actor_id: str = "governance-agent-bot",
        actor_roles: str = "governance-agent-bot",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.actor_id = actor_id
        self.actor_roles = actor_roles
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _headers(self) -> dict[str, str]:
        return {
            "X-Actor-Id": self.actor_id,
            "X-Actor-Name": self.actor_id,
            "X-Actor-Roles": self.actor_roles,
        }

    def create_dq_test_case(
        self,
        *,
        target_asset_fqn: str,
        test_definition_fqn: str,
        parameter_values: dict[str, Any],
        rule_id: str,
        worker_id: str,
        test_key: str | None = None,
        column_name: str | None = None,
        rationale: str | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/dq/test-cases. Raises BackendRestError on any non-2xx."""
        body: dict[str, Any] = {
            "target_asset_fqn": target_asset_fqn,
            "test_definition_fqn": test_definition_fqn,
            "parameter_values": parameter_values,
            "rule_id": rule_id,
            "worker_id": worker_id,
        }
        if test_key is not None:
            body["test_key"] = test_key
        if column_name is not None:
            body["column_name"] = column_name
        if rationale is not None:
            body["rationale"] = rationale

        response = self._client.post(
            f"{self.base_url}/dq/test-cases",
            json=body,
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise _parse_error(response)
        return response.json()


__all__ = ["BackendRestClient", "BackendRestError"]
