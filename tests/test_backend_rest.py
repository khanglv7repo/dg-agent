from __future__ import annotations

import json

import httpx
import pytest

from app.clients.backend_rest import BackendRestClient, BackendRestError


def _client(handler) -> BackendRestClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return BackendRestClient(
        base_url="http://127.0.0.1:8000/api/v1",
        client=http_client,
    )


def test_create_dq_test_case_sends_trusted_identity_headers_and_body() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": "tc-1",
                "natural_key_hash": "dg_abc123",
                "om_testcase_id": "om-1",
                "status": "STAGED",
            },
        )

    client = _client(handler)
    result = client.create_dq_test_case(
        target_asset_fqn="financial.crm.customers",
        test_definition_fqn="columnValuesToBeNotNull",
        parameter_values={},
        rule_id="rule-1",
        worker_id="agent-worker-1",
    )

    assert result["status"] == "STAGED"
    assert result["natural_key_hash"] == "dg_abc123"
    assert captured["url"] == "http://127.0.0.1:8000/api/v1/dq/test-cases"
    assert captured["headers"]["x-actor-id"] == "governance-agent-bot"
    assert captured["headers"]["x-actor-roles"] == "governance-agent-bot"
    assert captured["body"]["target_asset_fqn"] == "financial.crm.customers"
    assert captured["body"]["rule_id"] == "rule-1"
    assert "test_key" not in captured["body"]
    assert "column_name" not in captured["body"]


def test_optional_fields_included_only_when_provided() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "1", "natural_key_hash": "x", "om_testcase_id": None, "status": "STAGED"})

    client = _client(handler)
    client.create_dq_test_case(
        target_asset_fqn="a.b.c",
        test_definition_fqn="d",
        parameter_values={"k": "v"},
        rule_id="rule-1",
        worker_id="w-1",
        test_key="slot-2",
        column_name="email",
        rationale="pii check",
    )
    assert captured["body"]["test_key"] == "slot-2"
    assert captured["body"]["column_name"] == "email"
    assert captured["body"]["rationale"] == "pii check"


def test_409_conflict_maps_to_backend_rest_error_with_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"error": {"code": "CONFLICT", "message": "already reserved", "details": {}}},
        )

    client = _client(handler)
    with pytest.raises(BackendRestError) as exc_info:
        client.create_dq_test_case(
            target_asset_fqn="a.b.c",
            test_definition_fqn="d",
            parameter_values={},
            rule_id="rule-1",
            worker_id="w-1",
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "CONFLICT"
    assert exc_info.value.retryable is False


def test_422_validation_error_maps_correctly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"error": {"code": "VALIDATION_ERROR", "message": "bad input", "details": {}}},
        )

    client = _client(handler)
    with pytest.raises(BackendRestError) as exc_info:
        client.create_dq_test_case(
            target_asset_fqn="",
            test_definition_fqn="d",
            parameter_values={},
            rule_id="rule-1",
            worker_id="w-1",
        )
    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "VALIDATION_ERROR"


def test_503_retryable_external_system_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"error": {"code": "EXTERNAL_SYSTEM_ERROR", "message": "OM down", "details": {}}},
        )

    client = _client(handler)
    with pytest.raises(BackendRestError) as exc_info:
        client.create_dq_test_case(
            target_asset_fqn="a.b.c",
            test_definition_fqn="d",
            parameter_values={},
            rule_id="rule-1",
            worker_id="w-1",
        )
    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is True


def test_non_json_error_body_falls_back_to_generic_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    client = _client(handler)
    with pytest.raises(BackendRestError) as exc_info:
        client.create_dq_test_case(
            target_asset_fqn="a.b.c",
            test_definition_fqn="d",
            parameter_values={},
            rule_id="rule-1",
            worker_id="w-1",
        )
    assert exc_info.value.status_code == 500
    assert exc_info.value.code == "BACKEND_REST_ERROR"
