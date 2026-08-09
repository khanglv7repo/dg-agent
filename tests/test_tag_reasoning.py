from __future__ import annotations

from unittest.mock import MagicMock
import httpx
import pytest

from app.gateways.openmetadata import OpenMetadataGateway
from app.graph import compute_effective_allowed_tags, run_governance_graph
from app.schemas import TagRecommendation, TagReasoningResult


def test_effective_allowed_tags_case_a() -> None:
    """Case A: actual OM ["PII.Email", "PII.Phone"], caller ["PII.Email", "INVENTED.Tag"] -> ["PII.Email"]"""
    actual_om_tags = ["PII.Email", "PII.Phone"]
    caller_allowed = ["PII.Email", "INVENTED.Tag"]
    result = compute_effective_allowed_tags(actual_om_tags, caller_allowed)
    assert result == ["PII.Email"]


def test_effective_allowed_tags_case_b() -> None:
    """Case B: actual OM ["PII.Email", "PII.Phone"], caller [] -> ["PII.Email", "PII.Phone"]"""
    actual_om_tags = ["PII.Email", "PII.Phone"]
    caller_allowed = []
    result = compute_effective_allowed_tags(actual_om_tags, caller_allowed)
    assert result == ["PII.Email", "PII.Phone"]


def test_effective_allowed_tags_case_c() -> None:
    """Case C: actual OM [], caller ["PII.Email", "INVENTED.Tag"] -> []"""
    actual_om_tags = []
    caller_allowed = ["PII.Email", "INVENTED.Tag"]
    result = compute_effective_allowed_tags(actual_om_tags, caller_allowed)
    assert result == []


def test_effective_allowed_tags_case_d_exception() -> None:
    """Case D: taxonomy request fails -> actual OM [], caller ["PII.Email"] -> []"""
    actual_om_tags = []
    caller_allowed = ["PII.Email"]
    result = compute_effective_allowed_tags(actual_om_tags, caller_allowed)
    assert result == []


def test_taxonomy_failure_graph_fail_closed() -> None:
    """Taxonomy failure test: OM raises exception -> get_taxonomies() == [] -> recommendations == []"""
    om_mock = MagicMock(spec=OpenMetadataGateway)
    om_mock.get_entity_context.return_value = {"details": {"name": "user_table"}, "lineage": {}}
    # Simulate taxonomy retrieval raising an exception
    om_mock.get_taxonomies.side_effect = RuntimeError("OpenMetadata taxonomy service unreachable")

    classifier_mock = MagicMock()
    # Even if LLM proposes PII.Email
    classifier_mock.classify.return_value = TagReasoningResult(
        recommendations=[TagRecommendation(tag="PII.Email", confidence=0.95, rationale="User email column")],
        summary="Proposed PII.Email tag",
    )

    tag_result, _, _ = run_governance_graph(
        om_gateway=om_mock,
        gov_gateway=MagicMock(),
        tag_classifier=classifier_mock,
        policy_classifier=MagicMock(),
        request_type="TAG",
        entity_fqn="db.schema.user_table",
        allowed_tags=["PII.Email"],
    )

    assert tag_result is not None
    # Must fail closed: no recommendations returned because taxonomy is unavailable
    assert tag_result.recommendations == []


def test_malformed_taxonomy_response_fail_closed() -> None:
    """Malformed response test: simulated external boundary returns invalid/malformed response -> get_taxonomies() == []"""
    mock_fallback = MagicMock()
    # Simulate HTTP 200 with malformed/non-dict payload or invalid data field
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = "invalid non-dict JSON string"
    mock_fallback.client.get.return_value = mock_resp

    gateway = OpenMetadataGateway(
        endpoint="http://localhost:8585/mcp",
        token="test-token",
        fallback_mcp=mock_fallback,
    )
    gateway._sdk = None  # Force fallback transport

    tags = gateway.get_taxonomies()
    assert tags == []


def test_malformed_data_field_fail_closed() -> None:
    """Malformed response test: data key contains unexpected structure -> fail closed []"""
    mock_fallback = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": 12345}  # data is int instead of list
    mock_fallback.client.get.return_value = mock_resp

    gateway = OpenMetadataGateway(
        endpoint="http://localhost:8585/mcp",
        token="test-token",
        fallback_mcp=mock_fallback,
    )
    gateway._sdk = None

    tags = gateway.get_taxonomies()
    assert tags == []


def test_tag_reasoning_uses_effective_allowed_set_and_removes_unallowed_tags() -> None:
    om_mock = MagicMock(spec=OpenMetadataGateway)
    om_mock.get_entity_context.return_value = {"details": {}, "lineage": {}}
    om_mock.get_taxonomies.return_value = ["PII.Email", "PII.Phone"]

    classifier_mock = MagicMock()
    classifier_mock.classify.return_value = TagReasoningResult(
        recommendations=[
            TagRecommendation(tag="PII.Email", confidence=0.9, rationale="Valid allowed tag"),
            TagRecommendation(tag="INVENTED.Tag", confidence=0.9, rationale="Arbitrary invented tag"),
        ],
        summary="Test summary",
    )

    tag_result, _, _ = run_governance_graph(
        om_gateway=om_mock,
        gov_gateway=MagicMock(),
        tag_classifier=classifier_mock,
        policy_classifier=MagicMock(),
        request_type="TAG",
        entity_fqn="db.schema.table",
        allowed_tags=["PII.Email", "INVENTED.Tag"],
    )

    assert tag_result is not None
    assert len(tag_result.recommendations) == 1
    assert tag_result.recommendations[0].tag == "PII.Email"


def test_tag_reasoning_insufficient_evidence_produces_no_proposal() -> None:
    om_mock = MagicMock(spec=OpenMetadataGateway)
    om_mock.get_entity_context.return_value = {"details": {"name": "generic_table"}, "lineage": {}}
    om_mock.get_taxonomies.return_value = ["PII.Email"]

    classifier_mock = MagicMock()
    classifier_mock.classify.return_value = TagReasoningResult(
        recommendations=[],
        summary="No sensitive evidence found",
    )

    tag_result, _, _ = run_governance_graph(
        om_gateway=om_mock,
        gov_gateway=MagicMock(),
        tag_classifier=classifier_mock,
        policy_classifier=MagicMock(),
        request_type="TAG",
        entity_fqn="db.schema.generic_table",
        allowed_tags=["PII.Email"],
    )

    assert tag_result is not None
    assert len(tag_result.recommendations) == 0
    assert "No sensitive evidence" in tag_result.summary


def test_no_ranger_client_in_agent_tag_flow() -> None:
    om_mock = MagicMock(spec=OpenMetadataGateway)
    om_mock.get_entity_context.return_value = {"details": {}, "lineage": {}}
    om_mock.get_taxonomies.return_value = ["PII.Email"]

    classifier_mock = MagicMock()
    classifier_mock.classify.return_value = TagReasoningResult(recommendations=[], summary="No tags")

    gov_mock = MagicMock()

    tag_result, _, _ = run_governance_graph(
        om_gateway=om_mock,
        gov_gateway=gov_mock,
        tag_classifier=classifier_mock,
        policy_classifier=MagicMock(),
        request_type="TAG",
        entity_fqn="db.schema.table",
        allowed_tags=["PII.Email"],
    )

    gov_mock.inspect_ranger_state.assert_not_called()
    gov_mock.query_trino_readonly.assert_not_called()
