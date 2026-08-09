from __future__ import annotations

from unittest.mock import MagicMock
from app.gateways.openmetadata import OpenMetadataGateway
from app.graph import run_governance_graph
from app.schemas import TagRecommendation, TagReasoningResult


def test_tag_reasoning_filters_out_unallowed_tags() -> None:
    om_mock = MagicMock(spec=OpenMetadataGateway)
    om_mock.get_entity_context.return_value = {"details": {}, "lineage": {}}

    classifier_mock = MagicMock()
    # LLM proposes one allowed tag and one invented tag
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
        allowed_tags=["PII.Email", "PII.Phone"],
    )

    assert tag_result is not None
    # Invented tag MUST be filtered out
    assert len(tag_result.recommendations) == 1
    assert tag_result.recommendations[0].tag == "PII.Email"


def test_tag_reasoning_insufficient_evidence_produces_no_proposal() -> None:
    om_mock = MagicMock(spec=OpenMetadataGateway)
    om_mock.get_entity_context.return_value = {"details": {"name": "generic_table"}, "lineage": {}}

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

    # Governance gateway (Ranger / Trino) must not be called during tag reasoning
    gov_mock.inspect_ranger_state.assert_not_called()
    gov_mock.query_trino_readonly.assert_not_called()
