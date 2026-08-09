from __future__ import annotations

from unittest.mock import MagicMock
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway
from app.graph import run_governance_graph
from app.schemas import (
    LogicalPolicyProposal,
    PolicyReasoningResult,
    PolicyResource,
    Subject,
    TagRecommendation,
    TagReasoningResult,
)


def test_agent_graph_tag_execution() -> None:
    om_mock = MagicMock(spec=OpenMetadataGateway)
    om_mock.get_entity_context.return_value = {
        "details": {"name": "customer_table", "columns": [{"name": "email"}]},
        "lineage": {},
    }

    gov_mock = MagicMock(spec=GovernanceGateway)

    tag_classifier_mock = MagicMock()
    tag_classifier_mock.model_name = "gpt-4o-mini"
    tag_classifier_mock.classify.return_value = TagReasoningResult(
        recommendations=[
            TagRecommendation(
                tag="PII.Email",
                confidence=0.95,
                rationale="Column email contains customer email addresses",
                field_path="customer_table.email",
                action_recommendation="APPLY",
            )
        ],
        summary="Found PII email column",
    )

    policy_classifier_mock = MagicMock()

    tag_result, policy_result, context = run_governance_graph(
        om_gateway=om_mock,
        gov_gateway=gov_mock,
        tag_classifier=tag_classifier_mock,
        policy_classifier=policy_classifier_mock,
        request_type="TAG",
        entity_type="table",
        entity_fqn="service.db.schema.customer_table",
        allowed_tags=["PII.Email", "PII.Phone"],
        include_lineage=True,
    )

    assert tag_result is not None
    assert policy_result is None
    assert len(tag_result.recommendations) == 1
    assert tag_result.recommendations[0].tag == "PII.Email"
    assert "details" in context
    # Verify GovernanceGateway was not called for TAG path
    gov_mock.inspect_ranger_state.assert_not_called()


def test_agent_graph_policy_execution() -> None:
    om_mock = MagicMock(spec=OpenMetadataGateway)
    om_mock.get_entity_context.return_value = {
        "details": {"name": "customer_table", "columns": [{"name": "ssn"}]},
        "lineage": {},
    }

    gov_mock = MagicMock(spec=GovernanceGateway)
    gov_mock.inspect_ranger_state.return_value = {"services": ["trino"]}

    tag_classifier_mock = MagicMock()
    policy_classifier_mock = MagicMock()
    policy_classifier_mock.model_name = "gpt-4o-mini"
    policy_classifier_mock.reason_policy.return_value = PolicyReasoningResult(
        proposal=LogicalPolicyProposal(
            subjects=[Subject(subject_type="GROUP", name="analytics")],
            resource=PolicyResource(database="service", schema_name="db", table="customer_table"),
            access=["SELECT"],
            masks=[],
            row_filter=None,
        ),
        rationale="Analytics team needs read access to customer table",
        expected_impact="Allows SELECT access for analytics group",
        confidence=0.9,
    )

    tag_result, policy_result, context = run_governance_graph(
        om_gateway=om_mock,
        gov_gateway=gov_mock,
        tag_classifier=tag_classifier_mock,
        policy_classifier=policy_classifier_mock,
        request_type="POLICY",
        entity_type="table",
        entity_fqn="service.db.schema.customer_table",
        allowed_tags=[],
        include_lineage=True,
        target_subjects=[Subject(subject_type="GROUP", name="analytics")],
        policy_intent="Grant read access",
    )

    assert tag_result is None
    assert policy_result is not None
    assert policy_result.proposal is not None
    assert policy_result.proposal.subjects[0].subject_type == "GROUP"
    assert policy_result.proposal.subjects[0].name == "analytics"
    gov_mock.inspect_ranger_state.assert_called_once()
