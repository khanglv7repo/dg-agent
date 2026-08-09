from __future__ import annotations

from unittest.mock import MagicMock
from app.graph import run_governance_graph
from app.schemas import (
    ColumnMask,
    LogicalPolicyProposal,
    PolicyReasoningResult,
    PolicyResource,
    RowFilter,
    Subject,
)


def test_policy_reasoning_emits_logical_schema_with_catalog_and_access_allow_deny() -> None:
    policy_classifier_mock = MagicMock()
    policy_classifier_mock.reason_policy.return_value = PolicyReasoningResult(
        proposal=LogicalPolicyProposal(
            subjects=[
                Subject(subject_type="USER", name="alice"),
                Subject(subject_type="GROUP", name="finance"),
            ],
            resource=PolicyResource(catalog="lakehouse", schema="analytics", table="orders"),
            access={"select": "ALLOW", "insert": "DENY"},
            masks=[ColumnMask(column="credit_card", mask_type="MASK_HASH")],
            row_filter=RowFilter(expression="region = 'US'"),
        ),
        rationale="Finance group requires masked access to credit_card and row filtered access to region US",
        expected_impact="Protects credit card numbers while giving filtered row access",
        confidence=0.95,
        warnings=[],
    )

    _, policy_result, _ = run_governance_graph(
        om_gateway=MagicMock(),
        gov_gateway=MagicMock(),
        tag_classifier=MagicMock(),
        policy_classifier=policy_classifier_mock,
        request_type="POLICY",
        entity_fqn="lakehouse.analytics.orders",
        target_subjects=[Subject(subject_type="USER", name="alice"), Subject(subject_type="GROUP", name="finance")],
        policy_intent="Restricted order access",
    )

    assert policy_result is not None
    proposal = policy_result.proposal
    assert proposal is not None

    # Verify subjects remain explicit USER / GROUP
    assert len(proposal.subjects) == 2
    assert proposal.subjects[0].subject_type == "USER"
    assert proposal.subjects[0].name == "alice"
    assert proposal.subjects[1].subject_type == "GROUP"
    assert proposal.subjects[1].name == "finance"

    # Verify PolicyResource semantics: catalog, schema, table
    assert proposal.resource.catalog == "lakehouse"
    assert proposal.resource.schema_name == "analytics"
    assert proposal.resource.model_dump(by_alias=True)["schema"] == "analytics"
    assert proposal.resource.table == "orders"
    assert not hasattr(proposal.resource, "database") or "database" not in proposal.resource.model_dump()

    # Verify access ALLOW / DENY mapping
    assert proposal.access == {"select": "ALLOW", "insert": "DENY"}

    # Verify logical masks and row filter
    assert len(proposal.masks) == 1
    assert proposal.masks[0].column == "credit_card"
    assert proposal.masks[0].mask_type == "MASK_HASH"
    assert proposal.row_filter is not None
    assert proposal.row_filter.expression == "region = 'US'"

    # Verify native Ranger JSON structure (e.g. policyItems, allowExceptions, dataMaskInfo) is NOT present
    raw_dict = proposal.model_dump()
    assert "policyItems" not in raw_dict
    assert "allowExceptions" not in raw_dict
    assert "dataMaskInfo" not in raw_dict


def test_policy_reasoning_does_not_call_backend_activation() -> None:
    gov_mock = MagicMock()
    policy_classifier_mock = MagicMock()
    policy_classifier_mock.reason_policy.return_value = PolicyReasoningResult(
        proposal=None,
        rationale="Insufficient policy context",
        confidence=0.0,
    )

    _, policy_result, _ = run_governance_graph(
        om_gateway=MagicMock(),
        gov_gateway=gov_mock,
        tag_classifier=MagicMock(),
        policy_classifier=policy_classifier_mock,
        request_type="POLICY",
        entity_fqn="catalog.schema.table",
    )

    assert policy_result is not None
    # Verify no nonexistent backend policy mutation methods were called
    gov_mock.client.call_tool.assert_not_called()
