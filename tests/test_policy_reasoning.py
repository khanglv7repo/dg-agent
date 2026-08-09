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


def test_policy_reasoning_emits_logical_schema_not_ranger_json() -> None:
    policy_classifier_mock = MagicMock()
    policy_classifier_mock.reason_policy.return_value = PolicyReasoningResult(
        proposal=LogicalPolicyProposal(
            subjects=[
                Subject(subject_type="USER", name="alice"),
                Subject(subject_type="GROUP", name="finance"),
            ],
            resource=PolicyResource(database="catalog", schema_name="analytics", table="orders"),
            access=["SELECT"],
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
        entity_fqn="catalog.analytics.orders",
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

    # Verify logical resource and masks
    assert proposal.resource.database == "catalog"
    assert proposal.resource.table == "orders"
    assert len(proposal.masks) == 1
    assert proposal.masks[0].column == "credit_card"
    assert proposal.masks[0].mask_type == "MASK_HASH"
    assert proposal.row_filter is not None
    assert proposal.row_filter.expression == "region = 'US'"

    # Verify native Ranger JSON structure (e.g. policyItems, allowExceptions) is NOT present
    raw_dict = proposal.model_dump()
    assert "policyItems" not in raw_dict
    assert "allowExceptions" not in raw_dict
