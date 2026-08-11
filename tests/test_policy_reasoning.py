from __future__ import annotations

from unittest.mock import MagicMock

from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway
from app.graph import run_governance_graph
from app.schemas import (
    ColumnMask,
    LogicalPolicyProposal,
    PolicyReasoningResult,
    PolicyResource,
    RowFilter,
    Subject,
)


def _proposal() -> LogicalPolicyProposal:
    return LogicalPolicyProposal(
        subjects=[Subject(subject_type="USER", name="alice")],
        resource=PolicyResource(
            catalog="financial",
            schema="crm",
            table="customers",
        ),
        access={"select": "ALLOW"},
        masks=[ColumnMask(column="phone", mask_type="MASK")],
        row_filter=RowFilter(expression="customer_id <= 10"),
    )


def test_policy_flow_preview_conflict_and_draft_without_activation() -> None:
    om = MagicMock(spec=OpenMetadataGateway)
    om.get_entity_context.return_value = {
        "details": {
            "name": "customers",
            "service": {"name": "financial_postgres"},
        },
        "lineage": {},
    }
    gov = MagicMock(spec=GovernanceGateway)
    gov.inspect_ranger_state.return_value = {"kind": "health"}
    gov.resolve_resource_mapping.return_value = {
        "trino_catalog": "financial",
        "ranger_service_name": "dev_trino",
    }
    gov.get_policy.side_effect = Exception("should not be required for new key")
    # Avoid existing-policy branch failure by using Backend NOT_FOUND semantics in a
    # focused test below; here return a simple current policy.
    gov.get_policy.side_effect = None
    gov.get_policy.return_value = {"status": "ACTIVE", "version": 1}
    gov.list_policy_versions.return_value = [{"version": 1}]
    gov.get_ranger_sync_status.return_value = {"projections": []}
    gov.check_policy_conflict.return_value = {
        "conflict": False,
        "requires_review": False,
    }
    gov.preview_policy_change.return_value = {
        "projections": [
            {"projection_type": "ACCESS"},
            {"projection_type": "MASK"},
            {"projection_type": "ROW_FILTER"},
        ]
    }
    gov.create_policy_version.return_value = {
        "status": "DRAFT",
        "version": 2,
        "authority_changed": False,
        "dispatched": False,
    }

    classifier = MagicMock()
    classifier.reason_policy.return_value = PolicyReasoningResult(
        proposal=_proposal(),
        rationale="Explicit alice access proposal",
    )

    _, result, _ = run_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=MagicMock(),
        policy_classifier=classifier,
        request_type="POLICY",
        entity_type="table",
        entity_fqn="financial.crm.customers",
        target_subjects=[Subject(subject_type="USER", name="alice")],
        policy_intent="Mask phone and filter customer rows",
        policy_key="r6b-isolated-draft",
        persist_draft=True,
        environment="local",
    )

    assert result is not None
    assert result.backend_logical_policy["subjects"] == [
        {"type": "USER", "name": "alice"}
    ]
    assert result.conflict["conflict"] is False
    assert {p["projection_type"] for p in result.preview["projections"]} == {
        "ACCESS",
        "MASK",
        "ROW_FILTER",
    }
    assert result.draft["status"] == "DRAFT"
    gov.create_policy_version.assert_called_once()
    gov.activate_policy_version.assert_not_called()
    gov.rollback_policy.assert_not_called()
    gov.update_service_mapping.assert_not_called()
    gov.request_ranger_sync.assert_not_called()


def test_persist_draft_without_explicit_subjects_does_not_write() -> None:
    om = MagicMock(spec=OpenMetadataGateway)
    om.get_entity_context.return_value = {"details": {}, "lineage": {}}
    gov = MagicMock(spec=GovernanceGateway)
    gov.inspect_ranger_state.return_value = {"kind": "health"}
    classifier = MagicMock()
    classifier.reason_policy.return_value = PolicyReasoningResult(
        proposal=_proposal(),
        rationale="Conceptual proposal",
    )

    _, result, _ = run_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=MagicMock(),
        policy_classifier=classifier,
        request_type="POLICY",
        entity_fqn="financial.crm.customers",
        policy_key="r6b-no-subject",
        persist_draft=True,
    )
    assert result is not None
    assert result.draft is None
    gov.create_policy_version.assert_not_called()


def test_unresolved_service_mapping_blocks_draft_persistence() -> None:
    om = MagicMock(spec=OpenMetadataGateway)
    om.get_entity_context.return_value = {
        "details": {
            "name": "customers",
            "service": {"name": "unknown_postgres"},
        },
        "lineage": {},
    }
    gov = MagicMock(spec=GovernanceGateway)
    gov.inspect_ranger_state.return_value = {"kind": "health"}
    gov.resolve_resource_mapping.side_effect = RuntimeError("UNRESOLVED")
    gov.check_policy_conflict.return_value = {"conflict": False}
    gov.preview_policy_change.return_value = {"projections": []}

    classifier = MagicMock()
    classifier.reason_policy.return_value = PolicyReasoningResult(
        proposal=_proposal(),
        rationale="Explicit policy proposal",
    )

    _, result, _ = run_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=MagicMock(),
        policy_classifier=classifier,
        request_type="POLICY",
        entity_type="table",
        entity_fqn="financial.crm.customers",
        target_subjects=[Subject(subject_type="USER", name="alice")],
        policy_intent="Mask phone",
        policy_key="r6b-unmapped",
        persist_draft=True,
        environment="local",
    )

    assert result is not None
    assert result.draft is None
    assert any("service mapping" in warning.lower() for warning in result.warnings)
    gov.create_policy_version.assert_not_called()


def test_mapping_catalog_mismatch_blocks_draft_persistence() -> None:
    om = MagicMock(spec=OpenMetadataGateway)
    om.get_entity_context.return_value = {
        "details": {
            "name": "customers",
            "service": {"name": "financial_postgres"},
        },
        "lineage": {},
    }
    gov = MagicMock(spec=GovernanceGateway)
    gov.inspect_ranger_state.return_value = {"kind": "health"}
    gov.resolve_resource_mapping.return_value = {
        "trino_catalog": "warehouse",
        "ranger_service_name": "dev_trino",
    }
    gov.get_policy.return_value = {"status": "ACTIVE", "version": 1}
    gov.list_policy_versions.return_value = [{"version": 1}]
    gov.get_ranger_sync_status.return_value = {"projections": []}
    gov.check_policy_conflict.return_value = {"conflict": False}
    gov.preview_policy_change.return_value = {"projections": []}

    classifier = MagicMock()
    classifier.reason_policy.return_value = PolicyReasoningResult(
        proposal=_proposal(),
        rationale="Explicit policy proposal",
    )

    _, result, _ = run_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=MagicMock(),
        policy_classifier=classifier,
        request_type="POLICY",
        entity_type="table",
        entity_fqn="financial.crm.customers",
        target_subjects=[Subject(subject_type="USER", name="alice")],
        policy_intent="Mask phone",
        policy_key="r6b-mapping-mismatch",
        persist_draft=True,
        environment="local",
    )

    assert result is not None
    assert result.draft is None
    assert any("catalog" in warning.lower() for warning in result.warnings)
    gov.create_policy_version.assert_not_called()
