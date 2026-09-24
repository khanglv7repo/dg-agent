from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.schemas import TagRecommendation, TagReasoningResult
from app.services.classification_worker import (
    MAX_COMPLETION_RECOMMENDATIONS,
    ClassificationCompletionBoundError,
    ClassificationWorkerService,
)


def workflow(*, status: str = "WAITING_AI", generation: int = 2) -> dict:
    return {
        "source": "classification_execution",
        "id": "exec-1",
        "entity_type": "table",
        "entity_fqn": "financial.crm.customers",
        "generation": generation,
        "status": status,
    }


def recommendation(index: int, *, action: str = "APPLY") -> TagRecommendation:
    return TagRecommendation(
        tag=f"PII.Tag{index}",
        confidence=0.9,
        rationale=f"recommendation {index}",
        field_path=f"financial.crm.customers.column_{index}",
        action_recommendation=action,
    )


def configured_service(
    recommendations: list[TagRecommendation],
) -> tuple[
    ClassificationWorkerService,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    gov = MagicMock()
    gov.get_workflow_status.side_effect = [workflow(), workflow()]

    om = MagicMock()
    om.get_entity_context.return_value = {"details": {}}
    om.get_taxonomies.return_value = sorted({rec.tag for rec in recommendations})
    om.apply_tag_authoritative.return_value = {
        "status": "APPLIED",
        "entity_fqn": "financial.crm.customers",
        "field_path": "field",
        "tag_fqn": "PII.Tag",
        "mutation_count": 1,
        "transport": "NATIVE_API",
    }

    classifier = MagicMock()
    classifier.model_name = "test-model"
    classifier.prompt_version = "v3"
    classifier.classify.return_value = TagReasoningResult(
        recommendations=recommendations
    )

    completion = MagicMock()
    completion.complete.return_value = {
        "status": "COMPLETED",
        "authority_changed": True,
    }

    service = ClassificationWorkerService(
        governance=gov,
        openmetadata=om,
        classifier=classifier,
        completion=completion,
    )
    return service, gov, om, classifier, completion


def test_generation_mismatch_is_zero_mutation() -> None:
    gov = MagicMock()
    gov.get_workflow_status.return_value = workflow(generation=3)
    om = MagicMock()
    service = ClassificationWorkerService(
        governance=gov,
        openmetadata=om,
        classifier=MagicMock(),
    )
    result = service.handle(execution_id="exec-1", generation=2)
    assert result["status"] == "NOOP"
    assert result["om_mutation_count"] == 0
    om.apply_tag_authoritative.assert_not_called()


def test_completed_retry_is_zero_mutation() -> None:
    gov = MagicMock()
    gov.get_workflow_status.return_value = workflow(status="COMPLETED")
    om = MagicMock()
    service = ClassificationWorkerService(
        governance=gov,
        openmetadata=om,
        classifier=MagicMock(),
    )
    result = service.handle(execution_id="exec-1", generation=2)
    assert result["status"] == "NOOP"
    om.apply_tag_authoritative.assert_not_called()


def test_second_fence_blocks_superseded_execution() -> None:
    gov = MagicMock()
    gov.get_workflow_status.side_effect = [
        workflow(),
        workflow(status="SUPERSEDED"),
    ]
    om = MagicMock()
    om.get_entity_context.return_value = {"details": {}}
    om.get_taxonomies.return_value = ["PII.Email"]
    classifier = MagicMock()
    classifier.classify.return_value = TagReasoningResult(
        recommendations=[
            TagRecommendation(
                tag="PII.Email",
                confidence=0.9,
                rationale="email",
                action_recommendation="APPLY",
            )
        ]
    )
    service = ClassificationWorkerService(
        governance=gov,
        openmetadata=om,
        classifier=classifier,
    )
    result = service.handle(execution_id="exec-1", generation=2)
    assert result["status"] == "NOOP"
    assert result["om_mutation_count"] == 0
    om.apply_tag_authoritative.assert_not_called()


def test_missing_completion_channel_fails_safe_before_om_mutation() -> None:
    gov = MagicMock()
    gov.get_workflow_status.side_effect = [workflow(), workflow()]
    om = MagicMock()
    om.get_entity_context.return_value = {"details": {}}
    om.get_taxonomies.return_value = ["PII.Email"]
    classifier = MagicMock()
    classifier.classify.return_value = TagReasoningResult(
        recommendations=[
            TagRecommendation(
                tag="PII.Email",
                confidence=0.9,
                rationale="email",
                action_recommendation="APPLY",
            )
        ]
    )
    service = ClassificationWorkerService(
        governance=gov,
        openmetadata=om,
        classifier=classifier,
        completion=None,
    )
    result = service.handle(execution_id="exec-1", generation=2)
    assert result["status"] == "BLOCKED_COMPLETION_CHANNEL"
    assert result["decision"] == "APPLY"
    assert result["om_mutation_count"] == 0
    om.apply_tag_authoritative.assert_not_called()


def test_valid_flow_with_injected_completion_channel_is_idempotent() -> None:
    gov = MagicMock()
    gov.get_workflow_status.side_effect = [workflow(), workflow()]
    om = MagicMock()
    om.get_entity_context.return_value = {"details": {}}
    om.get_taxonomies.return_value = ["PII.Email"]
    om.apply_tag_authoritative.return_value = {
        "status": "NO_CHANGE",
        "mutation_count": 0,
    }
    classifier = MagicMock()
    classifier.classify.return_value = TagReasoningResult(
        recommendations=[
            TagRecommendation(
                tag="PII.Email",
                confidence=0.9,
                rationale="email",
                action_recommendation="APPLY",
            )
        ]
    )
    completion = MagicMock()
    completion.complete.return_value = {"status": "COMPLETED"}
    service = ClassificationWorkerService(
        governance=gov,
        openmetadata=om,
        classifier=classifier,
        completion=completion,
    )
    result = service.handle(execution_id="exec-1", generation=2)
    assert result["status"] == "COMPLETED"
    assert result["om_mutation_count"] == 0
    completion.complete.assert_called_once()


def test_exactly_20_apply_is_allowed_and_completion_contains_20() -> None:
    recommendations = [
        recommendation(index)
        for index in range(MAX_COMPLETION_RECOMMENDATIONS)
    ]
    service, gov, om, classifier, completion = configured_service(recommendations)

    result = service.handle(execution_id="exec-1", generation=2)

    assert result["status"] == "COMPLETED"
    assert om.apply_tag_authoritative.call_count == 20
    completion.complete.assert_called_once()

    call = completion.complete.call_args.kwargs
    assert len(call["result"]["recommendations"]) == 20
    assert len(call["result"]["mutations"]) == 20
    assert gov.get_workflow_status.call_count == 2


# ---------------------------------------------------------------------------
# TASK-09: I11 kill switches + I9 audit ref on the classification write boundary
# ---------------------------------------------------------------------------

def test_master_kill_switch_skips_all_om_reads_and_writes() -> None:
    """agent_write_to_om_enabled=False must fail safe before any OM read/write,
    not just before the mutation -- the Agent must not touch OM at all."""
    gov = MagicMock()
    gov.get_workflow_status.return_value = workflow()
    om = MagicMock()
    classifier = MagicMock()
    classifier.model_name = "test-model"
    classifier.prompt_version = "v3"
    service = ClassificationWorkerService(
        governance=gov,
        openmetadata=om,
        classifier=classifier,
    )
    result = service.handle(
        execution_id="exec-1",
        generation=2,
        agent_write_to_om_enabled=False,
    )
    assert result["status"] == "SKIPPED"
    assert result["reason_code"] == "KILL_SWITCH_DISABLED"
    assert result["om_mutation_count"] == 0
    om.get_entity_context.assert_not_called()
    om.get_taxonomies.assert_not_called()
    om.apply_tag_authoritative.assert_not_called()
    classifier.classify.assert_not_called()
    audit_ref = result["audit_ref"]
    assert audit_ref["reason_code"] == "KILL_SWITCH_DISABLED"
    assert audit_ref["validated_at"] is None
    assert audit_ref["model_fingerprint"]
    assert audit_ref["prompt_version"] == "v3"


def test_auto_apply_kill_switch_downgrades_apply_to_no_proposal() -> None:
    """auto_apply_tag_enabled=False must downgrade APPLY to SUGGEST-only:
    no OM mutation, Backend sees NO_PROPOSAL, reason code records why."""
    recommendations = [recommendation(1)]
    service, gov, om, classifier, completion = configured_service(recommendations)

    result = service.handle(
        execution_id="exec-1",
        generation=2,
        auto_apply_tag_enabled=False,
    )

    assert result["status"] == "NO_PROPOSAL"
    assert result["reason_code"] == "KILL_SWITCH_DISABLED"
    assert result["om_mutation_count"] == 0
    om.apply_tag_authoritative.assert_not_called()
    completion.complete.assert_called_once()
    call = completion.complete.call_args.kwargs
    assert call["result"]["recommendations"] == []
    assert call["result"]["mutations"] == []
    audit_ref = result["audit_ref"]
    assert audit_ref["reason_code"] == "KILL_SWITCH_DISABLED"
    assert audit_ref["validated_at"] is not None


def test_audit_ref_present_and_distinct_validated_at_on_normal_apply() -> None:
    """Hard Invariant #13: validated_at must be a real timestamp, set before
    the OM mutation / Backend completion call (I8 validate-before-write)."""
    recommendations = [recommendation(1)]
    service, gov, om, classifier, completion = configured_service(recommendations)

    result = service.handle(execution_id="exec-1", generation=2)

    assert result["status"] == "COMPLETED"
    assert result["reason_code"] == "RULE_TRUSTED_AUTO_APPLY"
    audit_ref = result["audit_ref"]
    assert audit_ref["model_fingerprint"]
    assert audit_ref["prompt_version"] == "v3"
    assert audit_ref["input_fingerprint"]
    assert audit_ref["validated_at"] is not None
    assert om.apply_tag_authoritative.call_count == 1


def test_no_match_reason_code_when_no_recommendations_survive_filtering() -> None:
    gov = MagicMock()
    gov.get_workflow_status.side_effect = [workflow(), workflow()]
    om = MagicMock()
    om.get_entity_context.return_value = {"details": {}}
    om.get_taxonomies.return_value = ["PII.Email"]
    classifier = MagicMock()
    classifier.model_name = "test-model"
    classifier.prompt_version = "v3"
    classifier.classify.return_value = TagReasoningResult(recommendations=[])
    completion = MagicMock()
    completion.complete.return_value = {"status": "NO_PROPOSAL"}
    service = ClassificationWorkerService(
        governance=gov,
        openmetadata=om,
        classifier=classifier,
        completion=completion,
    )
    result = service.handle(execution_id="exec-1", generation=2)
    assert result["status"] == "NO_PROPOSAL"
    assert result["reason_code"] == "NO_MATCH"


def test_21_apply_fails_closed_before_second_fence_or_first_om_mutation() -> None:
    recommendations = [
        recommendation(index)
        for index in range(MAX_COMPLETION_RECOMMENDATIONS + 1)
    ]
    service, gov, om, classifier, completion = configured_service(recommendations)

    with pytest.raises(ClassificationCompletionBoundError) as exc_info:
        service.handle(execution_id="exec-1", generation=2)

    assert exc_info.value.count == 21
    assert exc_info.value.limit == 20
    assert gov.get_workflow_status.call_count == 1
    om.apply_tag_authoritative.assert_not_called()
    completion.complete.assert_not_called()


def test_review_and_no_action_do_not_count_toward_completion_limit() -> None:
    recommendations = [
        recommendation(index)
        for index in range(MAX_COMPLETION_RECOMMENDATIONS)
    ]
    recommendations.extend(
        recommendation(100 + index, action="REVIEW")
        for index in range(30)
    )
    recommendations.extend(
        recommendation(200 + index, action="NO_ACTION")
        for index in range(30)
    )

    service, gov, om, classifier, completion = configured_service(recommendations)
    result = service.handle(execution_id="exec-1", generation=2)

    assert result["status"] == "COMPLETED"
    assert om.apply_tag_authoritative.call_count == 20

    call = completion.complete.call_args.kwargs
    assert len(call["result"]["recommendations"]) == 20
    assert len(call["result"]["mutations"]) == 20
