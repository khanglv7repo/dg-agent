from __future__ import annotations

from unittest.mock import MagicMock

from app.schemas import TagRecommendation, TagReasoningResult
from app.services.classification_worker import ClassificationWorkerService


def workflow(*, status: str = "WAITING_AI", generation: int = 2) -> dict:
    return {
        "source": "classification_execution",
        "id": "exec-1",
        "entity_type": "table",
        "entity_fqn": "financial.crm.customers",
        "generation": generation,
        "status": status,
    }


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
