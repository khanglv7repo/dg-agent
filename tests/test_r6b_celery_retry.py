from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.schemas import TagRecommendation, TagReasoningResult
from app.services.classification_worker import ClassificationCompletionBoundError
from app.tasks import classification as task_module


EXECUTION_ID = "2c74ee42-7dd4-491b-9ec2-57477b9fa6ba"
GENERATION = 5
ENTITY_FQN = "financial_postgres.financial_db.analytics.customer_360"


class RetryRequested(RuntimeError):
    pass


def workflow() -> dict:
    return {
        "source": "classification_execution",
        "id": EXECUTION_ID,
        "entity_type": "table",
        "entity_fqn": ENTITY_FQN,
        "generation": GENERATION,
        "status": "WAITING_AI",
    }


class FakeGovernance:
    """No create-generation capability exists in this fake by construction."""

    def __init__(self) -> None:
        self.workflow_reads: list[str] = []
        self.completion_calls: list[dict] = []
        self.close_calls = 0

    def get_workflow_status(self, execution_id: str) -> dict:
        self.workflow_reads.append(execution_id)
        return workflow()

    def complete_classification_execution(
        self,
        *,
        execution_id: str,
        generation: int,
        status: str,
        result: dict,
    ) -> dict:
        self.completion_calls.append(
            {
                "execution_id": execution_id,
                "generation": generation,
                "status": status,
                "result": result,
            }
        )
        if len(self.completion_calls) == 1:
            # First attempt has already applied the OM tag; completion transport
            # then fails transiently before Backend terminal state is recorded.
            raise RuntimeError("transient completion transport failure")
        return {
            "status": "COMPLETED",
            "execution_id": execution_id,
            "generation": generation,
            "authority_changed": True,
            "duplicate": False,
            "stale": False,
        }

    def close(self) -> None:
        self.close_calls += 1


class FakeOpenMetadata:
    def __init__(self) -> None:
        self.authoritative_tag_present = False
        self.authoritative_mutations = 0
        self.apply_results: list[str] = []
        self.context_reads = 0
        self.taxonomy_reads = 0
        self.close_calls = 0

    def get_entity_context(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        include_lineage: bool,
    ) -> dict:
        self.context_reads += 1
        return {
            "details": {
                "fullyQualifiedName": entity_fqn,
                "columns": [
                    {
                        "name": "customer_name",
                        "fullyQualifiedName": f"{entity_fqn}.customer_name",
                    }
                ],
            }
        }

    def get_taxonomies(self) -> list[str]:
        self.taxonomy_reads += 1
        return ["PII.Name"]

    def apply_tag_authoritative(
        self,
        *,
        entity_type: str,
        entity_fqn: str,
        tag_fqn: str,
        field_path: str | None = None,
    ) -> dict:
        if not self.authoritative_tag_present:
            self.authoritative_tag_present = True
            self.authoritative_mutations += 1
            self.apply_results.append("APPLIED")
            return {
                "status": "APPLIED",
                "entity_fqn": entity_fqn,
                "field_path": field_path,
                "tag_fqn": tag_fqn,
                "mutation_count": 1,
                "transport": "NATIVE_API",
            }

        self.apply_results.append("NO_CHANGE")
        return {
            "status": "NO_CHANGE",
            "entity_fqn": entity_fqn,
            "field_path": field_path,
            "tag_fqn": tag_fqn,
            "mutation_count": 0,
        }

    def close(self) -> None:
        self.close_calls += 1


class FakeClassifier:
    def classify(self, *, catalog_context: dict, allowed_tags: list[str]):
        return TagReasoningResult(
            recommendations=[
                TagRecommendation(
                    tag="PII.Name",
                    confidence=0.95,
                    rationale="customer_name is a name",
                    field_path=f"{ENTITY_FQN}.customer_name",
                    action_recommendation="APPLY",
                )
            ]
        )


def install_task_fakes(monkeypatch):
    governance = FakeGovernance()
    openmetadata = FakeOpenMetadata()
    classifier = FakeClassifier()

    monkeypatch.setattr(
        task_module,
        "GovernanceGateway",
        lambda *, endpoint: governance,
    )
    monkeypatch.setattr(
        task_module,
        "OpenMetadataGateway",
        lambda *, endpoint, token: openmetadata,
    )

    runtime = MagicMock()
    runtime.tag_classifier.return_value = classifier
    runtime_type = MagicMock()
    runtime_type.from_env.return_value = runtime
    monkeypatch.setattr(task_module, "LLMRuntimeConfig", runtime_type)

    return governance, openmetadata


def test_real_bounded_retry_recovers_same_generation_without_duplicate_authority(
    monkeypatch,
) -> None:
    governance, openmetadata = install_task_fakes(monkeypatch)

    retry_mock = MagicMock(
        side_effect=RetryRequested("celery retry scheduled")
    )
    monkeypatch.setattr(
        task_module.ai_classify_entity,
        "retry",
        retry_mock,
    )

    assert task_module.ai_classify_entity.max_retries == 2
    assert task_module.CLASSIFICATION_RETRY_DELAY_SECONDS == 2

    # Attempt 1: OM authoritative tag is applied, then completion fails.
    with pytest.raises(RetryRequested):
        task_module.ai_classify_entity.run(
            execution_id=EXECUTION_ID,
            generation=GENERATION,
        )

    assert openmetadata.authoritative_mutations == 1
    assert openmetadata.apply_results == ["APPLIED"]
    assert len(governance.completion_calls) == 1

    retry_mock.assert_called_once()
    retry_kwargs = retry_mock.call_args.kwargs
    assert isinstance(retry_kwargs["exc"], RuntimeError)
    assert "transient completion" in str(retry_kwargs["exc"])
    assert retry_kwargs["countdown"] == 2

    # Simulate Celery's scheduled retry using the exact same task payload.
    result = task_module.ai_classify_entity.run(
        execution_id=EXECUTION_ID,
        generation=GENERATION,
    )

    assert result["status"] == "COMPLETED"
    assert result["execution_id"] == EXECUTION_ID
    assert result["generation"] == GENERATION

    # Every attempt re-reads both Backend state and OM state/context.
    assert governance.workflow_reads == [EXECUTION_ID] * 4
    assert openmetadata.context_reads == 2
    assert openmetadata.taxonomy_reads == 2

    # Retry observes already-applied authority and produces NO_CHANGE.
    assert openmetadata.apply_results == ["APPLIED", "NO_CHANGE"]
    assert openmetadata.authoritative_mutations == 1
    assert result["om_mutation_count"] == 0

    # Both completion attempts are for the exact same durable generation.
    assert len(governance.completion_calls) == 2
    assert {
        (call["execution_id"], call["generation"])
        for call in governance.completion_calls
    } == {(EXECUTION_ID, GENERATION)}

    # There is no producer/new-generation operation in this retry path.
    assert not hasattr(governance, "create_classification_generation")


def test_deterministic_completion_bound_failure_is_not_retried(
    monkeypatch,
) -> None:
    governance = MagicMock()
    openmetadata = MagicMock()
    runtime = MagicMock()
    runtime.tag_classifier.return_value = MagicMock()
    runtime_type = MagicMock()
    runtime_type.from_env.return_value = runtime

    monkeypatch.setattr(
        task_module,
        "GovernanceGateway",
        lambda *, endpoint: governance,
    )
    monkeypatch.setattr(
        task_module,
        "OpenMetadataGateway",
        lambda *, endpoint, token: openmetadata,
    )
    monkeypatch.setattr(task_module, "LLMRuntimeConfig", runtime_type)

    class BoundFailingWorker:
        def __init__(self, **kwargs) -> None:
            pass

        def handle(self, *, execution_id: str, generation: int):
            raise ClassificationCompletionBoundError(count=21)

    monkeypatch.setattr(
        task_module,
        "ClassificationWorkerService",
        BoundFailingWorker,
    )

    retry_mock = MagicMock()
    monkeypatch.setattr(
        task_module.ai_classify_entity,
        "retry",
        retry_mock,
    )

    with pytest.raises(ClassificationCompletionBoundError):
        task_module.ai_classify_entity.run(
            execution_id=EXECUTION_ID,
            generation=GENERATION,
        )

    retry_mock.assert_not_called()
    openmetadata.close.assert_called_once_with()
    governance.close.assert_called_once_with()
