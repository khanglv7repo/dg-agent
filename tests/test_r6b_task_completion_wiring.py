from __future__ import annotations

from unittest.mock import MagicMock

from app.services.classification_completion import BackendClassificationCompletionChannel
from app.tasks import classification as task_module


def test_production_celery_task_injects_backend_completion_channel(monkeypatch) -> None:
    governance = MagicMock(); openmetadata = MagicMock(); classifier = MagicMock()
    runtime = MagicMock(); runtime.tag_classifier.return_value = classifier
    monkeypatch.setattr(task_module, "GovernanceGateway", lambda *, endpoint: governance)
    monkeypatch.setattr(task_module, "OpenMetadataGateway", lambda *, endpoint, token: openmetadata)
    runtime_type = MagicMock(); runtime_type.from_env.return_value = runtime
    monkeypatch.setattr(task_module, "LLMRuntimeConfig", runtime_type)
    captured = {}

    class Worker:
        def __init__(self, *, governance, openmetadata, classifier, completion):
            captured.update(governance=governance, openmetadata=openmetadata, classifier=classifier, completion=completion)
        def handle(self, *, execution_id: str, generation: int):
            return {"status": "NO_PROPOSAL", "execution_id": execution_id, "generation": generation}

    monkeypatch.setattr(task_module, "ClassificationWorkerService", Worker)
    result = task_module.ai_classify_entity.run(execution_id="exec-1", generation=2)
    assert result["status"] == "NO_PROPOSAL"
    assert isinstance(captured["completion"], BackendClassificationCompletionChannel)
    assert captured["completion"].governance is governance
    openmetadata.close.assert_called_once_with(); governance.close.assert_called_once_with()
