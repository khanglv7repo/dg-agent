"""Exact consumer task for Backend R3/R6-B ai.classification handoff."""
from __future__ import annotations

import os
from typing import Any

from app.celery_app import app
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway
from app.llm_runtime import LLMRuntimeConfig
from app.services.classification_completion import BackendClassificationCompletionChannel
from app.services.classification_worker import ClassificationWorkerService


@app.task(
    name="app.tasks.classification.ai_classify_entity",
    queue="ai.classification",
    bind=True,
    max_retries=2,
)
def ai_classify_entity(
    self,
    *,
    execution_id: str,
    generation: int,
) -> dict[str, Any]:
    governance = GovernanceGateway(
        endpoint=os.getenv("BACKEND_MCP_URL", "http://127.0.0.1:8001/mcp")
    )
    openmetadata = OpenMetadataGateway(
        endpoint=os.getenv("OPENMETADATA_MCP_URL", "http://localhost:8585/mcp"),
        token=os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", ""),
    )

    # The worker uses the same Responses API / structured-output configuration
    # as GovernanceAgentRunner. Do not duplicate provider options here.
    classifier = LLMRuntimeConfig.from_env().tag_classifier()
    completion = BackendClassificationCompletionChannel(governance)

    try:
        service = ClassificationWorkerService(
            governance=governance,
            openmetadata=openmetadata,
            classifier=classifier,
            completion=completion,
        )
        return service.handle(execution_id=execution_id, generation=generation)
    finally:
        openmetadata.close()
        governance.close()
