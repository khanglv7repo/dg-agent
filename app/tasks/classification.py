"""Exact consumer task for Backend R3/R5 ai.classification handoff."""
from __future__ import annotations

import os
from typing import Any

from app.celery_app import app
from app.classifier import OpenAIStructuredClassifier
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway
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
    classifier = OpenAIStructuredClassifier(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        base_url=os.getenv("LLM_BASE_URL") or None,
    )
    try:
        # No production completion callback exists in frozen Backend R5.
        service = ClassificationWorkerService(
            governance=governance,
            openmetadata=openmetadata,
            classifier=classifier,
            completion=None,
        )
        return service.handle(
            execution_id=execution_id,
            generation=generation,
        )
    finally:
        openmetadata.close()
        governance.close()
