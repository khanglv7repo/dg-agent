"""Exact consumer task for Backend R3/R6-B ai.classification handoff."""
from __future__ import annotations

import os
from typing import Any

from app.celery_app import app
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata_context import OpenMetadataGateway
from app.llm_runtime import LLMRuntimeConfig
from app.services.classification_completion import BackendClassificationCompletionChannel
from app.services.classification_worker import (
    ClassificationCompletionBoundError,
    ClassificationWorkerService,
)

CLASSIFICATION_RETRY_DELAY_SECONDS = 2


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
    """Run one generation-fenced attempt, retrying unexpected execution failures.

    Celery retry keeps the original task arguments, so recovery uses the same
    execution_id and generation. ClassificationWorkerService then re-reads
    Backend state and OpenMetadata on every attempt.
    """

    governance = None
    openmetadata = None

    try:
        governance = GovernanceGateway(
            endpoint=os.getenv(
                "BACKEND_MCP_URL",
                "http://127.0.0.1:8001/mcp",
            )
        )
        openmetadata = OpenMetadataGateway(
            endpoint=os.getenv(
                "OPENMETADATA_MCP_URL",
                "http://localhost:8585/mcp",
            ),
            token=os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", ""),
        )

        # The worker uses the same Responses API / structured-output configuration
        # as GovernanceAgentRunner. Do not duplicate provider options here.
        classifier = LLMRuntimeConfig.from_env().tag_classifier()
        completion = BackendClassificationCompletionChannel(governance)

        service = ClassificationWorkerService(
            governance=governance,
            openmetadata=openmetadata,
            classifier=classifier,
            completion=completion,
        )
        return service.handle(
            execution_id=execution_id,
            generation=generation,
        )

    except ClassificationCompletionBoundError:
        # Deterministic cross-system validation failure. Retrying the same LLM
        # result would not make 21+ APPLY targets fit Backend's bounded contract.
        raise

    except Exception as exc:
        # Real bounded recovery path. Celery enforces max_retries=2; once exhausted
        # the original exception fails visibly. No new classification generation
        # is created by this task.
        raise self.retry(
            exc=exc,
            countdown=CLASSIFICATION_RETRY_DELAY_SECONDS,
        )

    finally:
        if openmetadata is not None:
            openmetadata.close()
        if governance is not None:
            governance.close()
