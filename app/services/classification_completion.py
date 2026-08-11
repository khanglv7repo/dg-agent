"""Agent adapter for Backend's R6-B classification completion continuation."""
from __future__ import annotations

from typing import Any, Literal

from app.gateways.governance import GovernanceGateway


class BackendClassificationCompletionChannel:
    """Thin typed adapter; Backend remains authoritative for durable completion."""

    def __init__(self, governance: GovernanceGateway) -> None:
        self.governance = governance

    def complete(
        self,
        *,
        execution_id: str,
        generation: int,
        status: Literal["COMPLETED", "NO_PROPOSAL"],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return self.governance.complete_classification_execution(
            execution_id=execution_id,
            generation=generation,
            status=status,
            result=result,
        )
