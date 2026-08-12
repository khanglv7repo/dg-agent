"""Stale-safe Agent-side consumer logic for Backend ai.classification handoff."""
from __future__ import annotations

from typing import Any, Protocol

from app.classifier import StructuredClassifier
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway

# Must remain aligned with Backend R6-B ClassificationCompletionService.
# Backend currently accepts at most 20 recommendations and 20 mutation records.
MAX_COMPLETION_RECOMMENDATIONS = 20


class ClassificationCompletionBoundError(RuntimeError):
    """Fail-closed guard when Agent APPLY output cannot fit Backend completion."""

    def __init__(
        self,
        *,
        count: int,
        limit: int = MAX_COMPLETION_RECOMMENDATIONS,
    ) -> None:
        self.count = int(count)
        self.limit = int(limit)
        super().__init__(
            "R6-B APPLY recommendation count exceeds Backend completion limit: "
            f"{self.count} > {self.limit}"
        )


class ClassificationCompletionChannel(Protocol):
    """Bounded Backend R6-B continuation for the same dispatched generation."""

    def complete(
        self,
        *,
        execution_id: str,
        generation: int,
        status: str,
        result: dict[str, Any],
    ) -> dict[str, Any]: ...


class ClassificationWorkerService:
    def __init__(
        self,
        *,
        governance: GovernanceGateway,
        openmetadata: OpenMetadataGateway,
        classifier: StructuredClassifier,
        completion: ClassificationCompletionChannel | None = None,
    ) -> None:
        self.governance = governance
        self.openmetadata = openmetadata
        self.classifier = classifier
        self.completion = completion

    @staticmethod
    def _fence(
        workflow: dict[str, Any],
        *,
        execution_id: str,
        generation: int,
    ) -> str | None:
        if workflow.get("source") != "classification_execution":
            return "WRONG_WORKFLOW_SOURCE"
        if str(workflow.get("id")) != str(execution_id):
            return "EXECUTION_ID_MISMATCH"
        status = str(workflow.get("status") or "")
        if status != "WAITING_AI":
            return status or "NOT_WAITING_AI"
        try:
            current_generation = int(workflow.get("generation"))
        except (TypeError, ValueError):
            return "INVALID_GENERATION"
        if current_generation != int(generation):
            return "SUPERSEDED"
        return None

    def handle(self, *, execution_id: str, generation: int) -> dict[str, Any]:
        first = self.governance.get_workflow_status(execution_id)
        stale = self._fence(
            first,
            execution_id=execution_id,
            generation=generation,
        )
        if stale:
            return {
                "status": "NOOP",
                "reason": stale,
                "execution_id": execution_id,
                "generation": generation,
                "om_mutation_count": 0,
            }

        entity_type = str(first["entity_type"])
        entity_fqn = str(first["entity_fqn"])

        context = self.openmetadata.get_entity_context(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            include_lineage=True,
        )
        taxonomy = self.openmetadata.get_taxonomies()
        reasoning = self.classifier.classify(
            catalog_context=context,
            allowed_tags=taxonomy,
        )
        allowed = set(taxonomy)
        apply_recommendations = [
            rec
            for rec in reasoning.recommendations
            if rec.action_recommendation == "APPLY" and rec.tag in allowed
        ]

        # Cross-system invariant: never mutate more authoritative OM targets than
        # Backend can durably accept in the same completion transaction.
        # This guard deliberately runs before fence #2 and before the first OM write.
        if len(apply_recommendations) > MAX_COMPLETION_RECOMMENDATIONS:
            raise ClassificationCompletionBoundError(
                count=len(apply_recommendations)
            )

        # Mandatory immediate second stale-generation fence before any OM write.
        second = self.governance.get_workflow_status(execution_id)
        stale = self._fence(
            second,
            execution_id=execution_id,
            generation=generation,
        )
        if stale:
            return {
                "status": "NOOP",
                "reason": stale,
                "execution_id": execution_id,
                "generation": generation,
                "om_mutation_count": 0,
            }

        # Production R6-B injects a generation-fenced Backend completion adapter.
        # If it is absent, fail safe before authoritative OM mutation.
        if self.completion is None:
            return {
                "status": "BLOCKED_COMPLETION_CHANNEL",
                "decision": "APPLY" if apply_recommendations else "NO_PROPOSAL",
                "execution_id": execution_id,
                "generation": generation,
                "recommendation_count": len(apply_recommendations),
                "om_mutation_count": 0,
            }

        mutations: list[dict[str, Any]] = []
        for rec in apply_recommendations:
            mutations.append(
                self.openmetadata.apply_tag_authoritative(
                    entity_type=entity_type,
                    entity_fqn=entity_fqn,
                    tag_fqn=rec.tag,
                    field_path=rec.field_path,
                )
            )

        completion_status = "COMPLETED" if apply_recommendations else "NO_PROPOSAL"
        completion_result = self.completion.complete(
            execution_id=execution_id,
            generation=generation,
            status=completion_status,
            result={
                "entity_type": entity_type,
                "entity_fqn": entity_fqn,
                "recommendations": [
                    rec.model_dump(mode="json") for rec in apply_recommendations
                ],
                "mutations": mutations,
            },
        )
        return {
            "status": completion_status,
            "execution_id": execution_id,
            "generation": generation,
            "om_mutation_count": sum(
                int(item.get("mutation_count", 0)) for item in mutations
            ),
            "completion": completion_result,
        }
