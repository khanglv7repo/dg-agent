"""Stale-safe Agent-side consumer logic for Backend ai.classification handoff."""
from __future__ import annotations

from typing import Any, Protocol

from app.classifier import StructuredClassifier
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway


class ClassificationCompletionChannel(Protocol):
    """Future external Backend callback contract.

    Frozen R5 does not provide a supported implementation. Production therefore
    passes None and the worker stops fail-safe before authoritative OM mutation.
    """

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

        # R5 has no supported generation-fenced completion callback. Mutating OM
        # without a durable completion channel would leave WAITING_AI stuck and
        # invite retries. Fail safe before authority mutation.
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
