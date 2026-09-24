"""Stale-safe Agent-side consumer logic for Backend ai.classification handoff."""
from __future__ import annotations

from typing import Any, Protocol

from app.audit_fingerprint import input_fingerprint, model_fingerprint, utc_now_iso
from app.classifier import StructuredClassifier
from app.gateways.governance import GovernanceGateway
# TASK-09 Work Packet B: the production task (app/tasks/classification.py) and
# GovernanceAgentRunner both construct the hardened openmetadata_context
# gateway (validated read + fail-closed tool-error detection), never the base
# app.gateways.openmetadata one. The type hint here must say so explicitly --
# importing the base class was misleading even though duck typing meant no
# runtime bug existed (the subclass always satisfies the base annotation).
from app.gateways.openmetadata_context import OpenMetadataGateway

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

    def _audit_ref(
        self,
        *,
        reason_code: str,
        input_payload: Any,
        validated_at: str | None,
    ) -> dict[str, Any]:
        """I9 audit ref: actor(model)/reason/fingerprint, validated_at distinct
        from any later created_at per Hard Invariant #13."""
        return {
            "model_fingerprint": model_fingerprint(
                model_name=self.classifier.model_name,
                prompt_version=self.classifier.prompt_version,
            ),
            "prompt_version": self.classifier.prompt_version,
            "input_fingerprint": input_fingerprint(input_payload),
            "reason_code": reason_code,
            "validated_at": validated_at,
        }

    def handle(
        self,
        *,
        execution_id: str,
        generation: int,
        agent_write_to_om_enabled: bool = True,
        auto_apply_tag_enabled: bool = True,
    ) -> dict[str, Any]:
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
                "reason_code": "STALE_GENERATION",
                "audit_ref": None,
            }

        entity_type = str(first["entity_type"])
        entity_fqn = str(first["entity_fqn"])

        # I11 master kill switch: skip OM reasoning/reads entirely, not just the
        # write. When OFF, the Agent must not touch OpenMetadata at all for this
        # execution -- fail safe and let a human/operator re-drive it later.
        if not agent_write_to_om_enabled:
            return {
                "status": "SKIPPED",
                "execution_id": execution_id,
                "generation": generation,
                "om_mutation_count": 0,
                "reason_code": "KILL_SWITCH_DISABLED",
                "audit_ref": self._audit_ref(
                    reason_code="KILL_SWITCH_DISABLED",
                    input_payload={"execution_id": execution_id, "generation": generation},
                    validated_at=None,
                ),
            }

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

        # I11 per-path kill switch: APPLY downgrades to SUGGEST-only regardless
        # of rule confidence. No OM mutation happens; recommendations are still
        # reported to Backend as NO_PROPOSAL (nothing was authoritatively
        # applied by the Agent) so a human can review and apply manually.
        auto_apply_downgraded = bool(apply_recommendations) and not auto_apply_tag_enabled
        if auto_apply_downgraded:
            apply_recommendations = []

        # Cross-system invariant: never mutate more authoritative OM targets than
        # Backend can durably accept in the same completion transaction.
        # This guard deliberately runs before fence #2 and before the first OM write.
        if len(apply_recommendations) > MAX_COMPLETION_RECOMMENDATIONS:
            raise ClassificationCompletionBoundError(
                count=len(apply_recommendations)
            )

        # Validate-before-write (I8): validation (fencing, kill-switch check,
        # bound check, taxonomy filtering) is now complete. Timestamp this
        # instant -- any later completion/mutation write must show a distinct
        # created_at, per Hard Invariant #13.
        validated_at = utc_now_iso()

        reason_code = (
            "KILL_SWITCH_DISABLED" if auto_apply_downgraded
            else ("RULE_TRUSTED_AUTO_APPLY" if apply_recommendations else "NO_MATCH")
        )
        audit_ref = self._audit_ref(
            reason_code=reason_code,
            input_payload={
                "entity_type": entity_type,
                "entity_fqn": entity_fqn,
                "allowed_tags": sorted(allowed),
            },
            validated_at=validated_at,
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
                "reason_code": "STALE_GENERATION",
                "audit_ref": audit_ref,
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
                "reason_code": reason_code,
                "audit_ref": audit_ref,
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
            "reason_code": reason_code,
            "audit_ref": audit_ref,
        }
