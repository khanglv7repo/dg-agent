"""POLICY reasoning graph boundary, extracted from the former monolithic
`graph.py` (TASK-09 Work Packet D). Pure extraction -- behavior unchanged,
verified by the full pre-existing `tests/test_policy_reasoning.py` suite
still passing unmodified against it. `load_om_context` (shared with the TAG
path) stays imported from `graph_tag.py` rather than duplicated.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.adapters.policy import PolicyAdapterError, to_backend_logical_policy
from app.audit_fingerprint import input_fingerprint, model_fingerprint, utc_now_iso
from app.classifier import PolicyClassifier
from app.clients.backend_mcp import BackendMCPError
from app.gateways.governance import GovernanceGateway
from app.schemas import AIWriteAuditRef, PolicyReasoningResult, Subject


def _format_policy_chat_reply(result: dict[str, Any], entity_fqn: str) -> str:
    """Format a policy reasoning result as a readable chat message."""
    lines = [f"**Policy reasoning for `{entity_fqn}`**\n"]

    proposal = result.get("proposal")
    if proposal and isinstance(proposal, dict):
        lines.append(f"📝 **Proposed action:** `{proposal.get('action', '?')}`")
        resource = proposal.get("resource") or {}
        if isinstance(resource, dict):
            parts = [v for k, v in resource.items() if v and k != "columns"]
            if parts:
                lines.append(f"   Resource: {' / '.join(str(p) for p in parts)}")
        subjects = proposal.get("subjects") or []
        if subjects:
            subj_str = ", ".join(
                f"{s.get('type','?')}:`{s.get('name','?')}`"
                for s in subjects if isinstance(s, dict)
            )
            lines.append(f"   Subjects: {subj_str}")
        mask = proposal.get("mask_type")
        if mask:
            lines.append(f"   Mask type: `{mask}`")
        rationale = result.get("rationale") or proposal.get("rationale")
        if rationale:
            lines.append(f"\n💡 {rationale}")
    else:
        lines.append("ℹ️ No policy proposal generated.")
        rationale = result.get("rationale")
        if rationale:
            lines.append(f"\n💡 {rationale}")

    reason_code = result.get("reason_code")
    if reason_code:
        lines.append(f"\n🏷️ Reason code: `{reason_code}`")

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("\n⚠️ **Warnings:**")
        for w in warnings:
            lines.append(f"  - {w}")

    conflict = result.get("conflict")
    if isinstance(conflict, dict) and conflict.get("has_conflict"):
        lines.append("\n⛔ **Conflict detected** — review before activating.")

    draft = result.get("draft")
    if isinstance(draft, dict):
        lines.append(f"\n✅ **DRAFT created** — version {draft.get('version','?')} (not yet active, requires human activation).")

    return "\n".join(lines)




def _details_dict(context: dict[str, Any]) -> dict[str, Any]:
    details = context.get("details")
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    if isinstance(details, dict) and isinstance(details.get("data"), dict):
        details = details["data"]
    return details if isinstance(details, dict) else {}


def _om_service_name(context: dict[str, Any]) -> str | None:
    details = _details_dict(context)
    service = details.get("service")
    if hasattr(service, "model_dump"):
        service = service.model_dump()
    if isinstance(service, dict):
        for key in ("name", "fullyQualifiedName"):
            value = service.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(service, str) and service.strip():
        return service.strip()
    return None


def _append_policy_warning(state: dict[str, Any], message: str) -> dict[str, Any]:
    current = dict(state.get("policy_result") or {})
    warnings = list(current.get("warnings") or [])
    if message not in warnings:
        warnings.append(message)
    current["warnings"] = warnings
    return current


def build_policy_nodes(
    *,
    gov_gateway: GovernanceGateway,
    policy_classifier: PolicyClassifier,
) -> dict[str, Any]:
    """Returns the POLICY-domain node functions, ready to register on a
    `StateGraph(AgentState)`."""

    def load_backend_policy_context(state: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = {}
        warnings: list[str] = []

        try:
            context["ranger_health"] = gov_gateway.inspect_ranger_state(kind="health")
        except Exception as exc:
            warnings.append(f"Ranger diagnostic unavailable: {exc}")

        service_name = _om_service_name(state.get("catalog_context", {}))
        if service_name:
            try:
                context["service_mapping"] = gov_gateway.resolve_resource_mapping(
                    om_service_name=service_name,
                    environment=state.get("environment", "local"),
                )
            except Exception as exc:
                warnings.append(
                    f"Exact service mapping unresolved for {service_name!r}: {exc}"
                )
        else:
            warnings.append("OpenMetadata service identity unavailable for exact mapping")

        policy_key = state.get("policy_key")
        if policy_key:
            try:
                context["existing_policy"] = gov_gateway.get_policy(policy_key)
                context["versions"] = gov_gateway.list_policy_versions(policy_key)
                context["sync_status"] = gov_gateway.get_ranger_sync_status(
                    policy_key=policy_key
                )
            except BackendMCPError as exc:
                if exc.code == "NOT_FOUND":
                    context["existing_policy"] = None
                    context["versions"] = []
                else:
                    raise

        context["warnings"] = warnings
        return {"governance_context": context}

    def policy_reasoning(state: dict[str, Any]) -> dict[str, Any]:
        raw_subjects = state.get("target_subjects")
        subjects = [Subject.model_validate(s) for s in raw_subjects] if raw_subjects else None
        result = policy_classifier.reason_policy(
            catalog_context=state.get("catalog_context", {}),
            governance_context=state.get("governance_context", {}),
            target_subjects=subjects,
            policy_intent=state.get("policy_intent"),
        )
        result.backend_context = dict(state.get("governance_context", {}))
        return {"policy_result": result.model_dump(mode="json")}

    def normalize_backend_policy(state: dict[str, Any]) -> dict[str, Any]:
        result = dict(state.get("policy_result") or {})
        proposal_raw = result.get("proposal")
        if not proposal_raw:
            return {"policy_result": result, "backend_logical_policy": None}

        proposal = PolicyReasoningResult.model_validate(result).proposal
        assert proposal is not None
        raw_subjects = state.get("target_subjects")
        explicit = [Subject.model_validate(s) for s in raw_subjects] if raw_subjects else None
        try:
            document = to_backend_logical_policy(
                proposal,
                explicit_subjects=explicit,
                require_explicit_subjects=bool(state.get("persist_draft", False)),
            )
        except PolicyAdapterError as exc:
            result = _append_policy_warning(state, str(exc))
            result["backend_logical_policy"] = None
            return {"policy_result": result, "backend_logical_policy": None}

        result["backend_logical_policy"] = document
        return {"policy_result": result, "backend_logical_policy": document}

    def check_conflict(state: dict[str, Any]) -> dict[str, Any]:
        result = dict(state.get("policy_result") or {})
        document = state.get("backend_logical_policy")
        policy_key = state.get("policy_key")
        if not document or not policy_key:
            if document and not policy_key:
                result = _append_policy_warning(
                    state,
                    "policy_key is required for Backend conflict/preview and DRAFT persistence",
                )
            return {"policy_result": result}
        conflict = gov_gateway.check_policy_conflict(
            policy_key=policy_key,
            logical_policy=document,
        )
        result["conflict"] = conflict
        return {"policy_result": result}

    def preview_change(state: dict[str, Any]) -> dict[str, Any]:
        result = dict(state.get("policy_result") or {})
        document = state.get("backend_logical_policy")
        policy_key = state.get("policy_key")
        if not document or not policy_key:
            return {"policy_result": result}
        preview = gov_gateway.preview_policy_change(
            policy_key=policy_key,
            logical_policy=document,
        )
        result["preview"] = preview
        return {"policy_result": result}

    def _policy_audit_ref(
        state: dict[str, Any],
        *,
        reason_code: str,
        validated_at: str | None,
    ) -> dict[str, Any]:
        return AIWriteAuditRef(
            model_fingerprint=model_fingerprint(
                model_name=policy_classifier.model_name,
                prompt_version=policy_classifier.prompt_version,
            ),
            prompt_version=policy_classifier.prompt_version,
            input_fingerprint=input_fingerprint(
                {
                    "policy_key": state.get("policy_key"),
                    "entity_fqn": state.get("entity_fqn"),
                    "target_subjects": state.get("target_subjects"),
                }
            ),
            reason_code=reason_code,
            validated_at=validated_at,
        ).model_dump(mode="json")

    def optional_create_draft(state: dict[str, Any]) -> dict[str, Any]:
        result = dict(state.get("policy_result") or {})
        if not state.get("persist_draft", False):
            return {"policy_result": result}

        # I11 master kill switch. When OFF, the Agent must not persist any
        # Backend write for this run -- fail safe, same as the OM-write path.
        if not state.get("agent_write_to_om_enabled", True):
            result = _append_policy_warning(state, "DRAFT_SKIPPED: kill switch disabled")
            result["reason_code"] = "KILL_SWITCH_DISABLED"
            result["audit_ref"] = _policy_audit_ref(
                state, reason_code="KILL_SWITCH_DISABLED", validated_at=None
            )
            return {"policy_result": result}

        document = state.get("backend_logical_policy")
        policy_key = state.get("policy_key")
        if not policy_key:
            result = _append_policy_warning(
                state, "persist_draft=true requires explicit policy_key"
            )
            result["reason_code"] = "DRAFT_SKIPPED_NO_KEY"
            return {"policy_result": result}
        if not state.get("target_subjects"):
            result = _append_policy_warning(
                state, "persist_draft=true requires explicit target_subjects"
            )
            result["reason_code"] = "DRAFT_SKIPPED_NO_SUBJECTS"
            return {"policy_result": result}
        if not document:
            result = _append_policy_warning(
                state, "DRAFT was not persisted because policy normalization failed"
            )
            result["reason_code"] = "DRAFT_SKIPPED_NORMALIZATION"
            return {"policy_result": result}

        mapping = (state.get("governance_context") or {}).get("service_mapping")
        if not isinstance(mapping, dict):
            result = _append_policy_warning(
                state,
                "persist_draft=true requires an exact resolved OpenMetadata service mapping",
            )
            result["reason_code"] = "DRAFT_SKIPPED_MAPPING"
            return {"policy_result": result}
        mapped_catalog = str(mapping.get("trino_catalog") or "").strip()
        document_catalog = str(
            (document.get("resource") or {}).get("catalog") or ""
        ).strip()
        if not mapped_catalog:
            result = _append_policy_warning(
                state,
                "resolved service mapping is missing trino_catalog",
            )
            result["reason_code"] = "DRAFT_SKIPPED_MAPPING"
            return {"policy_result": result}
        if document_catalog != mapped_catalog:
            result = _append_policy_warning(
                state,
                (
                    "policy resource catalog does not match exact service mapping: "
                    f"{document_catalog!r} != {mapped_catalog!r}"
                ),
            )
            result["reason_code"] = "DRAFT_SKIPPED_MAPPING"
            return {"policy_result": result}

        if result.get("preview") is None or result.get("conflict") is None:
            result = _append_policy_warning(
                state, "DRAFT was not persisted because normalization/preview/conflict failed"
            )
            result["reason_code"] = "CONFLICT_CHECK_FAILED"
            return {"policy_result": result}

        # Validate-before-write (I8) is complete at this exact point -- every
        # prior check (kill switch, key/subjects presence, normalization,
        # mapping, conflict/preview) has passed. Timestamp it now, distinct
        # from any later created_at on the persisted DRAFT row (Hard
        # Invariant #13).
        validated_at = utc_now_iso()

        # HITL (per user's explicit choice): pause here, before ANY Backend
        # write happens -- even the DRAFT write, which is itself already
        # authority-safe (create_policy_version never changes authority;
        # only a separate, human-triggered activate_policy_version call
        # does). Requires a checkpointer to resume correctly; if none is
        # configured this call raises (LangGraph's own behavior), which is
        # the correct fail-closed outcome -- a graph that can't durably
        # pause should not silently skip the pause and write anyway.
        approval = interrupt(
            {
                "kind": "POLICY_DRAFT_APPROVAL",
                "policy_key": policy_key,
                "logical_policy": document,
                "conflict": result.get("conflict"),
                "preview": result.get("preview"),
                "policy_intent": state.get("policy_intent"),
                "message": (
                    f"Agent proposes creating a DRAFT policy version for "
                    f"{policy_key!r}. Approve to persist the DRAFT (still "
                    "not active -- a separate human activation step is "
                    "required afterward), or reject to discard."
                ),
            }
        )
        if not isinstance(approval, dict) or approval.get("decision") != "APPROVE":
            result = _append_policy_warning(
                state,
                "DRAFT was not persisted: human reviewer rejected the proposal"
                if isinstance(approval, dict)
                else "DRAFT was not persisted: invalid approval response",
            )
            result["reason_code"] = "DRAFT_SKIPPED_REJECTED"
            return {"policy_result": result}

        draft = gov_gateway.create_policy_version(
            policy_key=policy_key,
            logical_policy=document,
            reason=state.get("policy_intent"),
        )
        if draft.get("status") != "DRAFT":
            raise RuntimeError("Backend create_policy_version did not return DRAFT")
        if draft.get("authority_changed") is not False:
            raise RuntimeError("Backend DRAFT unexpectedly changed authority")
        if draft.get("dispatched") is not False:
            raise RuntimeError("Backend DRAFT unexpectedly dispatched reconciliation")
        result["draft"] = draft
        result["reason_code"] = "DRAFT_CREATED"
        result["audit_ref"] = _policy_audit_ref(
            state, reason_code="DRAFT_CREATED", validated_at=validated_at
        )
        return {"policy_result": result}

    return {
        "load_backend_policy_context": load_backend_policy_context,
        "policy_reasoning": policy_reasoning,
        "normalize_backend_policy": normalize_backend_policy,
        "check_policy_conflict": check_conflict,
        "preview_policy_change": preview_change,
        "optional_create_draft": optional_create_draft,
    }


__all__ = ["build_policy_nodes"]
