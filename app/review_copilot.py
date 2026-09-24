"""RAG review copilot: answers a human reviewer's question about a pending
DRAFT policy / tag suggestion / DQ test case, and optionally gives an
Approve/Reject suggestion with rationale.

Per the Mermaid flow agreed with the user: the copilot NEVER calls
`activate_policy_version`/`apply_tag`/`rollback_policy` or any other
authority-changing Backend/OM operation -- it is retrieval + reasoning
only. The human always makes the actual Approve/Reject call through the
existing Backend routes (`app/api/routes/data_access_policies.py`'s
`activate_policy_version`, etc.), completely outside this module.

Retrieval sources (per user's explicit choice):
- Backend audit log + policy version history, via `GovernanceGateway`
  (`get_audit_summary`, `list_policy_versions`, `get_ranger_sync_status`,
  `get_policy`) -- all already-existing, read-only typed methods.
- OpenMetadata catalog context (entity/lineage/tags), via
  `OpenMetadataGateway.get_entity_context` -- same method the TAG/POLICY
  reasoning paths already use.

No new Backend/OM call surface is introduced; this module only composes
calls that already exist and are already proven safe (`GovernanceGateway`
has no generic tool-choice escape hatch, per `tests/test_boundaries.py`).
"""
from __future__ import annotations

import json
from typing import Any, Protocol

from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata_context import OpenMetadataGateway
from app.schemas import CopilotRecommendation


class CopilotChatModel(Protocol):
    """Bounded LLM interface: a plain-text answer call and a separate,
    schema-constrained recommendation call. Two different LLM invocations
    on purpose -- mixing free-text chat and structured output in one call
    is exactly the `PolicyLLMOutput` bug this codebase already hit once
    (see its docstring in schemas.py): letting the LLM free-write into a
    field that also carries a structured contract invites it to invent
    values outside that contract."""

    def answer(self, *, prompt: str) -> str: ...

    def recommend(self, *, prompt: str) -> CopilotRecommendation: ...


def gather_review_context(
    *,
    om_gateway: OpenMetadataGateway,
    gov_gateway: GovernanceGateway,
    entity_type: str,
    entity_fqn: str,
    policy_key: str | None,
    include_lineage: bool = True,
) -> dict[str, Any]:
    """Read-only evidence bundle, same shape family as
    `graph_verification.py`'s evidence gathering -- reused pattern, not
    duplicated logic (that module is read-only-evidence-for-incident-
    analysis; this is read-only-evidence-for-a-pending-review, a distinct
    trigger but the same underlying gateways/methods)."""
    context: dict[str, Any] = {"warnings": []}

    try:
        context["catalog_context"] = om_gateway.get_entity_context(
            entity_type=entity_type,
            entity_fqn=entity_fqn,
            include_lineage=include_lineage,
        )
    except Exception as exc:
        context["warnings"].append(f"Catalog context unavailable: {exc}")

    try:
        context["audit_summary"] = gov_gateway.get_audit_summary(
            object_id=entity_fqn, policy_key=policy_key, limit=20
        )
    except Exception as exc:
        context["warnings"].append(f"Audit summary unavailable: {exc}")

    if policy_key:
        try:
            context["existing_policy"] = gov_gateway.get_policy(policy_key)
            context["policy_versions"] = gov_gateway.list_policy_versions(policy_key)
            context["ranger_sync_status"] = gov_gateway.get_ranger_sync_status(
                policy_key=policy_key
            )
        except Exception as exc:
            context["warnings"].append(
                f"Policy history unavailable for {policy_key!r}: {exc}"
            )

    return context


def _build_answer_prompt(*, question: str, context: dict[str, Any]) -> str:
    return (
        "You are a Data Governance Review Copilot helping a human reviewer "
        "decide whether to approve or reject a pending policy/tag/DQ "
        "change. Answer the reviewer's question using ONLY the evidence "
        "below. If the evidence doesn't answer the question, say so "
        "plainly -- never invent audit history, policy versions, or "
        "catalog facts not present below. You are advisory only; you "
        "cannot approve, reject, or apply anything yourself.\n\n"
        f"Reviewer's question:\n{question}\n\n"
        "Evidence (Backend audit log, policy history, OpenMetadata catalog "
        "context):\n"
        f"{json.dumps(context, default=str)[:20000]}"
    )


def _build_recommendation_prompt(*, context: dict[str, Any]) -> str:
    return (
        "You are a Data Governance Review Copilot. Based ONLY on the "
        "evidence below, suggest APPROVE, REJECT, or NEEDS_MORE_INFO for "
        "the pending change, with a rationale grounded in that evidence "
        "and a list of concrete risks if any exist. This is a suggestion "
        "for a human reviewer, not a decision -- the human always makes "
        "the actual call.\n\n"
        "Evidence (Backend audit log, policy history, OpenMetadata catalog "
        "context):\n"
        f"{json.dumps(context, default=str)[:20000]}"
    )


class ReviewCopilotService:
    def __init__(self, *, chat_model: CopilotChatModel) -> None:
        self.chat_model = chat_model

    def ask(
        self,
        *,
        context: dict[str, Any],
        question: str | None,
        want_recommendation: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"answer": None, "recommendation": None}

        if question:
            prompt = _build_answer_prompt(question=question, context=context)
            result["answer"] = self.chat_model.answer(prompt=prompt)

        if want_recommendation:
            prompt = _build_recommendation_prompt(context=context)
            recommendation = self.chat_model.recommend(prompt=prompt)
            result["recommendation"] = recommendation.model_dump(mode="json")

        return result


__all__ = ["ReviewCopilotService", "CopilotChatModel", "gather_review_context"]
