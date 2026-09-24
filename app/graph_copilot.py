"""COPILOT graph boundary (TASK-09 addendum): RAG review copilot for a
human reviewer, wired into the shared `StateGraph(AgentState)`. Read-only,
advisory only -- see `app/review_copilot.py`'s module docstring for the
full boundary rationale (never calls activate_policy_version/apply_tag/
rollback_policy or any other authority-changing operation).
"""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata_context import OpenMetadataGateway
from app.review_copilot import CopilotChatModel, ReviewCopilotService, gather_review_context


def build_copilot_nodes(
    *,
    om_gateway: OpenMetadataGateway,
    gov_gateway: GovernanceGateway,
    chat_model: CopilotChatModel,
) -> dict[str, Any]:
    """Returns the COPILOT-domain node function, ready to register on a
    `StateGraph(AgentState)`."""

    service = ReviewCopilotService(chat_model=chat_model)

    def run_review_copilot(state: dict[str, Any]) -> dict[str, Any]:
        context = gather_review_context(
            om_gateway=om_gateway,
            gov_gateway=gov_gateway,
            entity_type=state.get("entity_type", "table"),
            entity_fqn=state["entity_fqn"],
            policy_key=state.get("policy_key"),
            include_lineage=bool(state.get("include_lineage", True)),
        )
        result = service.ask(
            context=context,
            question=state.get("copilot_question"),
            want_recommendation=bool(state.get("copilot_want_recommendation", False)),
        )

        # HITL (per user's explicit choice): the copilot never writes
        # anything itself (see module docstring), so this is a plain
        # acknowledgment interrupt, not a write gate -- it surfaces the
        # recommendation in the UI with Approve/Reject buttons purely so
        # the human's decision is recorded in the same place they read the
        # advice, but nothing downstream in THIS graph branches on the
        # answer. Only fires when there actually is a recommendation to ack.
        if result.get("recommendation") is not None:
            ack = interrupt(
                {
                    "kind": "COPILOT_RECOMMENDATION_ACK",
                    "recommendation": result["recommendation"],
                    "answer": result.get("answer"),
                    "message": (
                        "Copilot suggestion (advisory only -- this does not "
                        "trigger any write). Acknowledge to continue."
                    ),
                }
            )
            result["human_acknowledgment"] = (
                ack.get("decision") if isinstance(ack, dict) else None
            )

        return {"copilot_result": result}

    return {"run_review_copilot": run_review_copilot}


__all__ = ["build_copilot_nodes"]
