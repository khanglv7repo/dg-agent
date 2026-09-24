"""TAG reasoning graph boundary, extracted from the former monolithic
`graph.py` (TASK-09 Work Packet D: "split or compose Policy, Classification,
DQ, and Verification routing in graph.py as frozen. Preserve reusable nodes;
do not rewrite the whole graph when a node can be extracted.").

Behavior is unchanged from the pre-split `graph.py` -- this is a pure
extraction, verified by the full pre-existing test suite
(`tests/test_tag_reasoning.py`, `tests/test_agent_graph.py`) still passing
unmodified against it.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.classifier import StructuredClassifier
from app.gateways.openmetadata_context import OpenMetadataGateway

logger = logging.getLogger(__name__)


def _format_tag_chat_reply(raw: Any, entity_fqn: str) -> str:
    """Format a TagReasoningResult as a readable chat message."""
    recs = getattr(raw, "recommendations", []) or []
    summary = getattr(raw, "summary", None) or ""

    if not recs:
        return (
            f"**Tag classification for `{entity_fqn}`**\n\n"
            "No tag recommendations — the entity may already be classified "
            "or no matching tags exist in the taxonomy.\n\n"
            f"_{summary}_" if summary else
            f"**Tag classification for `{entity_fqn}`**\n\nNo tag recommendations."
        )

    lines = [f"**Tag classification for `{entity_fqn}`**\n"]
    apply = [r for r in recs if getattr(r, "action_recommendation", "") == "APPLY"]
    review = [r for r in recs if getattr(r, "action_recommendation", "") == "REVIEW"]

    if apply:
        lines.append("✅ **APPLY** (high-confidence, deterministic whitelist eligible):")
        for r in apply:
            fp = f" → `{r.field_path}`" if getattr(r, "field_path", None) else ""
            lines.append(f"- `{r.tag}`{fp} ({int(r.confidence * 100)}%) — {r.rationale}")

    if review:
        lines.append("\n🔍 **REVIEW** (needs human confirmation):")
        for r in review:
            fp = f" → `{r.field_path}`" if getattr(r, "field_path", None) else ""
            lines.append(f"- `{r.tag}`{fp} ({int(r.confidence * 100)}%) — {r.rationale}")

    if summary:
        lines.append(f"\n_{summary}_")

    return "\n".join(lines)



def compute_effective_allowed_tags(
    actual_om_tags: list[str],
    caller_allowed_tags: list[str],
) -> list[str]:
    if not actual_om_tags:
        return []
    if caller_allowed_tags:
        actual_set = set(actual_om_tags)
        return [tag for tag in caller_allowed_tags if tag in actual_set]
    return list(actual_om_tags)


def build_tag_nodes(
    *,
    om_gateway: OpenMetadataGateway,
    tag_classifier: StructuredClassifier,
) -> dict[str, Any]:
    """Returns the TAG-domain node functions, ready to register on a
    `StateGraph(AgentState)`. Kept as plain functions (not a class) to match
    the existing closure-over-gateway pattern used throughout this codebase."""

    def load_om_context(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "catalog_context": om_gateway.get_entity_context(
                entity_type=state.get("entity_type", "table"),
                entity_fqn=state["entity_fqn"],
                include_lineage=bool(state.get("include_lineage", True)),
            )
        }

    def tag_reasoning(state: dict[str, Any]) -> dict[str, Any]:
        try:
            actual_om_tags = om_gateway.get_taxonomies()
        except Exception as exc:
            logger.warning("Failed to fetch OM taxonomy: %s", exc)
            actual_om_tags = []
        effective = compute_effective_allowed_tags(
            actual_om_tags,
            state.get("allowed_tags", []),
        )
        raw = tag_classifier.classify(
            catalog_context=state.get("catalog_context", {}),
            allowed_tags=effective,
        )
        allowed = set(effective)
        raw.recommendations = [rec for rec in raw.recommendations if rec.tag in allowed]
        result: dict[str, Any] = {"tag_result": raw.model_dump(mode="json")}
        # Append a human-readable reply when this run came via the chat UI
        # (messages path). Structured-API callers never set messages, so this
        # branch is unreachable for them -- zero behavior change for runner.py,
        # Celery worker, or any existing test.
        if state.get("messages"):
            result["messages"] = [
                AIMessage(content=_format_tag_chat_reply(raw, state.get("entity_fqn", "")))
            ]
        return result


    return {
        "load_om_context": load_om_context,
        "tag_reasoning": tag_reasoning,
    }


__all__ = ["compute_effective_allowed_tags", "build_tag_nodes"]
