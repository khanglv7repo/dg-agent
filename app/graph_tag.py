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

from app.classifier import StructuredClassifier
from app.gateways.openmetadata_context import OpenMetadataGateway

logger = logging.getLogger(__name__)


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
        return {"tag_result": raw.model_dump(mode="json")}

    return {
        "load_om_context": load_om_context,
        "tag_reasoning": tag_reasoning,
    }


__all__ = ["compute_effective_allowed_tags", "build_tag_nodes"]
