"""LangGraph orchestration graph for AI Agent.

Per R6-A final correctness requirements:
- Explicit split between TAG reasoning and POLICY reasoning.
- Effective allowed tag set intersects actual OpenMetadata taxonomy with caller whitelist.
- Any LLM tag proposal not in the effective allowed set is strictly removed.
- Transport clients (OpenMetadataGateway, GovernanceGateway) decoupled from reasoning nodes.
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

from app.classifier import PolicyClassifier, StructuredClassifier
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway
from app.schemas import (
    PolicyReasoningResult,
    Subject,
    TagReasoningResult,
)


class AgentState(TypedDict, total=False):
    request_type: str  # "TAG" or "POLICY"
    entity_type: str
    entity_fqn: str
    include_lineage: bool
    allowed_tags: list[str]
    target_subjects: list[dict[str, Any]] | None
    policy_intent: str | None
    catalog_context: dict[str, Any]
    governance_context: dict[str, Any]
    tag_result: dict[str, Any]
    policy_result: dict[str, Any]


def compute_effective_allowed_tags(
    actual_om_tags: list[str],
    caller_allowed_tags: list[str],
) -> list[str]:
    """Compute effective allowed tags via intersection of actual OM taxonomy and caller whitelist.

    OpenMetadata is authoritative for tag existence. Caller allowed_tags can ONLY restrict
    OpenMetadata truth, never replace it.
    """
    if not actual_om_tags:
        return []
    if caller_allowed_tags:
        actual_set = set(actual_om_tags)
        return [tag for tag in caller_allowed_tags if tag in actual_set]
    return list(actual_om_tags)


def build_governance_graph(
    *,
    om_gateway: OpenMetadataGateway,
    gov_gateway: GovernanceGateway,
    tag_classifier: StructuredClassifier,
    policy_classifier: PolicyClassifier,
):
    """Build LangGraph workflow with separate TAG and POLICY reasoning branches."""

    def route_intent(state: AgentState) -> str:
        req_type = (state.get("request_type") or "TAG").upper()
        if req_type == "POLICY":
            return "POLICY"
        return "TAG"

    def load_om_context(state: AgentState) -> AgentState:
        context = om_gateway.get_entity_context(
            entity_type=state.get("entity_type", "table"),
            entity_fqn=state["entity_fqn"],
            include_lineage=bool(state.get("include_lineage", True)),
        )
        return {"catalog_context": context}

    def load_governance_context(state: AgentState) -> AgentState:
        try:
            gov_info = gov_gateway.inspect_ranger_state()
        except Exception as exc:
            gov_info = {"status": "unavailable", "error": str(exc)}
        return {"governance_context": gov_info}

    def tag_reasoning(state: AgentState) -> AgentState:
        try:
            actual_om_tags = om_gateway.get_taxonomies()
        except Exception as exc:
            logger.warning(f"Failed to fetch OM taxonomies in tag_reasoning: {exc}")
            actual_om_tags = []
        caller_allowed = state.get("allowed_tags", [])
        effective_allowed = compute_effective_allowed_tags(actual_om_tags, caller_allowed)

        raw_result = tag_classifier.classify(
            catalog_context=state.get("catalog_context", {}),
            allowed_tags=effective_allowed,
        )

        effective_set = set(effective_allowed)
        # Strict validation: filter out any suggested tags not in effective allowed set
        filtered_recs = [rec for rec in raw_result.recommendations if rec.tag in effective_set]
        raw_result.recommendations = filtered_recs
        return {"tag_result": raw_result.model_dump(mode="json")}

    def policy_reasoning(state: AgentState) -> AgentState:
        raw_subjects = state.get("target_subjects")
        subjects = [Subject.model_validate(s) for s in raw_subjects] if raw_subjects else None

        result = policy_classifier.reason_policy(
            catalog_context=state.get("catalog_context", {}),
            governance_context=state.get("governance_context"),
            target_subjects=subjects,
            policy_intent=state.get("policy_intent"),
        )
        return {"policy_result": result.model_dump(mode="json")}

    graph = StateGraph(AgentState)
    graph.add_node("load_om_context", load_om_context)
    graph.add_node("load_governance_context", load_governance_context)
    graph.add_node("tag_reasoning", tag_reasoning)
    graph.add_node("policy_reasoning", policy_reasoning)

    graph.add_conditional_edges(
        START,
        route_intent,
        {
            "TAG": "load_om_context",
            "POLICY": "load_om_context",
        },
    )

    def after_om_context(state: AgentState) -> str:
        req_type = (state.get("request_type") or "TAG").upper()
        if req_type == "POLICY":
            return "load_governance_context"
        return "tag_reasoning"

    graph.add_conditional_edges(
        "load_om_context",
        after_om_context,
        {
            "load_governance_context": "load_governance_context",
            "tag_reasoning": "tag_reasoning",
        },
    )

    graph.add_edge("load_governance_context", "policy_reasoning")
    graph.add_edge("tag_reasoning", END)
    graph.add_edge("policy_reasoning", END)

    return graph.compile()


def run_governance_graph(
    *,
    om_gateway: OpenMetadataGateway,
    gov_gateway: GovernanceGateway,
    tag_classifier: StructuredClassifier,
    policy_classifier: PolicyClassifier,
    request_type: str = "TAG",
    entity_type: str = "table",
    entity_fqn: str,
    allowed_tags: list[str] | None = None,
    include_lineage: bool = True,
    target_subjects: list[Subject] | None = None,
    policy_intent: str | None = None,
) -> tuple[TagReasoningResult | None, PolicyReasoningResult | None, dict[str, Any]]:
    graph = build_governance_graph(
        om_gateway=om_gateway,
        gov_gateway=gov_gateway,
        tag_classifier=tag_classifier,
        policy_classifier=policy_classifier,
    )

    subjects_dump = [s.model_dump(mode="json") for s in target_subjects] if target_subjects else None

    result = graph.invoke(
        {
            "request_type": request_type,
            "entity_type": entity_type,
            "entity_fqn": entity_fqn,
            "allowed_tags": allowed_tags or [],
            "include_lineage": include_lineage,
            "target_subjects": subjects_dump,
            "policy_intent": policy_intent,
        }
    )

    tag_result = (
        TagReasoningResult.model_validate(result["tag_result"])
        if result.get("tag_result")
        else None
    )
    policy_result = (
        PolicyReasoningResult.model_validate(result["policy_result"])
        if result.get("policy_result")
        else None
    )

    return tag_result, policy_result, dict(result.get("catalog_context", {}))
