from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.classifier import StructuredClassifier
from app.clients.mcp import OpenMetadataMCPClient
from app.schemas import AgentDecision


class AgentState(TypedDict, total=False):
    entity_type: str
    entity_fqn: str
    include_lineage: bool
    allowed_tags: list[str]
    catalog_context: dict[str, Any]
    decision: dict[str, Any]


def build_classification_graph(
    *,
    mcp: OpenMetadataMCPClient,
    classifier: StructuredClassifier,
):
    """Build a read-only LangGraph classification flow.

    Retrieves entity context using OpenMetadata MCP client and returns a structured decision.
    """

    def load_context(state: AgentState) -> AgentState:
        context = mcp.entity_context(
            entity_type=state["entity_type"],
            entity_fqn=state["entity_fqn"],
            include_lineage=bool(state.get("include_lineage", True)),
        )
        return {"catalog_context": context}

    def classify(state: AgentState) -> AgentState:
        decision = classifier.classify(
            catalog_context=state["catalog_context"],
            allowed_tags=state["allowed_tags"],
        )
        allowed = set(state["allowed_tags"])
        decision.suggestions = [item for item in decision.suggestions if item.tag in allowed]
        return {"decision": decision.model_dump(mode="json")}

    graph = StateGraph(AgentState)
    graph.add_node("load_context", load_context)
    graph.add_node("classify", classify)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify")
    graph.add_edge("classify", END)
    return graph.compile()


def run_classification_graph(
    *,
    mcp: OpenMetadataMCPClient,
    classifier: StructuredClassifier,
    entity_type: str,
    entity_fqn: str,
    allowed_tags: list[str],
    include_lineage: bool,
) -> tuple[AgentDecision, dict[str, Any]]:
    graph = build_classification_graph(mcp=mcp, classifier=classifier)
    result = graph.invoke(
        {
            "entity_type": entity_type,
            "entity_fqn": entity_fqn,
            "allowed_tags": allowed_tags,
            "include_lineage": include_lineage,
        }
    )
    return (
        AgentDecision.model_validate(result.get("decision", {})),
        dict(result.get("catalog_context", {})),
    )
