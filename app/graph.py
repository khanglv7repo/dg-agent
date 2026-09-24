"""Copilot routing graph: composes the Policy, Tag(Classification), DQ, and
Verification domain graphs (TASK-09 Work Packet D) into one
`StateGraph(AgentState)`.

Each domain's nodes live in its own module (`graph_tag.py`, `graph_policy.py`,
`graph_dq.py`, `graph_verification.py`) -- this file is now only the shared
`AgentState` contract, the request_type router, and the wiring that composes
those nodes into edges. This mirrors the manifest's own instruction: "Preserve
reusable nodes; do not rewrite the whole graph when a node can be extracted."
`build_governance_graph`/`run_governance_graph`/`compute_effective_allowed_tags`
remain the stable external contract every existing caller (`runner.py`,
`tests/test_agent_graph.py`, `tests/test_tag_reasoning.py`,
`tests/test_policy_reasoning.py`) already imports from `app.graph`.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.classifier import PolicyClassifier, StructuredClassifier
from app.gateways.governance import GovernanceGateway
# The runner constructs the hardened OpenMetadata context gateway for all graph reads.
from app.gateways.openmetadata_context import OpenMetadataGateway
from app.graph_dq import build_dq_nodes
from app.graph_policy import build_policy_nodes
from app.graph_tag import build_tag_nodes, compute_effective_allowed_tags
from app.graph_verification import build_verification_nodes
from app.schemas import PolicyReasoningResult, Subject, TagReasoningResult
from app.services.dq_writer import DQWriterService


class AgentState(TypedDict, total=False):
    request_type: str
    entity_type: str
    entity_fqn: str
    include_lineage: bool
    allowed_tags: list[str]
    target_subjects: list[dict[str, Any]] | None
    policy_intent: str | None
    policy_key: str | None
    persist_draft: bool
    environment: str
    # I11 master kill switch: when False, skip DRAFT persistence entirely
    # (mirrors the Agent->OM master switch's fail-safe semantics for the
    # Agent->Backend write path).
    agent_write_to_om_enabled: bool
    catalog_context: dict[str, Any]
    governance_context: dict[str, Any]
    tag_result: dict[str, Any]
    policy_result: dict[str, Any]
    backend_logical_policy: dict[str, Any] | None
    # DQ domain (graph_dq.py) -- caller-supplied spec, no LLM reasoning here.
    dq_test_definition_fqn: str | None
    dq_rule_id: str | None
    dq_worker_id: str | None
    dq_parameter_values: dict[str, Any] | None
    dq_test_key: str | None
    dq_column_name: str | None
    dq_rationale: str | None
    dq_result: dict[str, Any]
    # Verification domain (graph_verification.py) -- read-only evidence only.
    verification_audit_limit: int
    verification_trino_check_sql: str | None
    verification_result: dict[str, Any]


def build_governance_graph(
    *,
    om_gateway: OpenMetadataGateway,
    gov_gateway: GovernanceGateway,
    tag_classifier: StructuredClassifier,
    policy_classifier: PolicyClassifier,
    dq_writer: DQWriterService | None = None,
    checkpointer=None,
):
    """Compose all 4 domain graphs behind one Copilot router. `dq_writer` is
    optional (None) because most callers only exercise TAG/POLICY today --
    the DQ branch simply isn't reachable via `route_intent` unless a caller
    passes `request_type="DQ"`, and building a real `DQWriterService`
    requires a live `BackendRestClient` the TAG/POLICY-only callers
    (`runner.py`'s synchronous path, most existing tests) don't construct."""

    dq_available = dq_writer is not None

    def route_from_start(state: AgentState) -> str:
        request_type = (state.get("request_type") or "TAG").upper()
        if request_type == "POLICY":
            return "POLICY"
        if request_type == "VERIFICATION":
            return "VERIFICATION"
        if request_type == "DQ":
            return "DQ" if dq_available else "DQ_UNAVAILABLE"
        return "TAG"

    def route_om_context(state: AgentState) -> str:
        return "POLICY" if (state.get("request_type") or "TAG").upper() == "POLICY" else "TAG"

    def dq_unavailable(state: AgentState) -> AgentState:
        return {
            "dq_result": {
                "status": "UNAVAILABLE",
                "reason_code": "NO_MATCH",
                "error": (
                    "DQ writer was not configured for this graph instance "
                    "(no BackendRestClient/DQWriterService was constructed)"
                ),
            }
        }

    tag_nodes = build_tag_nodes(om_gateway=om_gateway, tag_classifier=tag_classifier)
    policy_nodes = build_policy_nodes(gov_gateway=gov_gateway, policy_classifier=policy_classifier)
    verification_nodes = build_verification_nodes(gov_gateway=gov_gateway)

    graph = StateGraph(AgentState)

    # TAG domain
    graph.add_node("load_om_context", tag_nodes["load_om_context"])
    graph.add_node("tag_reasoning", tag_nodes["tag_reasoning"])

    # POLICY domain (reuses TAG's load_om_context for shared catalog context)
    graph.add_node("load_backend_policy_context", policy_nodes["load_backend_policy_context"])
    graph.add_node("policy_reasoning", policy_nodes["policy_reasoning"])
    graph.add_node("normalize_backend_policy", policy_nodes["normalize_backend_policy"])
    graph.add_node("check_policy_conflict", policy_nodes["check_policy_conflict"])
    graph.add_node("preview_policy_change", policy_nodes["preview_policy_change"])
    graph.add_node("optional_create_draft", policy_nodes["optional_create_draft"])

    # VERIFICATION domain (read-only, no catalog context needed)
    graph.add_node("gather_verification_evidence", verification_nodes["gather_verification_evidence"])

    # DQ domain (no catalog context needed -- caller supplies the spec directly)
    graph.add_node("dq_unavailable", dq_unavailable)
    if dq_available:
        dq_nodes = build_dq_nodes(dq_writer=dq_writer)
        graph.add_node("write_dq_test_case", dq_nodes["write_dq_test_case"])
        graph.add_edge("write_dq_test_case", END)

    branches = {
        "TAG": "load_om_context",
        "POLICY": "load_om_context",
        "VERIFICATION": "gather_verification_evidence",
        "DQ_UNAVAILABLE": "dq_unavailable",
    }
    if dq_available:
        branches["DQ"] = "write_dq_test_case"
    graph.add_conditional_edges(START, route_from_start, branches)

    graph.add_conditional_edges(
        "load_om_context",
        route_om_context,
        {"TAG": "tag_reasoning", "POLICY": "load_backend_policy_context"},
    )
    graph.add_edge("tag_reasoning", END)
    graph.add_edge("load_backend_policy_context", "policy_reasoning")
    graph.add_edge("policy_reasoning", "normalize_backend_policy")
    graph.add_edge("normalize_backend_policy", "check_policy_conflict")
    graph.add_edge("check_policy_conflict", "preview_policy_change")
    graph.add_edge("preview_policy_change", "optional_create_draft")
    graph.add_edge("optional_create_draft", END)
    graph.add_edge("gather_verification_evidence", END)
    graph.add_edge("dq_unavailable", END)

    return graph.compile(checkpointer=checkpointer)


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
    policy_key: str | None = None,
    persist_draft: bool = False,
    environment: str = "local",
    agent_write_to_om_enabled: bool = True,
    checkpointer=None,
    thread_id: str | None = None,
) -> tuple[TagReasoningResult | None, PolicyReasoningResult | None, dict[str, Any]]:
    graph = build_governance_graph(
        om_gateway=om_gateway,
        gov_gateway=gov_gateway,
        tag_classifier=tag_classifier,
        policy_classifier=policy_classifier,
        checkpointer=checkpointer,
    )
    # LangGraph requires a thread_id whenever a checkpointer is attached (it's
    # the resume/dedup key -- restarting with the same thread_id continues
    # from the last completed node instead of re-running from START). Default
    # to entity_fqn when the caller doesn't supply one explicitly; callers
    # that need duplicate-delivery dedup on a specific event should pass
    # their own stable id (e.g. event_id) instead.
    invoke_config = (
        {"configurable": {"thread_id": thread_id or entity_fqn}}
        if checkpointer is not None
        else None
    )
    result = graph.invoke(
        {
            "request_type": request_type,
            "entity_type": entity_type,
            "entity_fqn": entity_fqn,
            "allowed_tags": allowed_tags or [],
            "include_lineage": include_lineage,
            "target_subjects": (
                [s.model_dump(mode="json") for s in target_subjects]
                if target_subjects
                else None
            ),
            "policy_intent": policy_intent,
            "policy_key": policy_key,
            "persist_draft": persist_draft,
            "environment": environment,
            "agent_write_to_om_enabled": agent_write_to_om_enabled,
        },
        config=invoke_config,
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


__all__ = [
    "AgentState",
    "build_governance_graph",
    "run_governance_graph",
    "compute_effective_allowed_tags",
]
