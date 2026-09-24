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

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.classifier import PolicyClassifier, StructuredClassifier
from app.gateways.governance import GovernanceGateway
# TASK-09: the runner always constructs the hardened openmetadata_context
# gateway; this type hint must say so (see the same fix in
# app/services/classification_worker.py for the full rationale).
from app.gateways.openmetadata_context import OpenMetadataGateway
from app.graph_chat import ChatRouter, build_chat_nodes
from app.graph_copilot import build_copilot_nodes
from app.graph_dq import build_dq_nodes
from app.graph_policy import build_policy_nodes
from app.graph_tag import build_tag_nodes, compute_effective_allowed_tags
from app.graph_verification import build_verification_nodes
from app.review_copilot import CopilotChatModel
from app.schemas import PolicyReasoningResult, Subject, TagReasoningResult
from app.services.dq_writer import DQWriterService


class AgentState(TypedDict, total=False):
    # Chat entry (graph_chat.py) -- only populated by callers using Agent
    # Chat UI's free-text messages path (e.g. `POST /threads/{id}/runs`
    # with `{"messages": [...]}`). Every existing structured caller
    # (runner.py, the Celery worker, every pre-existing test) never sets
    # this and is entirely unaffected -- see route_entry below.
    messages: Annotated[list, add_messages]
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
    # Copilot domain (graph_copilot.py) -- RAG review copilot, read-only,
    # advisory only. policy_key doubles as the "which pending item" key,
    # shared with the POLICY domain's own state field.
    copilot_question: str | None
    copilot_want_recommendation: bool
    copilot_result: dict[str, Any]


def build_governance_graph(
    *,
    om_gateway: OpenMetadataGateway,
    gov_gateway: GovernanceGateway,
    tag_classifier: StructuredClassifier,
    policy_classifier: PolicyClassifier,
    dq_writer: DQWriterService | None = None,
    copilot_chat_model: CopilotChatModel | None = None,
    chat_router: ChatRouter | None = None,
    checkpointer=None,
):
    """Compose all domain graphs behind one Copilot router. `dq_writer`/
    `copilot_chat_model`/`chat_router` are optional (None) because most
    callers only exercise TAG/POLICY today -- those branches simply aren't
    reachable via `route_from_start` unless a caller passes the matching
    `request_type`, and building the real dependencies (a live
    `BackendRestClient`, a real chat-capable LLM) requires setup the
    TAG/POLICY-only callers (`runner.py`'s synchronous path, most existing
    tests) don't do."""

    dq_available = dq_writer is not None
    copilot_available = copilot_chat_model is not None
    chat_available = chat_router is not None

    def route_entry(state: AgentState) -> str:
        # Only take the chat path when the caller actually used the
        # messages/chat UI input AND didn't already supply a structured
        # request_type -- a direct structured request (runner.py, Celery,
        # every pre-existing test) always has request_type set and skips
        # this entirely, unaffected by whether a chat_router is configured.
        request_type = (state.get("request_type") or "").upper()
        if (
            state.get("messages")
            and chat_available
            and (not request_type or request_type == "CHAT_REPLY_ONLY")
        ):
            return "CHAT"
        return route_from_start(state)

    def route_from_start(state: AgentState) -> str:
        request_type = (state.get("request_type") or "TAG").upper()
        if request_type == "POLICY":
            return "POLICY"
        if request_type == "VERIFICATION":
            return "VERIFICATION"
        if request_type == "DQ":
            return "DQ" if dq_available else "DQ_UNAVAILABLE"
        if request_type == "COPILOT":
            return "COPILOT" if copilot_available else "COPILOT_UNAVAILABLE"
        if request_type == "CHAT_REPLY_ONLY":
            return "CHAT_REPLY_ONLY"
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

    def copilot_unavailable(state: AgentState) -> AgentState:
        return {
            "copilot_result": {
                "answer": None,
                "recommendation": None,
                "error": (
                    "Review copilot was not configured for this graph "
                    "instance (no chat-capable LLM was constructed)"
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

    # COPILOT domain (RAG review copilot, read-only/advisory)
    graph.add_node("copilot_unavailable", copilot_unavailable)
    if copilot_available:
        copilot_nodes = build_copilot_nodes(
            om_gateway=om_gateway,
            gov_gateway=gov_gateway,
            chat_model=copilot_chat_model,
        )
        graph.add_node("run_review_copilot", copilot_nodes["run_review_copilot"])
        graph.add_edge("run_review_copilot", END)

    # Chat entry (graph_chat.py) -- only reachable when a caller used the
    # messages/chat path (see route_entry above). Fills in request_type/
    # entity_fqn from the classified intent, or sets
    # request_type="CHAT_REPLY_ONLY" (handled as its own terminal branch)
    # when the message didn't carry enough information. Registered after
    # every domain node above so its conditional edges can target them.
    graph.add_node("chat_reply_only", lambda state: {})
    graph.add_edge("chat_reply_only", END)
    if chat_available:
        chat_nodes = build_chat_nodes(chat_router=chat_router)
        graph.add_node("chat_entry", chat_nodes["chat_entry"])
        chat_exit_branches = {
            "TAG": "load_om_context",
            "POLICY": "load_om_context",
            "VERIFICATION": "gather_verification_evidence",
            "DQ_UNAVAILABLE": "dq_unavailable",
            "COPILOT_UNAVAILABLE": "copilot_unavailable",
            "CHAT_REPLY_ONLY": "chat_reply_only",
        }
        if dq_available:
            chat_exit_branches["DQ"] = "write_dq_test_case"
        if copilot_available:
            chat_exit_branches["COPILOT"] = "run_review_copilot"
        graph.add_conditional_edges("chat_entry", route_from_start, chat_exit_branches)

    branches = {
        "TAG": "load_om_context",
        "POLICY": "load_om_context",
        "VERIFICATION": "gather_verification_evidence",
        "DQ_UNAVAILABLE": "dq_unavailable",
        "COPILOT_UNAVAILABLE": "copilot_unavailable",
    }
    if dq_available:
        branches["DQ"] = "write_dq_test_case"
    if copilot_available:
        branches["COPILOT"] = "run_review_copilot"
    if chat_available:
        branches["CHAT"] = "chat_entry"
    branches["CHAT_REPLY_ONLY"] = "chat_reply_only"
    graph.add_conditional_edges(START, route_entry, branches)

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

    # POLICY chat reply: append AIMessage when the run came via chat UI.
    # Structured-API callers never set messages, so this node is a no-op for
    # them (returns {} immediately) -- zero behavior change.
    from langchain_core.messages import AIMessage as _AIMessage
    from app.graph_policy import _format_policy_chat_reply

    def policy_chat_reply(state: dict) -> dict:
        if not state.get("messages"):
            return {}
        policy_result = state.get("policy_result") or {}
        return {
            "messages": [
                _AIMessage(content=_format_policy_chat_reply(
                    policy_result, state.get("entity_fqn", "")
                ))
            ]
        }

    graph.add_node("policy_chat_reply", policy_chat_reply)
    graph.add_edge("optional_create_draft", "policy_chat_reply")
    graph.add_edge("policy_chat_reply", END)
    graph.add_edge("gather_verification_evidence", END)
    graph.add_edge("dq_unavailable", END)
    graph.add_edge("copilot_unavailable", END)


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
