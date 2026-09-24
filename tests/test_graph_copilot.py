from __future__ import annotations

from unittest.mock import MagicMock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import AgentState, build_governance_graph
from app.schemas import CopilotRecommendation


def _graph(*, chat_model=None, checkpointer=None):
    om = MagicMock()
    gov = MagicMock()
    tag_classifier = MagicMock()
    policy_classifier = MagicMock()
    policy_classifier.model_name = "test-model"
    policy_classifier.prompt_version = "v2"
    return (
        build_governance_graph(
            om_gateway=om,
            gov_gateway=gov,
            tag_classifier=tag_classifier,
            policy_classifier=policy_classifier,
            copilot_chat_model=chat_model,
            checkpointer=checkpointer,
        ),
        om,
        gov,
    )


def test_copilot_route_without_chat_model_returns_unavailable() -> None:
    graph, om, gov = _graph(chat_model=None)
    result: AgentState = graph.invoke(
        {
            "request_type": "COPILOT",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "copilot_question": "Is this policy risky?",
        }
    )
    assert result["copilot_result"]["error"]
    om.get_entity_context.assert_not_called()


def test_copilot_answers_question_using_gathered_evidence() -> None:
    om = MagicMock()
    gov = MagicMock()
    chat_model = MagicMock()
    chat_model.answer.return_value = "Based on the evidence, this looks low-risk."

    from app.graph import build_governance_graph

    graph = build_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=MagicMock(),
        policy_classifier=MagicMock(model_name="m", prompt_version="v2"),
        copilot_chat_model=chat_model,
    )
    om.get_entity_context.return_value = {"details": {}}
    gov.get_audit_summary.return_value = {"entries": []}

    result: AgentState = graph.invoke(
        {
            "request_type": "COPILOT",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "copilot_question": "Is this policy risky?",
            "copilot_want_recommendation": False,
        }
    )
    assert result["copilot_result"]["answer"] == "Based on the evidence, this looks low-risk."
    assert result["copilot_result"]["recommendation"] is None
    chat_model.answer.assert_called_once()
    chat_model.recommend.assert_not_called()
    # Copilot never mutates anything.
    gov.create_policy_version.assert_not_called()
    gov.activate_policy_version.assert_not_called()
    om.apply_tag_authoritative.assert_not_called()


def test_copilot_recommendation_pauses_for_human_acknowledgment() -> None:
    """HITL: when a recommendation is produced, the copilot pauses via
    interrupt() so the human can acknowledge it -- but this never gates a
    write (the copilot has none)."""
    om = MagicMock()
    gov = MagicMock()
    om.get_entity_context.return_value = {"details": {}}
    gov.get_audit_summary.return_value = {}
    chat_model = MagicMock()
    chat_model.recommend.return_value = CopilotRecommendation(
        suggestion="APPROVE",
        rationale="No conflicting audit history found.",
        confidence=0.8,
        key_risks=[],
    )

    graph = build_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=MagicMock(),
        policy_classifier=MagicMock(model_name="m", prompt_version="v2"),
        copilot_chat_model=chat_model,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "copilot-test-1"}}

    paused: AgentState = graph.invoke(
        {
            "request_type": "COPILOT",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "copilot_want_recommendation": True,
        },
        config=config,
    )
    assert "__interrupt__" in paused
    interrupt_payload = paused["__interrupt__"][0].value
    assert interrupt_payload["kind"] == "COPILOT_RECOMMENDATION_ACK"
    assert interrupt_payload["recommendation"]["suggestion"] == "APPROVE"

    result = graph.invoke(Command(resume={"decision": "ACK"}), config=config)
    assert result["copilot_result"]["human_acknowledgment"] == "ACK"
    assert result["copilot_result"]["recommendation"]["suggestion"] == "APPROVE"


def test_copilot_without_recommendation_never_pauses() -> None:
    om = MagicMock()
    gov = MagicMock()
    om.get_entity_context.return_value = {"details": {}}
    gov.get_audit_summary.return_value = {}
    chat_model = MagicMock()
    chat_model.answer.return_value = "Just an answer, no recommendation."

    graph = build_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=MagicMock(),
        policy_classifier=MagicMock(model_name="m", prompt_version="v2"),
        copilot_chat_model=chat_model,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "copilot-test-2"}}
    result: AgentState = graph.invoke(
        {
            "request_type": "COPILOT",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "copilot_question": "What is this table for?",
            "copilot_want_recommendation": False,
        },
        config=config,
    )
    assert "__interrupt__" not in result
    assert result["copilot_result"]["answer"] == "Just an answer, no recommendation."
