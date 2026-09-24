from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import AgentState, build_governance_graph
from app.schemas import ChatIntent


def _graph(*, chat_router=None, copilot_chat_model=None):
    om = MagicMock()
    gov = MagicMock()
    tag_classifier = MagicMock()
    policy_classifier = MagicMock(model_name="m", prompt_version="v2")
    return (
        build_governance_graph(
            om_gateway=om,
            gov_gateway=gov,
            tag_classifier=tag_classifier,
            policy_classifier=policy_classifier,
            copilot_chat_model=copilot_chat_model,
            chat_router=chat_router,
        ),
        om,
        gov,
    )


def test_greeting_gets_a_conversational_reply_not_a_crash() -> None:
    """Regression for a live bug (2026-09-24): Agent Chat UI posts
    {"messages": [...]}, and the graph used to require entity_fqn
    unconditionally, so a plain "hi" crashed with KeyError('entity_fqn')
    instead of getting a reply."""
    chat_router = MagicMock()
    chat_router.classify.return_value = ChatIntent(
        needs_clarification=True,
        reply_message="Hi there! Which table would you like to look at?",
    )
    graph, om, gov = _graph(chat_router=chat_router)

    result: AgentState = graph.invoke({"messages": [HumanMessage(content="hi")]})

    assert "dq_result" not in result
    assert result["request_type"] == "CHAT_REPLY_ONLY"
    replies = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(replies) == 1
    assert replies[0].content == "Hi there! Which table would you like to look at?"
    om.get_entity_context.assert_not_called()
    gov.get_audit_summary.assert_not_called()


def test_reply_only_thread_accepts_followup_prompt_without_crashing() -> None:
    """Regression for the real Agent Chat UI thread state: after a reply-only
    run, the next message arrives with the previous request_type still present
    in thread state. That must route back through chat_entry, not crash at START
    with KeyError('CHAT_REPLY_ONLY')."""
    chat_router = MagicMock()
    graph, om, gov = _graph(chat_router=chat_router)
    tool_response = MagicMock()
    tool_response.__enter__.return_value.read.return_value = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {"name": "search_metadata"},
                    {"name": "get_entity_details"},
                ]
            },
        }
    ).encode()

    from app import graph_chat

    original_urlopen = graph_chat.urllib.request.urlopen
    urlopen_mock = MagicMock(return_value=tool_response)
    graph_chat.urllib.request.urlopen = urlopen_mock
    try:
        with patch.dict("os.environ", {"OPENMETADATA_AGENT_BOT_TOKEN": "test-token"}):
            result: AgentState = graph.invoke(
                {
                    "request_type": "CHAT_REPLY_ONLY",
                    "messages": [
                        AIMessage(content="Hi there! Which table would you like to look at?"),
                        HumanMessage(content="list mcp kết nối"),
                    ],
                }
            )
    finally:
        graph_chat.urllib.request.urlopen = original_urlopen

    assert result["request_type"] == "CHAT_REPLY_ONLY"
    chat_router.classify.assert_not_called()
    replies = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert "OpenMetadata MCP:" in replies[-1].content
    assert "Backend MCP:" in replies[-1].content
    assert "OpenMetadata MCP tools/list:" in replies[-1].content
    assert "- search_metadata" in replies[-1].content
    assert "- get_entity_details" in replies[-1].content
    om.get_entity_context.assert_not_called()
    gov.get_audit_summary.assert_not_called()


def test_metadata_search_keyword_calls_openmetadata_mcp_not_chat_router() -> None:
    chat_router = MagicMock()
    graph, om, gov = _graph(chat_router=chat_router)
    tool_response = MagicMock()
    tool_response.__enter__.return_value.headers.get.return_value = None
    tool_response.__enter__.return_value.read.return_value = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "totalFound": 2,
                                "hasMore": False,
                                "results": [
                                    {
                                        "entityType": "databaseSchema",
                                        "fullyQualifiedName": "financial_postgres.financial_db.crm",
                                        "displayName": "crm",
                                    },
                                    {
                                        "entityType": "table",
                                        "fullyQualifiedName": "financial_postgres.financial_db.crm.customers",
                                        "displayName": "customers",
                                    },
                                ],
                            }
                        ),
                    }
                ]
            },
        }
    ).encode()

    from app import graph_chat

    original_urlopen = graph_chat.urllib.request.urlopen
    urlopen_mock = MagicMock(return_value=tool_response)
    graph_chat.urllib.request.urlopen = urlopen_mock
    try:
        with patch.dict("os.environ", {"OPENMETADATA_AGENT_BOT_TOKEN": "test-token"}):
            result: AgentState = graph.invoke({"messages": [HumanMessage(content="khám phá xem có cái nào là crm không")]})
    finally:
        graph_chat.urllib.request.urlopen = original_urlopen

    assert result["request_type"] == "CHAT_REPLY_ONLY"
    chat_router.classify.assert_not_called()
    request = urlopen_mock.call_args.args[0]
    payload = json.loads(request.data.decode())
    assert payload["method"] == "tools/call"
    assert payload["params"] == {"name": "search_metadata", "arguments": {"query": "crm"}}
    replies = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert "search_metadata('crm')" in replies[-1].content
    assert "financial_postgres.financial_db.crm.customers" in replies[-1].content
    om.get_entity_context.assert_not_called()
    gov.get_audit_summary.assert_not_called()


def test_metadata_search_followup_choice_uses_previous_crm_context() -> None:
    chat_router = MagicMock()
    graph, _, _ = _graph(chat_router=chat_router)
    tool_response = MagicMock()
    tool_response.__enter__.return_value.headers.get.return_value = None
    tool_response.__enter__.return_value.read.return_value = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "totalFound": 1,
                                "hasMore": False,
                                "results": [
                                    {
                                        "entityType": "table",
                                        "fullyQualifiedName": "financial_postgres.financial_db.crm.customers",
                                    }
                                ],
                            }
                        ),
                    }
                ]
            },
        }
    ).encode()

    from app import graph_chat

    original_urlopen = graph_chat.urllib.request.urlopen
    graph_chat.urllib.request.urlopen = MagicMock(return_value=tool_response)
    try:
        with patch.dict("os.environ", {"OPENMETADATA_AGENT_BOT_TOKEN": "test-token"}):
            result: AgentState = graph.invoke(
                {
                    "request_type": "CHAT_REPLY_ONLY",
                    "messages": [
                        HumanMessage(content="khám phá xem có cái nào là crm không"),
                        AIMessage(content='Bạn muốn search metadata theo từ khóa "crm" không?'),
                        HumanMessage(content="1"),
                    ],
                }
            )
    finally:
        graph_chat.urllib.request.urlopen = original_urlopen

    assert result["request_type"] == "CHAT_REPLY_ONLY"
    chat_router.classify.assert_not_called()
    replies = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert "financial_postgres.financial_db.crm.customers" in replies[-1].content


def test_openmetadata_metadata_discovery_calls_search_all() -> None:
    chat_router = MagicMock()
    graph, _, _ = _graph(chat_router=chat_router)
    tool_response = MagicMock()
    tool_response.__enter__.return_value.headers.get.return_value = None
    tool_response.__enter__.return_value.read.return_value = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "totalFound": 956,
                                "hasMore": True,
                                "results": [
                                    {
                                        "entityType": "database",
                                        "fullyQualifiedName": "financial_postgres.financial_db",
                                    }
                                ],
                            }
                        ),
                    }
                ]
            },
        }
    ).encode()

    from app import graph_chat

    urlopen_mock = MagicMock(return_value=tool_response)
    original_urlopen = graph_chat.urllib.request.urlopen
    graph_chat.urllib.request.urlopen = urlopen_mock
    try:
        with patch.dict("os.environ", {"OPENMETADATA_AGENT_BOT_TOKEN": "test-token"}):
            result: AgentState = graph.invoke({"messages": [HumanMessage(content="sử dụng OM mcp khám phá xem metadata thế nào")]})
    finally:
        graph_chat.urllib.request.urlopen = original_urlopen

    chat_router.classify.assert_not_called()
    request = urlopen_mock.call_args.args[0]
    payload = json.loads(request.data.decode())
    assert payload["params"] == {"name": "search_metadata", "arguments": {"query": "*"}}
    replies = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert "search_metadata('* / all metadata')" in replies[-1].content
    assert "financial_postgres.financial_db" in replies[-1].content


def test_chat_router_hallucinated_fqn_is_blocked_before_tool_call() -> None:
    chat_router = MagicMock()
    chat_router.classify.return_value = ChatIntent(
        needs_clarification=False,
        request_type="TAG",
        entity_fqn="mysql.sales.public.orders",
        entity_type="table",
    )
    graph, om, gov = _graph(chat_router=chat_router)

    result: AgentState = graph.invoke({"messages": [HumanMessage(content="tag this table")]})

    assert result["request_type"] == "CHAT_REPLY_ONLY"
    replies = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert "will not invent an entity FQN" in replies[-1].content
    om.get_entity_context.assert_not_called()
    gov.get_audit_summary.assert_not_called()


def test_chat_message_with_enough_info_routes_to_copilot() -> None:
    """Chat entry always sets copilot_want_recommendation=True (see
    graph_chat.py), so this reaches the COPILOT domain's own HITL
    interrupt() -- confirms the chat path correctly wires into the
    existing recommendation flow rather than bypassing it."""
    chat_router = MagicMock()
    chat_router.classify.return_value = ChatIntent(
        needs_clarification=False,
        request_type="COPILOT",
        entity_fqn="financial.crm.customers",
        entity_type="table",
        copilot_question="What risks are there?",
    )
    copilot_chat_model = MagicMock()
    copilot_chat_model.answer.return_value = "Some grounded answer."
    from app.schemas import CopilotRecommendation

    copilot_chat_model.recommend.return_value = CopilotRecommendation(
        suggestion="NEEDS_MORE_INFO", rationale="Not enough evidence.", confidence=0.3
    )

    graph, om, gov = _graph(chat_router=chat_router, copilot_chat_model=copilot_chat_model)
    om.get_entity_context.return_value = {"details": {}}
    gov.get_audit_summary.return_value = {}

    result: AgentState = graph.invoke(
        {"messages": [HumanMessage(content="What risks are there for financial.crm.customers?")]}
    )
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["kind"] == "COPILOT_RECOMMENDATION_ACK"
    om.get_entity_context.assert_called_once()


def test_chat_message_with_enough_info_routes_to_tag() -> None:
    chat_router = MagicMock()
    chat_router.classify.return_value = ChatIntent(
        needs_clarification=False,
        request_type="TAG",
        entity_fqn="financial.crm.customers",
        entity_type="table",
    )
    tag_classifier = MagicMock()
    from app.schemas import TagReasoningResult

    tag_classifier.classify.return_value = TagReasoningResult(recommendations=[], summary="none")

    om = MagicMock()
    gov = MagicMock()
    om.get_entity_context.return_value = {"details": {}}
    om.get_taxonomies.return_value = []

    graph = build_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=tag_classifier,
        policy_classifier=MagicMock(model_name="m", prompt_version="v2"),
        chat_router=chat_router,
    )
    result: AgentState = graph.invoke(
        {"messages": [HumanMessage(content="What tags fit financial.crm.customers?")]}
    )
    assert result["tag_result"]["summary"] == "none"


def test_structured_request_skips_chat_entry_even_with_chat_router_configured() -> None:
    """A direct structured request_type must never be intercepted by the
    chat router, even when one is configured (existing callers -- runner.py,
    the Celery worker, every pre-existing test -- must be unaffected)."""
    chat_router = MagicMock()
    graph, om, gov = _graph(chat_router=chat_router)
    om.get_entity_context.return_value = {"details": {}}
    gov.get_audit_summary.return_value = {}
    gov.inspect_ranger_state.return_value = {}

    result: AgentState = graph.invoke(
        {
            "request_type": "VERIFICATION",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
        }
    )
    assert "verification_result" in result
    chat_router.classify.assert_not_called()


def test_chat_without_router_configured_falls_through_to_default_tag_route() -> None:
    """When no chat_router is configured at all, a messages-only input must
    not crash -- it falls back to route_from_start's TAG default, which
    will itself fail on missing entity_fqn (existing, documented behavior
    for structured callers) rather than silently pretending to understand
    chat it can't actually parse."""
    om = MagicMock()
    gov = MagicMock()
    graph, _, _ = _graph(chat_router=None)
    # No chat_router configured -> route_entry falls through to
    # route_from_start, which defaults to TAG and requires entity_fqn.
    try:
        graph.invoke({"messages": [HumanMessage(content="hi")]})
        raised = False
    except KeyError:
        raised = True
    assert raised
