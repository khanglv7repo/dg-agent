"""LangGraph API server entry point.

`langgraph dev`/`langgraph up` needs a zero-argument factory that returns a
compiled graph -- `build_governance_graph()` itself takes required gateway/
classifier arguments (constructed per-request in `runner.py` /
`app/tasks/classification.py` today), so this module is the one place that
builds real, process-lifetime instances of those dependencies the same way
`GovernanceAgentRunner.__init__` does, then hands LangGraph a single
pre-wired compiled graph.

This does NOT change how the Celery worker or `GovernanceAgentRunner` call
the graph -- both still call `build_governance_graph`/`run_governance_graph`
directly, in-process, exactly as before. This module only adds a second,
optional way to reach the same graph: over HTTP via `langgraph dev`/`up`,
for Studio UI debugging and any future external caller. See
`agent/langgraph.json` for the server config.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.checkpointer import build_checkpointer, checkpointer_enabled
from app.clients.backend_rest import BackendRestClient
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata_context import OpenMetadataGateway
from app.graph import build_governance_graph
from app.llm_runtime import LLMRuntimeConfig
from app.services.dq_writer import DQWriterService

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def graph():
    """Zero-argument factory required by `langgraph.json`'s `graphs` map.

    Constructs one set of process-lifetime gateways/classifiers -- mirrors
    `GovernanceAgentRunner.__init__` -- then returns the fully composed,
    compiled `StateGraph`. Called once per server process by the LangGraph
    runtime, not once per request.
    """
    mcp_url = os.getenv("OPENMETADATA_MCP_URL", "http://localhost:8585/mcp")
    backend_mcp_url = os.getenv("BACKEND_MCP_URL", "http://127.0.0.1:8001/mcp")
    backend_api_url = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/api/v1")
    agent_bot_token = os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", "")

    om_gateway = OpenMetadataGateway(endpoint=mcp_url, token=agent_bot_token)
    gov_gateway = GovernanceGateway(endpoint=backend_mcp_url)

    llm_config = LLMRuntimeConfig.from_env()
    tag_classifier = llm_config.tag_classifier()
    policy_classifier = llm_config.policy_classifier()

    backend_rest = BackendRestClient(base_url=backend_api_url)
    dq_writer = DQWriterService(
        backend=backend_rest,
        model_name="dq-writer-deterministic",
        prompt_version="v1",
    )

    copilot_chat_model = llm_config.copilot_chat_model()
    chat_router = llm_config.chat_router()

    checkpointer = build_checkpointer() if checkpointer_enabled() else None

    return build_governance_graph(
        om_gateway=om_gateway,
        gov_gateway=gov_gateway,
        tag_classifier=tag_classifier,
        policy_classifier=policy_classifier,
        dq_writer=dq_writer,
        copilot_chat_model=copilot_chat_model,
        chat_router=chat_router,
        checkpointer=checkpointer,
    )


__all__ = ["graph"]
