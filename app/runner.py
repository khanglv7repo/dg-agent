"""High-level Agent runner over bounded OpenMetadata and Backend gateways."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata_context import OpenMetadataGateway
from app.graph import run_governance_graph
from app.llm_runtime import LLMRuntimeConfig
from app.schemas import (
    AgentDecision,
    AgentRunRequest,
    AgentRunResponse,
    AgentTagSuggestion,
)


class GovernanceAgentRunner:
    def __init__(self) -> None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        self.mcp_url = os.getenv("OPENMETADATA_MCP_URL", "http://localhost:8585/mcp")
        self.backend_mcp_url = os.getenv(
            "BACKEND_MCP_URL", "http://127.0.0.1:8001/mcp"
        )
        self.agent_bot_token = os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", "")
        self.environment = os.getenv("GOVERNANCE_ENVIRONMENT", "local")

        # One canonical LLM config is shared with the Celery worker path.
        self.llm_config = LLMRuntimeConfig.from_env()

        # Compatibility attributes retained for existing diagnostics/tests.
        self.llm_api_key = self.llm_config.api_key
        self.llm_model = self.llm_config.model
        self.llm_base_url = self.llm_config.base_url

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        om_gateway = OpenMetadataGateway(
            endpoint=self.mcp_url,
            token=self.agent_bot_token,
        )
        gov_gateway = GovernanceGateway(endpoint=self.backend_mcp_url)
        tag_classifier = self.llm_config.tag_classifier()
        policy_classifier = self.llm_config.policy_classifier()
        try:
            tag_result, policy_result, _context = run_governance_graph(
                om_gateway=om_gateway,
                gov_gateway=gov_gateway,
                tag_classifier=tag_classifier,
                policy_classifier=policy_classifier,
                request_type=request.request_type,
                entity_type=request.entity_type,
                entity_fqn=request.entity_fqn,
                allowed_tags=request.allowed_tags,
                include_lineage=request.include_lineage,
                target_subjects=request.target_subjects,
                policy_intent=request.policy_intent,
                policy_key=request.policy_key,
                persist_draft=request.persist_draft,
                environment=request.environment or self.environment,
            )
        finally:
            om_gateway.close()
            gov_gateway.close()

        decision = AgentDecision()
        if tag_result and tag_result.recommendations:
            decision = AgentDecision(
                suggestions=[
                    AgentTagSuggestion(
                        tag=rec.tag,
                        confidence=rec.confidence,
                        rationale=rec.rationale,
                        field_path=rec.field_path,
                    )
                    for rec in tag_result.recommendations
                ],
                summary=tag_result.summary,
            )

        return AgentRunResponse(
            status="completed",
            request_type=request.request_type,
            decision=decision,
            tag_result=tag_result,
            policy_result=policy_result,
            openmetadata_suggestion_ids=[],
        )
