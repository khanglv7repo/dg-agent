"""Main Agent runner executing governance graphs via OpenMetadata Gateway and Governance Gateway.

Per R6-A requirements:
- No longer assumes OpenMetadata Suggestion creation as primary orchestration.
- Interfacing with OpenMetadata is handled via OpenMetadataGateway.
- Interfacing with Backend MCP is handled via GovernanceGateway.
- Supports distinct TAG and POLICY reasoning workflows.
"""
from __future__ import annotations

import os
from typing import Any

from app.classifier import OpenAIPolicyClassifier, OpenAIStructuredClassifier
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway
from app.graph import run_governance_graph
from app.schemas import (
    AgentDecision,
    AgentRunRequest,
    AgentRunResponse,
    AgentTagSuggestion,
)


class GovernanceAgentRunner:
    """Main Agent runner executing TAG and POLICY governance flows."""

    def __init__(self) -> None:
        self.mcp_url = os.getenv("OPENMETADATA_MCP_URL", "http://localhost:8585/mcp")
        self.backend_mcp_url = os.getenv("BACKEND_MCP_URL", "http://localhost:8000/mcp")
        self.om_base_url = os.getenv("OPENMETADATA_BASE_URL", "http://localhost:8585")
        self.agent_bot_token = os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_base_url = os.getenv("LLM_BASE_URL") or None

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        om_gateway = OpenMetadataGateway(
            endpoint=self.mcp_url,
            token=self.agent_bot_token,
        )
        gov_gateway = GovernanceGateway(
            endpoint=self.backend_mcp_url,
        )
        tag_classifier = OpenAIStructuredClassifier(
            model=self.llm_model,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )
        policy_classifier = OpenAIPolicyClassifier(
            model=self.llm_model,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )

        try:
            tag_result, policy_result, context = run_governance_graph(
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
            )
        finally:
            om_gateway.close()
            gov_gateway.close()

        # Build backward-compatibility decision if tag_result exists
        decision = AgentDecision()
        if tag_result and tag_result.recommendations:
            legacy_suggestions = [
                AgentTagSuggestion(
                    tag=rec.tag,
                    confidence=rec.confidence,
                    rationale=rec.rationale,
                    field_path=rec.field_path,
                )
                for rec in tag_result.recommendations
            ]
            decision = AgentDecision(
                suggestions=legacy_suggestions,
                summary=tag_result.summary,
            )

        return AgentRunResponse(
            status="completed",
            request_type=request.request_type,
            decision=decision,
            tag_result=tag_result,
            policy_result=policy_result,
            openmetadata_suggestion_ids=[],  # Suggestion creation is NOT required in primary flow
        )
