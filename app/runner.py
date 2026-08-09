from __future__ import annotations

import os
from app.classifier import OpenAIStructuredClassifier
from app.clients.mcp import OpenMetadataMCPClient
from app.clients.openmetadata import OpenMetadataAgentClient
from app.graph import run_classification_graph
from app.schemas import AgentRunRequest, AgentRunResponse


class GovernanceAgentRunner:
    """Main Agent runner executing classification graphs and posting native OpenMetadata suggestions directly."""

    def __init__(self) -> None:
        self.mcp_url = os.getenv("OPENMETADATA_MCP_URL", "http://localhost:8585/mcp")
        self.om_base_url = os.getenv("OPENMETADATA_BASE_URL", "http://localhost:8585")
        self.agent_bot_token = os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_base_url = os.getenv("LLM_BASE_URL") or None

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        mcp = OpenMetadataMCPClient(
            endpoint=self.mcp_url,
            token=self.agent_bot_token,
        )
        classifier = OpenAIStructuredClassifier(
            model=self.llm_model,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )
        try:
            decision, context = run_classification_graph(
                mcp=mcp,
                classifier=classifier,
                entity_type=request.entity_type,
                entity_fqn=request.entity_fqn,
                allowed_tags=request.allowed_tags,
                include_lineage=request.include_lineage,
            )
        finally:
            mcp.close()

        om_client = OpenMetadataAgentClient(
            base_url=self.om_base_url,
            token=self.agent_bot_token,
        )
        suggestion_ids: list[str] = []
        try:
            for item in decision.suggestions:
                target_fqn = item.field_path if item.field_path else request.entity_fqn
                res = om_client.create_suggestion(
                    entity_type=request.entity_type if not item.field_path else "column",
                    entity_fqn=target_fqn,
                    tag_fqn=item.tag,
                    rationale=item.rationale,
                )
                if "id" in res:
                    suggestion_ids.append(str(res["id"]))
        finally:
            om_client.close()

        return AgentRunResponse(
            status="completed",
            decision=decision,
            openmetadata_suggestion_ids=suggestion_ids,
        )
