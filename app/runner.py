"""High-level Agent runner over bounded OpenMetadata and Backend gateways."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.checkpointer import build_checkpointer, checkpointer_enabled
from app.clients.backend_rest import BackendRestClient
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata_context import OpenMetadataGateway
from app.graph import build_governance_graph, run_governance_graph
from app.llm_runtime import LLMRuntimeConfig
from app.schemas import (
    AgentDecision,
    AgentRunRequest,
    AgentRunResponse,
    AgentTagSuggestion,
    AIWriteAuditRef,
    CopilotRecommendation,
    PolicyReasoningResult,
    TagReasoningResult,
)
from app.services.dq_writer import DQSpecDraft, DQWriterService

_PENDING_APPROVAL_STATUS = "PENDING_APPROVAL"


def _pending_interrupt_payload(result: dict) -> dict | None:
    """Detect a HITL `interrupt()` pause in a raw `graph.invoke()` result
    (per `app/graph_policy.py`/`graph_dq.py`/`graph_copilot.py`'s
    interrupt() calls). Returns the interrupt's `.value` payload if one
    fired, else None. Verified live shape:
    `{"__interrupt__": [Interrupt(value=..., id=...)]}`.

    This synchronous runner surfaces a paused graph as
    `status="PENDING_APPROVAL"` with this payload attached -- callers using
    this path directly (not through the LangGraph API server, which
    handles interrupts natively) are responsible for resuming via a
    second `graph.invoke(Command(resume=...), config=...)` call using the
    same thread_id. Agent Chat UI / the LangGraph server path does not go
    through this runner at all and handles interrupts itself."""
    pending = result.get("__interrupt__")
    if not pending:
        return None
    first = pending[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"value": value}


class GovernanceAgentRunner:
    def __init__(self) -> None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        self.mcp_url = os.getenv("OPENMETADATA_MCP_URL", "http://localhost:8585/mcp")
        self.backend_mcp_url = os.getenv(
            "BACKEND_MCP_URL", "http://127.0.0.1:8001/mcp"
        )
        # TASK-09: Backend's plain REST API (DQ TestCase creation), a
        # separate process/port from BACKEND_MCP_URL -- see
        # app/clients/backend_rest.py's module docstring.
        self.backend_api_url = os.getenv(
            "BACKEND_API_URL", "http://127.0.0.1:8000/api/v1"
        )
        self.agent_bot_token = os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", "")
        self.environment = os.getenv("GOVERNANCE_ENVIRONMENT", "local")

        # One canonical LLM config is shared with the Celery worker path.
        self.llm_config = LLMRuntimeConfig.from_env()

        # Compatibility attributes retained for existing diagnostics/tests.
        self.llm_api_key = self.llm_config.api_key
        self.llm_model = self.llm_config.model
        self.llm_base_url = self.llm_config.base_url

        # TASK-10 unblocked this (agent_checkpoint_db now provisioned) --
        # TASK-09's "Checkpoint persistence" manifest row. One checkpointer
        # per process (PostgresSaver manages its own connection pool), not
        # one per request. AGENT_CHECKPOINT_ENABLED=false (or no DB password
        # configured) disables it -- graph.compile(checkpointer=None) is
        # langgraph's own no-persistence default, so this fails safe to the
        # exact pre-existing behavior when disabled, never a hard error.
        self.checkpointer = build_checkpointer() if checkpointer_enabled() else None

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        if request.request_type == "DQ":
            return self._run_dq(request)
        if request.request_type == "VERIFICATION":
            return self._run_verification(request)
        if request.request_type == "COPILOT":
            return self._run_copilot(request)
        return self._run_tag_or_policy(request)

    def _run_tag_or_policy(self, request: AgentRunRequest) -> AgentRunResponse:
        om_gateway = OpenMetadataGateway(
            endpoint=self.mcp_url,
            token=self.agent_bot_token,
        )
        gov_gateway = GovernanceGateway(endpoint=self.backend_mcp_url)
        tag_classifier = self.llm_config.tag_classifier()
        policy_classifier = self.llm_config.policy_classifier()
        try:
            graph = build_governance_graph(
                om_gateway=om_gateway,
                gov_gateway=gov_gateway,
                tag_classifier=tag_classifier,
                policy_classifier=policy_classifier,
                checkpointer=self.checkpointer,
            )
            thread_id = request.correlation_id or request.event_id
            invoke_config = (
                {"configurable": {"thread_id": thread_id}}
                if self.checkpointer is not None
                else None
            )
            raw_result = graph.invoke(
                {
                    "request_type": request.request_type,
                    "entity_type": request.entity_type,
                    "entity_fqn": request.entity_fqn,
                    "allowed_tags": request.allowed_tags,
                    "include_lineage": request.include_lineage,
                    "target_subjects": (
                        [s.model_dump(mode="json") for s in request.target_subjects]
                        if request.target_subjects
                        else None
                    ),
                    "policy_intent": request.policy_intent,
                    "policy_key": request.policy_key,
                    "persist_draft": request.persist_draft,
                    "environment": request.environment or self.environment,
                    "agent_write_to_om_enabled": request.agent_write_to_om_enabled,
                },
                config=invoke_config,
            )
        finally:
            om_gateway.close()
            gov_gateway.close()

        # HITL: the POLICY DRAFT write path (graph_policy.py) pauses via
        # interrupt() before ever calling create_policy_version. A paused
        # graph's result has neither tag_result nor policy_result yet (the
        # node hasn't returned) -- surface this distinctly rather than
        # silently returning empty results, per the same rationale as
        # _run_dq/_run_copilot below.
        pending = _pending_interrupt_payload(raw_result)
        if pending is not None:
            return AgentRunResponse(
                status=_PENDING_APPROVAL_STATUS,
                request_type=request.request_type,
                pending_approval={**pending, "thread_id": thread_id},
            )

        tag_result = (
            TagReasoningResult.model_validate(raw_result["tag_result"])
            if raw_result.get("tag_result")
            else None
        )
        policy_result = (
            PolicyReasoningResult.model_validate(raw_result["policy_result"])
            if raw_result.get("policy_result")
            else None
        )

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

        # TASK-09: surface the write-boundary reason code + audit ref onto the
        # top-level response. Only the POLICY path (optional_create_draft)
        # populates these today; the TAG path is the standalone ai.classification
        # Celery worker (app/tasks/classification.py), not this synchronous runner.
        reason_code = policy_result.reason_code if policy_result else None
        audit_ref = (
            AIWriteAuditRef.model_validate(policy_result.audit_ref)
            if policy_result and policy_result.audit_ref
            else None
        )

        return AgentRunResponse(
            status="completed",
            request_type=request.request_type,
            decision=decision,
            tag_result=tag_result,
            policy_result=policy_result,
            openmetadata_suggestion_ids=[],
            reason_code=reason_code,
            audit_ref=audit_ref,
        )

    def _run_dq(self, request: AgentRunRequest) -> AgentRunResponse:
        """TASK-09: DQ has no LLM reasoning step -- the caller supplies the
        already-decided TestCase spec (dq_* request fields); this runner's
        job is only to wire the real BackendRestClient/DQWriterService and
        invoke the DQ branch of the shared graph."""
        # om_gateway/gov_gateway aren't used by the DQ branch itself, but
        # build_governance_graph requires them positionally -- construct the
        # same real gateways as the TAG/POLICY path for consistency (closing
        # them afterwards) rather than passing None and special-casing the
        # graph builder's signature.
        om_gateway = OpenMetadataGateway(endpoint=self.mcp_url, token=self.agent_bot_token)
        gov_gateway = GovernanceGateway(endpoint=self.backend_mcp_url)
        backend_rest = BackendRestClient(base_url=self.backend_api_url)
        dq_writer = DQWriterService(
            backend=backend_rest,
            model_name="dq-writer-deterministic",
            prompt_version="v1",
        )
        try:
            graph = build_governance_graph(
                om_gateway=om_gateway,
                gov_gateway=gov_gateway,
                tag_classifier=self.llm_config.tag_classifier(),
                policy_classifier=self.llm_config.policy_classifier(),
                dq_writer=dq_writer,
                checkpointer=self.checkpointer,
            )
            thread_id = request.correlation_id or request.event_id
            invoke_config = (
                {"configurable": {"thread_id": thread_id}}
                if self.checkpointer is not None
                else None
            )
            result = graph.invoke(
                {
                    "request_type": "DQ",
                    "entity_type": request.entity_type,
                    "entity_fqn": request.entity_fqn,
                    "dq_test_definition_fqn": request.dq_test_definition_fqn,
                    "dq_rule_id": request.dq_rule_id,
                    "dq_worker_id": request.dq_worker_id,
                    "dq_parameter_values": request.dq_parameter_values,
                    "dq_test_key": request.dq_test_key,
                    "dq_column_name": request.dq_column_name,
                    "dq_rationale": request.dq_rationale,
                    "agent_write_to_om_enabled": request.agent_write_to_om_enabled,
                },
                config=invoke_config,
            )
        finally:
            om_gateway.close()
            gov_gateway.close()
            backend_rest.close()

        # HITL: graph_dq.py pauses via interrupt() before the real
        # create_dq_test_case Backend call.
        pending = _pending_interrupt_payload(result)
        if pending is not None:
            return AgentRunResponse(
                status=_PENDING_APPROVAL_STATUS,
                request_type="DQ",
                pending_approval={**pending, "thread_id": thread_id},
            )

        dq_result = result.get("dq_result") or {}
        audit_ref = (
            AIWriteAuditRef.model_validate(dq_result["audit_ref"])
            if dq_result.get("audit_ref")
            else None
        )
        return AgentRunResponse(
            status="completed",
            request_type="DQ",
            dq_result=dq_result,
            reason_code=dq_result.get("reason_code"),
            audit_ref=audit_ref,
        )

    def _run_verification(self, request: AgentRunRequest) -> AgentRunResponse:
        """TASK-09: read-only evidence gathering, no write boundary at all --
        no reason_code/audit_ref (those are for AI-originated writes, per I9;
        this path never writes anything)."""
        om_gateway = OpenMetadataGateway(endpoint=self.mcp_url, token=self.agent_bot_token)
        gov_gateway = GovernanceGateway(endpoint=self.backend_mcp_url)
        try:
            graph = build_governance_graph(
                om_gateway=om_gateway,
                gov_gateway=gov_gateway,
                tag_classifier=self.llm_config.tag_classifier(),
                policy_classifier=self.llm_config.policy_classifier(),
            )
            result = graph.invoke(
                {
                    "request_type": "VERIFICATION",
                    "entity_type": request.entity_type,
                    "entity_fqn": request.entity_fqn,
                    "policy_key": request.policy_key,
                    "verification_audit_limit": request.verification_audit_limit,
                    "verification_trino_check_sql": request.verification_trino_check_sql,
                }
            )
        finally:
            om_gateway.close()
            gov_gateway.close()

        return AgentRunResponse(
            status="completed",
            request_type="VERIFICATION",
            verification_result=result.get("verification_result") or {},
        )

    def _run_copilot(self, request: AgentRunRequest) -> AgentRunResponse:
        """TASK-09 addendum: RAG review copilot. Never writes anything
        itself (see app/review_copilot.py's module docstring) -- the only
        interrupt() here is an acknowledgment pause, not a write gate."""
        om_gateway = OpenMetadataGateway(endpoint=self.mcp_url, token=self.agent_bot_token)
        gov_gateway = GovernanceGateway(endpoint=self.backend_mcp_url)
        copilot_chat_model = self.llm_config.copilot_chat_model()
        try:
            graph = build_governance_graph(
                om_gateway=om_gateway,
                gov_gateway=gov_gateway,
                tag_classifier=self.llm_config.tag_classifier(),
                policy_classifier=self.llm_config.policy_classifier(),
                copilot_chat_model=copilot_chat_model,
                checkpointer=self.checkpointer,
            )
            thread_id = request.correlation_id or request.event_id
            invoke_config = (
                {"configurable": {"thread_id": thread_id}}
                if self.checkpointer is not None
                else None
            )
            result = graph.invoke(
                {
                    "request_type": "COPILOT",
                    "entity_type": request.entity_type,
                    "entity_fqn": request.entity_fqn,
                    "include_lineage": request.include_lineage,
                    "policy_key": request.policy_key,
                    "copilot_question": request.copilot_question,
                    "copilot_want_recommendation": request.copilot_want_recommendation,
                },
                config=invoke_config,
            )
        finally:
            om_gateway.close()
            gov_gateway.close()

        pending = _pending_interrupt_payload(result)
        if pending is not None:
            return AgentRunResponse(
                status=_PENDING_APPROVAL_STATUS,
                request_type="COPILOT",
                pending_approval={**pending, "thread_id": thread_id},
            )

        copilot_result = result.get("copilot_result") or {}
        recommendation = (
            CopilotRecommendation.model_validate(copilot_result["recommendation"])
            if copilot_result.get("recommendation")
            else None
        )
        return AgentRunResponse(
            status="completed",
            request_type="COPILOT",
            copilot_answer=copilot_result.get("answer"),
            copilot_recommendation=recommendation,
        )
