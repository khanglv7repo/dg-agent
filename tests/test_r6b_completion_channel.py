from __future__ import annotations

from unittest.mock import MagicMock

from fastmcp import FastMCP

from app.clients.backend_mcp import EXPECTED_BACKEND_TOOLS, R6B_BACKEND_TOOLS, BackendMCPClient
from app.gateways.governance import GovernanceGateway
from app.schemas import TagReasoningResult
from app.services.classification_completion import BackendClassificationCompletionChannel
from app.services.classification_worker import ClassificationWorkerService


def _r6b_server() -> FastMCP:
    mcp = FastMCP("R6-B Contract Test")

    @mcp.tool
    def get_policy(policy_key: str, version: int | None = None) -> dict: return {}
    @mcp.tool
    def list_policy_versions(policy_key: str) -> list[dict]: return []
    @mcp.tool
    def preview_policy_change(policy_key: str, logical_policy: dict) -> dict: return {}
    @mcp.tool
    def check_policy_conflict(policy_key: str, logical_policy: dict) -> dict: return {"conflict": False}
    @mcp.tool
    def resolve_resource_mapping(om_service_name: str, environment: str) -> dict: return {}
    @mcp.tool
    def get_ranger_sync_status(policy_key: str, version: int | None = None) -> dict: return {}
    @mcp.tool
    def get_workflow_status(execution_id: str) -> dict: return {"id": execution_id, "status": "WAITING_AI"}
    @mcp.tool
    def get_audit_summary(limit: int = 20) -> dict: return {"limit": limit}
    @mcp.tool
    def inspect_ranger_state(kind: str, name: str | None = None, policy_key: str | None = None) -> dict: return {}
    @mcp.tool
    def query_trino_readonly(sql: str) -> dict: return {"sql": sql}
    @mcp.tool
    def create_policy_version(policy_key: str, logical_policy: dict, reason: str | None = None) -> dict: return {}
    @mcp.tool
    def activate_policy_version(policy_key: str, version: int, confirmed: bool = False, approval_reason: str | None = None) -> dict: return {}
    @mcp.tool
    def rollback_policy(policy_key: str, target_version: int, confirmed: bool = False, reason: str | None = None) -> dict: return {}
    @mcp.tool
    def update_service_mapping(om_service_name: str, trino_catalog: str, ranger_service_name: str, environment: str, confirmed: bool = False, ranger_tag_service_name: str | None = None, enabled: bool = True, reason: str | None = None) -> dict: return {}
    @mcp.tool
    def request_ranger_sync(policy_key: str) -> dict: return {"authority_changed": False}
    @mcp.tool
    def complete_classification_execution(execution_id: str, generation: int, status: str, result: dict) -> dict:
        return {"status": status, "execution_id": execution_id, "generation": generation, "authority_changed": True, "result": result}
    return mcp


def test_r6b_contract_is_frozen_r5_plus_one_completion_tool() -> None:
    assert R6B_BACKEND_TOOLS[:-1] == EXPECTED_BACKEND_TOOLS
    assert R6B_BACKEND_TOOLS[-1] == "complete_classification_execution"
    probe = BackendMCPClient(source=_r6b_server()).validate_r6b_contract()
    assert [item["name"] for item in probe["tools"]] == list(R6B_BACKEND_TOOLS)


def test_gateway_completion_wrapper_uses_exact_bounded_tool() -> None:
    mock = MagicMock(spec=BackendMCPClient)
    mock.call_tool.return_value = {"status": "NO_PROPOSAL"}
    gateway = GovernanceGateway(client=mock)
    payload = {"entity_type": "table", "entity_fqn": "svc.db.schema.table", "recommendations": [], "mutations": []}
    gateway.complete_classification_execution(execution_id="exec-1", generation=2, status="NO_PROPOSAL", result=payload)
    mock.call_tool.assert_called_once_with("complete_classification_execution", {"execution_id": "exec-1", "generation": 2, "status": "NO_PROPOSAL", "result": payload})


def test_completion_adapter_forwards_without_confirmation_flag() -> None:
    governance = MagicMock()
    governance.complete_classification_execution.return_value = {"status": "NO_PROPOSAL"}
    channel = BackendClassificationCompletionChannel(governance)
    payload = {"entity_type": "table", "entity_fqn": "svc.db.schema.table", "recommendations": [], "mutations": []}
    response = channel.complete(execution_id="exec-1", generation=2, status="NO_PROPOSAL", result=payload)
    assert response["status"] == "NO_PROPOSAL"
    governance.complete_classification_execution.assert_called_once_with(execution_id="exec-1", generation=2, status="NO_PROPOSAL", result=payload)


def test_worker_no_proposal_completes_without_om_mutation() -> None:
    wf = {"source": "classification_execution", "id": "exec-1", "entity_type": "table", "entity_fqn": "svc.db.schema.table", "generation": 2, "status": "WAITING_AI"}
    governance = MagicMock(); governance.get_workflow_status.side_effect = [wf, wf]
    openmetadata = MagicMock(); openmetadata.get_entity_context.return_value = {"details": {}}; openmetadata.get_taxonomies.return_value = ["PII.Email"]
    classifier = MagicMock(); classifier.classify.return_value = TagReasoningResult(recommendations=[], summary="none")
    completion = MagicMock(); completion.complete.return_value = {"status": "NO_PROPOSAL"}
    service = ClassificationWorkerService(governance=governance, openmetadata=openmetadata, classifier=classifier, completion=completion)
    result = service.handle(execution_id="exec-1", generation=2)
    assert result["status"] == "NO_PROPOSAL"
    assert result["om_mutation_count"] == 0
    openmetadata.apply_tag_authoritative.assert_not_called()
    completion.complete.assert_called_once_with(execution_id="exec-1", generation=2, status="NO_PROPOSAL", result={"entity_type": "table", "entity_fqn": "svc.db.schema.table", "recommendations": [], "mutations": []})
