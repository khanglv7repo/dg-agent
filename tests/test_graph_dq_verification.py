from __future__ import annotations

from unittest.mock import MagicMock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import AgentState, build_governance_graph
from app.services.dq_writer import DQWriterService


def _graph(*, dq_writer=None, checkpointer=None):
    om = MagicMock()
    gov = MagicMock()
    tag_classifier = MagicMock()
    policy_classifier = MagicMock()
    policy_classifier.model_name = "test-model"
    policy_classifier.prompt_version = "v2"
    return build_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=tag_classifier,
        policy_classifier=policy_classifier,
        dq_writer=dq_writer,
        checkpointer=checkpointer,
    ), om, gov


def test_dq_route_without_writer_returns_unavailable() -> None:
    graph, om, gov = _graph(dq_writer=None)
    result: AgentState = graph.invoke(
        {
            "request_type": "DQ",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "dq_rule_id": "rule-1",
            "dq_worker_id": "worker-1",
            "dq_test_definition_fqn": "columnValuesToBeNotNull",
        }
    )
    assert result["dq_result"]["status"] == "UNAVAILABLE"
    om.get_entity_context.assert_not_called()
    gov.get_workflow_status.assert_not_called()


def test_dq_route_pauses_for_hitl_approval_before_backend_call() -> None:
    """HITL: graph_dq.py's interrupt() must fire BEFORE create_dq_test_case
    -- verified here by asserting the Backend call has NOT happened yet at
    the pause point."""
    backend = MagicMock()
    dq_writer = DQWriterService(backend=backend, model_name="m", prompt_version="v1")
    graph, om, gov = _graph(dq_writer=dq_writer, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "dq-test-1"}}

    result: AgentState = graph.invoke(
        {
            "request_type": "DQ",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "dq_rule_id": "rule-1",
            "dq_worker_id": "worker-1",
            "dq_test_definition_fqn": "columnValuesToBeNotNull",
        },
        config=config,
    )
    assert "dq_result" not in result
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "DQ_TESTCASE_APPROVAL"
    assert payload["target_asset_fqn"] == "financial.crm.customers"
    backend.create_dq_test_case.assert_not_called()


def test_dq_route_with_writer_calls_backend_and_skips_om_context() -> None:
    backend = MagicMock()
    backend.create_dq_test_case.return_value = {
        "id": "tc-1",
        "natural_key_hash": "dg_abc",
        "om_testcase_id": "om-1",
        "status": "STAGED",
    }
    dq_writer = DQWriterService(backend=backend, model_name="m", prompt_version="v1")
    graph, om, gov = _graph(dq_writer=dq_writer, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "dq-test-2"}}

    # First call pauses at the HITL interrupt (see the pause-only test
    # above); resume with an APPROVE decision to reach the real write.
    graph.invoke(
        {
            "request_type": "DQ",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "dq_rule_id": "rule-1",
            "dq_worker_id": "worker-1",
            "dq_test_definition_fqn": "columnValuesToBeNotNull",
        },
        config=config,
    )
    result: AgentState = graph.invoke(
        Command(resume={"decision": "APPROVE"}), config=config
    )
    assert result["dq_result"]["status"] == "STAGED"
    assert result["dq_result"]["natural_key_hash"] == "dg_abc"
    # DQ never touches OM context or Backend MCP workflow status -- the spec
    # is caller-supplied, not derived from catalog/entity reasoning.
    om.get_entity_context.assert_not_called()
    gov.get_workflow_status.assert_not_called()
    backend.create_dq_test_case.assert_called_once()


def test_dq_route_rejected_by_human_skips_backend_call() -> None:
    backend = MagicMock()
    dq_writer = DQWriterService(backend=backend, model_name="m", prompt_version="v1")
    graph, _, _ = _graph(dq_writer=dq_writer, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "dq-test-3"}}

    graph.invoke(
        {
            "request_type": "DQ",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "dq_rule_id": "rule-1",
            "dq_worker_id": "worker-1",
            "dq_test_definition_fqn": "columnValuesToBeNotNull",
        },
        config=config,
    )
    result: AgentState = graph.invoke(
        Command(resume={"decision": "REJECT"}), config=config
    )
    assert result["dq_result"]["status"] == "REJECTED"
    backend.create_dq_test_case.assert_not_called()


def test_dq_route_kill_switch_skips_backend_call() -> None:
    backend = MagicMock()
    dq_writer = DQWriterService(backend=backend, model_name="m", prompt_version="v1")
    graph, _, _ = _graph(dq_writer=dq_writer)

    result: AgentState = graph.invoke(
        {
            "request_type": "DQ",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "dq_rule_id": "rule-1",
            "dq_worker_id": "worker-1",
            "dq_test_definition_fqn": "columnValuesToBeNotNull",
            "agent_write_to_om_enabled": False,
        }
    )
    assert result["dq_result"]["status"] == "SKIPPED"
    assert result["dq_result"]["reason_code"] == "KILL_SWITCH_DISABLED"
    backend.create_dq_test_case.assert_not_called()


def test_verification_route_gathers_readonly_evidence_only() -> None:
    graph, om, gov = _graph()
    gov.get_audit_summary.return_value = {"entries": []}
    gov.inspect_ranger_state.return_value = {"status": "healthy"}
    gov.get_ranger_sync_status.return_value = {"projections": []}

    result: AgentState = graph.invoke(
        {
            "request_type": "VERIFICATION",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "policy_key": "policy-1",
            "verification_audit_limit": 10,
        }
    )
    evidence = result["verification_result"]
    assert evidence["audit_summary"] == {"entries": []}
    assert evidence["ranger_health"] == {"status": "healthy"}
    assert evidence["ranger_sync_status"] == {"projections": []}
    assert evidence["warnings"] == []
    # Verification never mutates anything -- no create/activate/rollback call.
    gov.create_policy_version.assert_not_called()
    gov.activate_policy_version.assert_not_called()
    om.apply_tag_authoritative.assert_not_called()


def test_verification_route_only_queries_trino_when_caller_supplies_sql() -> None:
    graph, _, gov = _graph()
    gov.get_audit_summary.return_value = {}
    gov.inspect_ranger_state.return_value = {}

    result: AgentState = graph.invoke(
        {
            "request_type": "VERIFICATION",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
        }
    )
    gov.query_trino_readonly.assert_not_called()
    assert "trino_check" not in result["verification_result"]

    gov.query_trino_readonly.return_value = {"rows": []}
    result2: AgentState = graph.invoke(
        {
            "request_type": "VERIFICATION",
            "entity_type": "table",
            "entity_fqn": "financial.crm.customers",
            "verification_trino_check_sql": "SELECT current_user",
        }
    )
    gov.query_trino_readonly.assert_called_once_with(sql="SELECT current_user")
    assert result2["verification_result"]["trino_check"] == {"rows": []}
