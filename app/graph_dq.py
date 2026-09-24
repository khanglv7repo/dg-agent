"""DQ TestCase write graph boundary (TASK-09 Work Packet D:
"Route DQ through the selected B2/B3 writer; create dq_writer.py only when
authorized" -- authorized per `docs/13_IMPLEMENTATION_SPEC.md`'s "DQ Writer
(Agent) CONDITIONAL ADD" section, B2 PASS + B3 PASS).

Unlike TAG/POLICY, DQ has no LLM reasoning step in this graph -- the caller
(a rule engine, a human-reviewed proposal, or a future classification-derived
DQ suggestion) supplies the already-decided TestCase spec directly via
`AgentState`'s dq_* fields. This graph's only job is the SPEC_DRAFT ->
VALIDATED -> STAGED write boundary itself (`DQWriterService`), not judgment
about which DQ check to propose.
"""
from __future__ import annotations

from typing import Any

from app.services.dq_writer import DQSpecDraft, DQWriterService


def build_dq_nodes(*, dq_writer: DQWriterService) -> dict[str, Any]:
    """Returns the DQ-domain node functions, ready to register on a
    `StateGraph(AgentState)`."""

    def write_dq_test_case(state: dict[str, Any]) -> dict[str, Any]:
        draft = DQSpecDraft(
            target_asset_fqn=state.get("entity_fqn"),
            test_definition_fqn=state.get("dq_test_definition_fqn"),
            rule_id=state.get("dq_rule_id"),
            worker_id=state.get("dq_worker_id"),
            parameter_values=state.get("dq_parameter_values") or {},
            test_key=state.get("dq_test_key"),
            column_name=state.get("dq_column_name"),
            rationale=state.get("dq_rationale"),
        )
        result = dq_writer.write(
            draft,
            agent_write_to_om_enabled=state.get("agent_write_to_om_enabled", True),
        )
        return {"dq_result": result}

    return {"write_dq_test_case": write_dq_test_case}


__all__ = ["build_dq_nodes"]
