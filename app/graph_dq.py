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

from langgraph.types import interrupt

from app.services.dq_writer import DQSpecDraft, DQValidationError, DQWriterService


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

        agent_write_to_om_enabled = state.get("agent_write_to_om_enabled", True)

        # HITL (per user's explicit choice): pause before the real OM/Backend
        # write, but only when there's actually something to write -- a
        # disabled kill switch or an invalid draft has nothing worth
        # interrupting for (DQWriterService.write() below will report the
        # correct SKIPPED/VALIDATION_FAILED status itself in those cases,
        # without ever reaching this interrupt).
        if agent_write_to_om_enabled:
            try:
                DQWriterService.validate(draft)
            except DQValidationError:
                pass
            else:
                approval = interrupt(
                    {
                        "kind": "DQ_TESTCASE_APPROVAL",
                        "target_asset_fqn": draft.target_asset_fqn,
                        "test_definition_fqn": draft.test_definition_fqn,
                        "rule_id": draft.rule_id,
                        "parameter_values": draft.parameter_values,
                        "rationale": draft.rationale,
                        "message": (
                            f"Agent proposes creating a DQ TestCase "
                            f"({draft.test_definition_fqn!r}) against "
                            f"{draft.target_asset_fqn!r}. Approve to create "
                            "it in OpenMetadata (STAGED, not yet "
                            "executable), or reject to discard."
                        ),
                    }
                )
                if not isinstance(approval, dict) or approval.get("decision") != "APPROVE":
                    return {
                        "dq_result": {
                            "status": "REJECTED",
                            "reason_code": "MANUAL_OVERRIDE_LOCKED",
                        }
                    }

        result = dq_writer.write(
            draft,
            agent_write_to_om_enabled=agent_write_to_om_enabled,
        )
        return {"dq_result": result}

    return {"write_dq_test_case": write_dq_test_case}


__all__ = ["build_dq_nodes"]
