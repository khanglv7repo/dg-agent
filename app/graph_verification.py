"""Verification / incident-analysis graph boundary (TASK-09 Work Packet D:
"Route incident analysis to read-only OM/Ranger/Trino evidence paths.").

This is deliberately read-only end to end: it gathers evidence
(`get_audit_summary`, `inspect_ranger_state`, `get_ranger_sync_status`, an
optional caller-supplied bounded `query_trino_readonly` check) and returns
it, without any LLM reasoning step and without any write. The exact SQL for
`query_trino_readonly` is a caller-supplied, pre-built string (e.g. from a
fixed reconciliation-check template) -- never LLM-generated -- matching
`GovernanceGateway`'s existing "no generic LLM tool loop can choose an
arbitrary call" boundary rule (`tests/test_boundaries.py`).
"""
from __future__ import annotations

from typing import Any

from app.gateways.governance import GovernanceGateway


def build_verification_nodes(*, gov_gateway: GovernanceGateway) -> dict[str, Any]:
    """Returns the VERIFICATION-domain node functions, ready to register on
    a `StateGraph(AgentState)`."""

    def gather_verification_evidence(state: dict[str, Any]) -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        warnings: list[str] = []

        policy_key = state.get("policy_key")

        try:
            evidence["audit_summary"] = gov_gateway.get_audit_summary(
                policy_key=policy_key,
                object_id=state.get("entity_fqn"),
                limit=state.get("verification_audit_limit", 20),
            )
        except Exception as exc:
            warnings.append(f"Audit summary unavailable: {exc}")

        try:
            evidence["ranger_health"] = gov_gateway.inspect_ranger_state(kind="health")
        except Exception as exc:
            warnings.append(f"Ranger health unavailable: {exc}")

        if policy_key:
            try:
                evidence["ranger_sync_status"] = gov_gateway.get_ranger_sync_status(
                    policy_key=policy_key
                )
            except Exception as exc:
                warnings.append(f"Ranger sync status unavailable for {policy_key!r}: {exc}")

        # Caller-supplied, pre-built read-only SQL only -- never LLM-authored.
        trino_check_sql = state.get("verification_trino_check_sql")
        if trino_check_sql:
            try:
                evidence["trino_check"] = gov_gateway.query_trino_readonly(
                    sql=trino_check_sql
                )
            except Exception as exc:
                warnings.append(f"Trino read-only check failed: {exc}")

        evidence["warnings"] = warnings
        return {"verification_result": evidence}

    return {"gather_verification_evidence": gather_verification_evidence}


__all__ = ["build_verification_nodes"]
