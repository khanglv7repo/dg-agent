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

import json
from typing import Any

from langchain_core.messages import AIMessage

from app.gateways.governance import GovernanceGateway


def _format_verification_chat_reply(evidence: dict[str, Any], entity_fqn: str) -> str:
    """Format verification evidence as a readable chat message."""
    lines = [f"**Verification evidence for `{entity_fqn}`**\n"]

    audit = evidence.get("audit_summary", {})
    if isinstance(audit, dict):
        total = audit.get("total", 0)
        entries = audit.get("entries") or audit.get("items") or []
        lines.append(f"📋 **Audit summary:** {total} event(s) found")
        for e in (entries if isinstance(entries, list) else [])[:5]:
            if isinstance(e, dict):
                lines.append(f"  - {e.get('action','?')} on {e.get('object_type','?')} `{e.get('object_id','?')}` at {e.get('created_at','?')}")

    ranger = evidence.get("ranger_health", {})
    if isinstance(ranger, dict):
        status = ranger.get("status") or ranger.get("health") or "unknown"
        lines.append(f"\n🛡️ **Ranger health:** {status}")

    sync = evidence.get("ranger_sync_status")
    if isinstance(sync, dict):
        lines.append(f"🔄 **Ranger sync:** {sync.get('status','unknown')} (version {sync.get('version','?')})")

    trino = evidence.get("trino_check")
    if trino:
        lines.append(f"\n🔍 **Trino check:** {json.dumps(trino, default=str)[:300]}")

    warnings = evidence.get("warnings") or []
    if warnings:
        lines.append("\n⚠️ **Warnings:**")
        for w in warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)



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
        result: dict[str, Any] = {"verification_result": evidence}
        if state.get("messages"):
            result["messages"] = [
                AIMessage(content=_format_verification_chat_reply(
                    evidence, state.get("entity_fqn", "")
                ))
            ]
        return result


    return {"gather_verification_evidence": gather_verification_evidence}


__all__ = ["build_verification_nodes"]
