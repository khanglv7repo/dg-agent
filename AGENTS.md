# AI Coding-Agent Instructions for `agent/`

This document provides binding instructions for AI Coding Agents working on `agent/`.

Read `../docs/00_README.md`, `../docs/DG_FINAL_SPEC.md`, `../docs/01_REQUIREMENTS.md`, `../docs/04_AI_LANGGRAPH_DESIGN.md`, and the relevant `../planning/tasks/` file first.

`../context/` was removed because it was stale. Do not rely on it or recreate it.

## 1. Project Boundary & Responsibilities

- **Project Location**: `agent/`
- **Core Role**: LangGraph AI Agent executing reasoning flows for data-access policy drafts, semantic classification suggestions, DQ spec assistance & RCA, and enforcement drift investigation.
- **Upstream Catalog**: OpenMetadata 2.0.2 (accessed via OpenMetadata MCP / AI SDK client).
- **Backend Application**: Governance Backend (accessed via Backend FastMCP / REST client).

---

## 2. Architecture Invariants & Guardrails

1. **AI Reasoning != Authority**: The Agent is an intelligent advisor and assistant, never an authoritative decision-maker or policy owner.
2. **Policy Proposals**: The Agent generates Logical Policy `DRAFT` proposals through the Backend. It CANNOT activate policies. Activation strictly requires explicit human approval.
3. **Classification Writes (SUGGEST Only)**: Semantic AI classification reasoning MUST output `SUGGEST`. It is NEVER directly applied to OpenMetadata. Direct classification/tag mutation is outside the Agent authority.
4. **Data Quality (DQ)**: The Agent may create bounded DQ TestCases only through the verified OpenMetadata MCP `create_test_case` path after code-level HITL/checkpointer approval. Deterministic test evaluation in OpenMetadata decides PASS/FAIL.
5. **No Direct Ranger or DB Writes**: The Agent MUST NOT hold Ranger or Backend PostgreSQL credentials, and MUST NOT connect directly to Ranger or PostgreSQL.
6. **Backend MCP Surface**: The Agent-facing Backend MCP contract is intentionally thin: `get_policy_context(asset_fqn?)`, `get_dq_requirement_spec(control_id)`, and `submit_policy_draft(draft)`. Do not invent policy activation, Ranger mutation, or Trino query tools unless the docs are explicitly changed.
7. **Idempotency & Downtime**: The Agent must be safe to restart and idempotent on re-execution. Missing executions are caught up via background reconciliation.

---

## 3. LangGraph Architecture

```text
agent/app/
├── schemas.py                 <-- Shared Data Contracts (Pydantic DTOs)
├── classifier.py              <-- Classification logic & structured schemas
├── graph.py                   <-- Core LangGraph StateGraph & Node Routing
├── runner.py                  <-- Agent Runner / Entrypoint
├── llm_runtime.py             <-- Configured LLM provider runtime
├── clients/
│   ├── mcp.py                 <-- OpenMetadata MCP Client
│   ├── backend_mcp.py         <-- Backend FastMCP Client
│   └── openmetadata.py        <-- OM adapter for reads and bounded DQ TestCase creation
├── gateways/                  <-- Context gateways
└── services/                  <-- Async classification worker and completion handlers
```

---

## 4. Verification

Run test suite after edits:
```bash
source /home/minh_chau/miniconda3/etc/profile.d/conda.sh
conda activate dg_backend
cd agent
pytest -v
```
