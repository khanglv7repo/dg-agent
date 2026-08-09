# AI Coding-Agent Instructions for `agent/`

This document provides binding instructions for AI Coding Agents working on `agent/`.

## 1. Project Boundary & Responsibilities

- **Project Location**: `agent/`
- **Core Role**: LangGraph AI Agent executing reasoning flows for classification tags and logical data-access policy proposals.
- **Upstream Catalog**: OpenMetadata (accessed via OpenMetadata AI SDK / MCP client).
- **Backend Application**: Governance Backend (accessed via read-only Backend FastMCP client).

---

## 2. Architecture Invariants

1. **OpenMetadata AI SDK/MCP**: Use OpenMetadata AI SDK / MCP to read metadata, taxonomy, and lineage context, and apply allowed classification tags directly to OpenMetadata.
2. **Backend MCP Client**: Use Backend MCP client for `inspect_ranger_state`, `query_trino_readonly`, and policy inspection.
3. **No Direct Ranger/DB Writes**: The Agent MUST NOT hold Ranger or PostgreSQL credentials, and MUST NOT write directly to Ranger or PostgreSQL.
4. **Authoritative Policy Commands**: Authority-changing data-access policy proposals MUST be presented to human users for approval before submitting commands to Backend MCP.
5. **No Duplicate Event Loops**: Successful classification-tag writes to OpenMetadata rely on OM webhooks for downstream `TAG_SYNC`; the Agent does not emit separate write notifications to Backend.
6. **Interface Isolation**: Module boundaries communicate via Pydantic schemas in `app/schemas.py`.

---

## 3. Modular Architecture

```text
agent/app/
├── schemas.py                 <-- Shared Data Contracts (Pydantic DTOs)
├── classifier.py              <-- Configured OpenAI-compatible provider adapter
├── graph.py                   <-- Core LangGraph StateGraph & Node Routing
├── runner.py                  <-- High-Level Orchestrator
├── clients/
│   ├── mcp.py                 <-- Read-only OpenMetadata MCP Client
│   ├── backend_mcp.py         <-- Backend FastMCP Client
│   └── openmetadata.py        <-- REST Client for Direct Tag Mutation / Suggestions
└── reasoning/                 <-- Specialist Reasoning Modules
    ├── lineage_risk.py        <-- Lineage-aware sensitivity propagation
    ├── conflict_detector.py   <-- Policy conflict detection
    └── impact_analyzer.py    <-- Impact analysis on downstream assets
```

---

## 4. Verification

Run test suite after edits:
```bash
source /home/minh_chau/miniconda3/etc/profile.d/conda.sh
conda activate dg_backend
cd agent
pytest
```
