# Governance Agent Service

Standalone OpenMetadata-native AI Agent service.

## Capabilities

- Read-only OpenMetadata discovery using OpenMetadata MCP Bot (`get_entity_details`, `get_entity_lineage`).
- LangGraph classification node graph for fallback metadata sensitivity assessment.
- Direct interaction with OpenMetadata for retrieving context and creating Suggestions.
- Machine-only Bot identity (`governance-agent-bot`).

## Architectural Documentation & Guides

- [ARCHITECTURE_PATTERNS.md](ARCHITECTURE_PATTERNS.md) — 5 Clean Code design patterns for LangGraph & Agent structure.
- [AGENTS.md](AGENTS.md) — Coding Agent rules, invariants, and boundaries for parallel AI development.
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) — Parallel development roadmap for Phase 2 & Phase 3 specialist reasoning nodes.
- [OPERATIONS.md](OPERATIONS.md) — Deployment, 9router configuration, security, verification, and troubleshooting guide.

## Quick Start

```bash
# Activate backend conda environment or virtualenv
conda activate dg_backend

# Install dependencies
pip install -e .

# Run agent unit tests
pytest

# Run agent service
python -m app.main
```

## LLM configuration

The agent has one OpenAI-compatible structured-output provider adapter. To use
9router, copy `.env.example` to your deployment secret store and configure the
endpoint, machine-only API key, and model ID supplied for your tenant. See
[OPERATIONS.md](OPERATIONS.md) for the complete procedure.

```bash
export LLM_BASE_URL="https://<9router-endpoint>/v1"
export LLM_API_KEY="<9router-machine-key>"
export LLM_MODEL="<9router-model-id>"
python -m app.main
```

`LLM_BASE_URL` is optional. When it is unset, the adapter uses the default
OpenAI endpoint. Do not place OpenMetadata Execution Bot, Ranger, or Trino
credentials in the Agent environment.
