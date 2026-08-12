# R6-B OpenMetadata Context Fallback Hotfix

Baseline: `khanglv7repo/dg-agent@20691f1a1a464073368ad842a619d8058b7ce706`

## Problem proven live

OpenMetadata native REST returns real entity metadata and columns, while both the
Official SDK MCP read and fallback MCP can return tool-error payloads. The previous
Agent path could forward those error payloads to the LLM as if they were metadata,
causing REVIEW / NO_PROPOSAL decisions even for valid entities.

## Final read contract

Entity context used for LLM reasoning is accepted only if an actual entity mapping
for the requested FQN can be recovered.

Order:

1. Official SDK MCP details; lineage is best-effort.
2. Fallback MCP details; lineage is best-effort.
3. Native OpenMetadata REST `GET /api/v1/{collection}/name/{fqn}`.
4. If all three fail or return unusable data, fail closed. Never reason over a
   transport/tool error payload.

The existing authoritative OpenMetadata mutation/read-back implementation is not
changed. Production runner and Celery classification worker are switched to the
validated context gateway.
