# Governance Agent Operations Guide

## 1. Purpose and boundary

`governance_agent` is a standalone LangGraph service. It reads catalog context
from OpenMetadata, asks one configured LLM provider for a structured
classification decision, filters that decision against an explicit allowed-tag
list, and creates native OpenMetadata Suggestions. A human reviewer accepts or
rejects each suggestion in OpenMetadata.

The Agent is not the backend and must not be coupled to it. It does not own a
FastAPI Control API, PostgreSQL job queue, Ranger policy reconciliation, Trino
verification, direct tag application, or a custom approval UI.

## 2. Safety contract

The following rules are enforced by the source code and must also be reflected
in deployment permissions:

| Area | Permitted | Forbidden |
|---|---|---|
| OpenMetadata MCP | `get_entity_details`, `get_entity_lineage`, `search_metadata`, `semantic_search`, `get_test_definitions` | Any MCP mutation tool |
| OpenMetadata REST | Create native `SuggestTagLabel` Suggestions using `governance-agent-bot` | Direct tag mutation, Suggestion acceptance, or use of the Execution Bot |
| LLM | One OpenAI-compatible endpoint with a machine credential | Personal API keys and raw data-row transfer |
| Enforcement | None | Ranger and Trino credentials, clients, or network access |

Every non-empty Agent result remains a proposed tag. It is never a confirmed
tag and never causes a Ranger or Trino action.

## 3. Runtime flow

```text
request
  -> OpenMetadata MCP: entity details (+ lineage when requested)
  -> configured 9router-compatible model: AgentDecision
  -> allow-list filter
  -> OpenMetadata REST: native SuggestTagLabel Suggestions
  -> OpenMetadata human review
```

The graph has two nodes: `load_context` and `classify`. `load_context` is
read-only. `classify` accepts only tags in the request's `allowed_tags`. The
runner then creates one native Suggestion for each remaining proposed tag. If
the result is empty, the run completes without a Suggestion mutation.

## 4. Configuration

Start from the checked-in template:

```bash
cp .env.example .env
```

Do not commit `.env`. Put production values in the deployment secret manager
or injected environment instead.

| Variable | Required | Default | Meaning |
|---|---:|---|---|
| `OPENMETADATA_MCP_URL` | Yes in deployment | `http://localhost:8585/mcp` | Read-only MCP endpoint for the Agent Bot |
| `OPENMETADATA_BASE_URL` | Yes in deployment | `http://localhost:8585` | REST endpoint used only to create Suggestions |
| `OPENMETADATA_AGENT_BOT_TOKEN` | Yes | none | Token for `governance-agent-bot`; it must not be the Execution Bot token |
| `LLM_BASE_URL` | Yes for 9router | unset | 9router OpenAI-compatible API base URL, normally ending in `/v1` when required by that tenant |
| `LLM_API_KEY` | Yes | falls back to `OPENAI_API_KEY` | 9router machine-only API key |
| `LLM_MODEL` | Yes in deployment | `gpt-4o-mini` | Exact model ID enabled for the 9router tenant |

The exact 9router endpoint URL and available model IDs are tenant-specific.
Use the values issued by its administrator or service documentation; this
repository deliberately does not hard-code an unverified URL or model.

The implementation passes `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` to
LangChain's `ChatOpenAI` client. This is an OpenAI-compatible transport, not a
second provider adapter.

## 5. Local setup and verification

The supported Python environment is `dg_backend`.

```bash
conda run -n dg_backend pip install -e '.[dev]'
conda run -n dg_backend pytest
conda run -n dg_backend python -m compileall -q app
```

Expected unit-test scope:

- read-only MCP tool allow-list rejects mutations before network invocation;
- Agent graph retrieves context and keeps only allow-listed tags;
- the classifier passes a configured `LLM_BASE_URL` to the one provider
  adapter.

These checks do not call OpenMetadata or 9router and do not prove credentials
or production API compatibility.

## 6. Deployment checklist

1. Create a distinct `governance-agent-bot` in OpenMetadata.
2. Grant that Bot only the MCP reads it needs and permission to create native
   Suggestions; do not grant direct tag-write or approval permissions.
3. Inject only the six variables in section 4 that apply to the deployment.
4. Allow egress only to the OpenMetadata MCP/REST hosts and the 9router host.
5. Confirm the runtime environment contains no Ranger, Trino, database, or
   Execution Bot credentials.
6. Run the local verification commands from section 5 on the release artifact.
7. Perform a non-production capability check with a known test entity and
   verify that the result appears as `Suggested` in OpenMetadata.
8. Have a human reviewer accept or reject it in OpenMetadata; do not automate
   that review step.

## 7. Runtime invocation and current limitation

The current `python -m app.main` entrypoint prints a service startup banner.
It does not yet expose an HTTP API or an automatic queue consumer. A caller
that integrates the Agent must instantiate `GovernanceAgentRunner` and pass an
`AgentRunRequest` containing an event ID, entity type/FQN, and a non-empty
allow-list. This boundary is intentionally separate from `governance_app`.

`include_lineage=true` causes the Agent to request `get_entity_lineage`; set it
to false only when lineage context is intentionally excluded for a run.

## 8. Troubleshooting

| Symptom | Likely cause | Safe action |
|---|---|---|
| `langchain-openai package is missing` | Dependencies are not installed in `dg_backend` | Run the install command in section 5 |
| `MCP tool is not allowed` | Code or integration requested a mutation tool | Keep the request within the read-only allow-list; do not bypass it |
| HTTP 401/403 from MCP or Suggestions | Agent Bot token/role is missing or wrong | Check the Bot identity and least-privilege role; never substitute a human or Execution token |
| HTTP 404/400 from 9router | Incorrect `LLM_BASE_URL` or model ID | Obtain the tenant-specific endpoint/model from 9router and retest in non-production |
| Suggested tag is missing | LLM returned no decision, or tag is absent from `allowed_tags` | Inspect the allowed-tag configuration and MCP metadata; do not auto-apply a replacement |
| Live test succeeds but no approval exists | A Suggestion was created, not confirmed | Complete human review in OpenMetadata |

## 9. Evidence required before production enablement

Record the date, target environment, Bot name (not token), model ID, and
result for each capability check. Production validation is complete only after
all of the following have executed successfully:

- MCP entity-detail retrieval using the Agent Bot;
- optional MCP lineage retrieval when enabled;
- 9router structured output parses as `AgentDecision`;
- an allow-listed tag creates exactly one native Suggestion;
- an unknown tag creates no Suggestion;
- a reviewer can see and decide the Suggestion in OpenMetadata.

Never record prompts containing secrets, authorization headers, or raw
business rows in validation evidence.
