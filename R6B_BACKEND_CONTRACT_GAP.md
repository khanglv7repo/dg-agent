# R6-B BACKEND CONTRACT GAP

Backend baseline:

`f6d348d3422bd7f9f95cf125a85aef74829949ba`

## Missing capability

External Agent classification completion callback.

Frozen R5 exposes `get_workflow_status(execution_id)` but no supported MCP or REST
write contract that can finish the same `ClassificationExecution`.

The legacy `AgentClassificationResultService` creates a separate
`ClassificationRun`/Suggestion-oriented result and is not a public,
generation-fenced `ClassificationExecution` completion channel.

## Required semantics

```text
execution_id + generation
-> verify source == classification_execution
-> verify current status == WAITING_AI
-> verify generation is still current
-> set status COMPLETED or NO_PROPOSAL
-> persist bounded result/evidence
-> idempotent duplicate completion
-> stale generation must perform zero authority change
-> audit completion
```

## Agent behavior until Backend adds the contract

The `ai.classification` worker re-reads and fences Backend state, performs TAG
reasoning, then stops with `BLOCKED_COMPLETION_CHANNEL` before authoritative OM
mutation. It never accesses Backend PostgreSQL and never invents a Backend MCP
tool or undocumented endpoint.
