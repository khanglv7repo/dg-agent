# R6-B Agent Completion Wiring

Baseline: `khanglv7repo/dg-agent@f60b7e2d77c0e25ffafac719ce138ba8f9b2db8f`

This overlay connects the existing `ClassificationWorkerService` completion
protocol to Backend R6-B's bounded MCP continuation tool.

Frozen R5 remains exactly 15 tools. R6-B is exactly R5 plus
`complete_classification_execution`.

Production flow: `ai.classification` -> generation fence #1 -> OM context and
taxonomy -> structured LLM reasoning -> generation fence #2 -> authoritative OM
tag mutation/read-back -> Backend completion -> durable terminal state/audit.

No human confirmation is added because completion continues the same already-
dispatched generation; it does not create a new governance intent.
