# Clean Architecture Patterns for `governance_agent`

This document defines the 5 standard, production-grade design patterns enforced in the `governance_agent` project to ensure code remains clean, maintainable, testable, and simple.

---

## 1. State-Graph Pattern (LangGraph Directed State Machine)

- **Concept**: Model the agent flow as a deterministic Directed State Graph (`StateGraph`).
- **Implementation**: Each graph node is a pure function that consumes `AgentState` and returns updated state fields:
  $$\text{Node: } \text{AgentState} \rightarrow \text{Partial[AgentState]}$$
- **Benefits**: Eliminates monolithic `if/else` logic, makes node transitions explicit and visualizable.

```python
# app/graph.py
graph = StateGraph(AgentState)
graph.add_node("load_context", load_context_node)
graph.add_node("classify", classify_node)

graph.add_edge(START, "load_context")
graph.add_edge("load_context", "classify")
graph.add_edge("classify", END)
```

---

## 2. Composable Pipeline / Chain-of-Nodes Pattern (SRP)

- **Concept**: Enforce Single Responsibility Principle (SRP) per node.
- **Node Pipeline Breakdown**:
  1. `LoadContextNode` — Fetches entity and lineage context via read-only MCP.
  2. `ClassifyNode` — Invokes LLM structured classification.
  3. `SanitizeNode` — Filters output against tag allow-list & confidence thresholds.
  4. `SuggestionNode` — Creates native OpenMetadata Suggestions via REST API.
- **Benefits**: Nodes are independently unit-testable with mock state inputs.

---

## 3. Strategy Pattern (Structured LLM Provider Abstraction)

- **Concept**: Decouple the reasoning graph from the configured OpenAI-compatible provider.
- **Implementation**: Define a Python `Protocol` interface (`StructuredClassifier`).

```python
# app/classifier.py
from typing import Protocol, Any
from app.schemas import AgentDecision

class StructuredClassifier(Protocol):
    """Provider-agnostic interface for LLM classifiers."""
    model_name: str
    def classify(self, *, catalog_context: dict[str, Any], allowed_tags: list[str]) -> AgentDecision:
        ...
```

- **Benefits**: Changing the configured model or injecting a mock classifier requires zero changes to `graph.py`. The current deployment has one provider adapter; adding provider-specific adapters is deferred.

---

## 4. Adapter Pattern (External System Clients)

- **Concept**: Encapsulate JSON-RPC protocols, SSE decoding, HTTP headers, and error handling inside client adapters.
- **Adapters**:
  - `OpenMetadataMCPClient` (`app/clients/mcp.py`): Wraps read-only JSON-RPC/SSE MCP calls (`get_entity_details`, `get_entity_lineage`).
  - `OpenMetadataAgentClient` (`app/clients/openmetadata.py`): Wraps REST API for submitting native Suggestions.
- **Benefits**: Graph nodes work with high-level Python methods instead of HTTP payloads.

---

## 5. Schema-Driven DTO Pattern (Pydantic / TypedDict)

- **Concept**: Enforce strict data contracts for inputs, outputs, and graph states.
- **Implementation**:
  - `AgentState`: `TypedDict` for LangGraph execution state.
  - `AgentRunRequest`, `AgentTagSuggestion`, `AgentDecision`, `AgentRunResponse`: Pydantic models in `app/schemas.py`.
- **Benefits**: Eliminates `KeyError` runtime crashes and untyped dictionary parsing.
