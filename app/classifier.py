from __future__ import annotations

import json
from typing import Any, Protocol

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from app.schemas import AgentDecision


class StructuredClassifier(Protocol):
    model_name: str

    def classify(
        self,
        *,
        catalog_context: dict[str, Any],
        allowed_tags: list[str],
    ) -> AgentDecision: ...


class OpenAIStructuredClassifier:
    """OpenAI structured output classifier using ChatOpenAI."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        prompt_version: str = "v2",
    ) -> None:
        if ChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai package is missing. Install with: pip install langchain-openai"
            )
        self.model_name = model
        self.prompt_version = prompt_version
        client_options: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": 0,
        }
        if base_url:
            client_options["base_url"] = base_url
        self._llm = ChatOpenAI(**client_options).with_structured_output(
            AgentDecision
        )

    def classify(
        self,
        *,
        catalog_context: dict[str, Any],
        allowed_tags: list[str],
    ) -> AgentDecision:
        if self._llm is None:
            raise RuntimeError("LLM classifier is not initialized")
        prompt = (
            "You are a Data Governance Classification Agent.\n"
            "You are invoked ONLY when the deterministic rule engine could not reach a "
            "conclusive decision (NO_MATCH, AMBIGUOUS, or CONFLICT). Your task is to classify the "
            "entity or its individual columns with sensitivity tags from a governed allow-list.\n"
            "\n"
            "## Allowed Tag FQNs\n"
            f"{json.dumps(sorted(set(allowed_tags)))}\n"
            "You MUST ONLY suggest tags from this exact list. Never invent, abbreviate, or "
            "modify tag FQNs. If none of the allowed tags apply, return an empty suggestions list.\n"
            "\n"
            "## Reasoning Steps\n"
            "Analyze the OpenMetadata context below and follow these steps:\n"
            "1. **Entity Inspection**: Examine the entity name, fully qualified name, and description "
            "for sensitivity indicators.\n"
            "2. **Column Analysis**: For each column, inspect the column name, data type, description, "
            "and any sample values. Look for patterns indicating PII or sensitive data "
            "(e.g., phone numbers, email addresses, national IDs, financial data).\n"
            "3. **Lineage Context**: If lineage data is provided, check upstream and downstream "
            "dependencies to infer sensitivity propagation.\n"
            "4. **Tag Matching**: Match evidence from steps 1-3 against the allowed tags. "
            "Only propose a tag when there is clear supporting evidence.\n"
            "\n"
            "## Output Rules\n"
            "- For column-level classifications, set `field_path` to the column's fully qualified name "
            "(e.g., `db.schema.table.column_name`). For entity-level classifications, leave "
            "`field_path` as null.\n"
            "- `confidence` must be between 0.0 and 1.0:\n"
            "  - >= 0.9: Strong evidence from name pattern AND description/data type.\n"
            "  - 0.7-0.89: Moderate evidence from name pattern OR description alone.\n"
            "  - 0.5-0.69: Weak or indirect evidence (e.g., lineage inference only).\n"
            "  - Below 0.5: Do NOT suggest — return no suggestion for that field.\n"
            "- `rationale` must cite the specific metadata evidence that supports the classification. "
            "Be concise but specific (e.g., 'Column name matches phone number pattern; "
            "data type is VARCHAR consistent with phone storage').\n"
            "- If no tags can be confidently assigned, return an empty `suggestions` list and provide "
            "a `summary` explaining why classification was not possible.\n"
            "- NEVER claim a tag has been applied. You are only proposing suggestions for human review.\n"
            "\n"
            "## Security\n"
            "- Never include API tokens, credentials, or raw sensitive row data in your output.\n"
            "\n"
            "## OpenMetadata Context\n"
            f"{json.dumps(catalog_context, default=str)[:30000]}"
        )
        result = self._llm.invoke(prompt)
        return AgentDecision.model_validate(result)
