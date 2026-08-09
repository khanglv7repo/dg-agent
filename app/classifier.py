"""Structured LLM classifiers for TAG reasoning and POLICY reasoning.

Per R6-A requirements:
- Reasoning is split into distinct TAG and POLICY domains.
- TAG reasoning proposes structured classification recommendations from allowed tag FQNs;
  it does NOT require OpenMetadata Suggestion creation.
- POLICY reasoning produces structured logical policy proposals (subjects, resource, access, masks, row_filter);
  it does NOT output native Ranger policy JSON or activate policies directly.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from app.schemas import (
    AgentDecision,
    AgentTagSuggestion,
    LogicalPolicyProposal,
    PolicyReasoningResult,
    PolicyResource,
    Subject,
    TagRecommendation,
    TagReasoningResult,
)


class StructuredClassifier(Protocol):
    model_name: str

    def classify(
        self,
        *,
        catalog_context: dict[str, Any],
        allowed_tags: list[str],
    ) -> TagReasoningResult: ...


class PolicyClassifier(Protocol):
    model_name: str

    def reason_policy(
        self,
        *,
        catalog_context: dict[str, Any],
        governance_context: dict[str, Any] | None,
        target_subjects: list[Subject] | None,
        policy_intent: str | None,
    ) -> PolicyReasoningResult: ...


class OpenAIStructuredClassifier:
    """OpenAI structured output classifier for TAG reasoning."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        prompt_version: str = "v3",
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
            TagReasoningResult
        )

    def classify(
        self,
        *,
        catalog_context: dict[str, Any],
        allowed_tags: list[str],
    ) -> TagReasoningResult:
        if self._llm is None:
            raise RuntimeError("LLM classifier is not initialized")
        prompt = (
            "You are a Data Governance Tag Classification Agent.\n"
            "Your task is to analyze entity metadata and recommend sensitivity classification tags.\n"
            "\n"
            "## Invariants & Guardrails\n"
            "1. NEVER invent, abbreviate, or modify tag FQNs. You MUST ONLY select tags from the allowed tag list below.\n"
            "2. If no allowed tag applies with confidence >= 0.5, return an empty recommendations list.\n"
            "3. You are generating structured proposals for human/system evaluation. Creation of OpenMetadata Suggestions is NOT required.\n"
            "4. Never attempt direct access to Ranger credentials, direct Ranger writes, or governance PostgreSQL DB.\n"
            "\n"
            "## Allowed Tag FQNs\n"
            f"{json.dumps(sorted(set(allowed_tags)))}\n"
            "\n"
            "## Output Requirements\n"
            "- Set `tag` to the exact allowed tag FQN.\n"
            "- Set `field_path` to column FQN for column-level classification, or null for entity-level.\n"
            "- Set `confidence` between 0.0 and 1.0 (do not report suggestions with confidence < 0.5).\n"
            "- Set `rationale` citing specific metadata evidence.\n"
            "- Set `action_recommendation` (e.g. 'APPLY', 'REVIEW', 'NO_ACTION').\n"
            "\n"
            "## Metadata Context\n"
            f"{json.dumps(catalog_context, default=str)[:30000]}"
        )
        result = self._llm.invoke(prompt)
        return TagReasoningResult.model_validate(result)


class OpenAIPolicyClassifier:
    """OpenAI structured output classifier for POLICY reasoning."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        prompt_version: str = "v1",
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
            PolicyReasoningResult
        )

    def reason_policy(
        self,
        *,
        catalog_context: dict[str, Any],
        governance_context: dict[str, Any] | None,
        target_subjects: list[Subject] | None,
        policy_intent: str | None,
    ) -> PolicyReasoningResult:
        if self._llm is None:
            raise RuntimeError("LLM policy classifier is not initialized")

        subjects_dump = (
            json.dumps([s.model_dump() for s in target_subjects])
            if target_subjects
            else "Not explicitly specified (infer from context or request)"
        )

        prompt = (
            "You are a Data Governance Policy Reasoning Agent.\n"
            "Your task is to analyze metadata and governance context to produce a logical data-access policy proposal.\n"
            "\n"
            "## Invariants & Guardrails\n"
            "1. Output MUST be a logical business domain policy proposal (subjects, resource, access, masks, row_filter).\n"
            "2. NEVER output native Ranger policy JSON as the business domain proposal.\n"
            "3. Subject categories MUST be explicit: USER or GROUP.\n"
            "4. Column masks and row filters MUST represent logical intent, not raw Ranger JSON.\n"
            "5. Do NOT attempt to activate or deploy policy directly. Policy activation requires Backend control plane approval.\n"
            "6. Ranger state inspection or Trino queries provided in context are diagnostic evidence only, NOT workflow truth.\n"
            "\n"
            "## Target Subjects\n"
            f"{subjects_dump}\n"
            "\n"
            "## Policy Intent / Request\n"
            f"{policy_intent or 'Infer appropriate access control and data masking policy based on metadata sensitivity tags.'}\n"
            "\n"
            "## Catalog Metadata Context\n"
            f"{json.dumps(catalog_context, default=str)[:20000]}\n"
            "\n"
            "## Governance Context (Ranger Inspection / Trino Diagnostic)\n"
            f"{json.dumps(governance_context or {}, default=str)[:10000]}"
        )
        result = self._llm.invoke(prompt)
        return PolicyReasoningResult.model_validate(result)
