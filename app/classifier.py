"""Structured LLM classifiers for bounded TAG and POLICY reasoning."""
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
    PolicyReasoningResult,
    Subject,
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


def _build_structured_llm(
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    use_responses_api: bool,
    structured_output_method: str | None,
    schema: type,
):
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai package is missing")

    options: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
    }
    if base_url:
        options["base_url"] = base_url
    if use_responses_api:
        options["use_responses_api"] = True

    llm = ChatOpenAI(**options)
    structured_options: dict[str, Any] = {}
    if structured_output_method:
        structured_options["method"] = structured_output_method
    return llm.with_structured_output(schema, **structured_options)


class OpenAIStructuredClassifier:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        prompt_version: str = "v3",
        use_responses_api: bool = False,
        structured_output_method: str | None = None,
    ) -> None:
        self.model_name = model
        self.prompt_version = prompt_version
        self._llm = _build_structured_llm(
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_responses_api=use_responses_api,
            structured_output_method=structured_output_method,
            schema=TagReasoningResult,
        )

    def classify(
        self,
        *,
        catalog_context: dict[str, Any],
        allowed_tags: list[str],
    ) -> TagReasoningResult:
        prompt = (
            "You are a Data Governance Tag Classification Agent.\n"
            "Use only exact tag FQNs from Allowed Tag FQNs. Never invent taxonomy.\n"
            "Return structured recommendations with action_recommendation APPLY, REVIEW, "
            "or NO_ACTION. Automated mutation is controlled later by deterministic code; "
            "only APPLY can ever be mutated.\n"
            "Set field_path to a current entity column FQN/path or null for entity level.\n"
            "Do not access Ranger, Trino administration, or governance PostgreSQL directly.\n\n"
            "Allowed Tag FQNs:\n"
            f"{json.dumps(sorted(set(allowed_tags)))}\n\n"
            "Metadata Context:\n"
            f"{json.dumps(catalog_context, default=str)[:30000]}"
        )
        return TagReasoningResult.model_validate(self._llm.invoke(prompt))


class OpenAIPolicyClassifier:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        prompt_version: str = "v2",
        use_responses_api: bool = False,
        structured_output_method: str | None = None,
    ) -> None:
        self.model_name = model
        self.prompt_version = prompt_version
        self._llm = _build_structured_llm(
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_responses_api=use_responses_api,
            structured_output_method=structured_output_method,
            schema=PolicyReasoningResult,
        )

    def reason_policy(
        self,
        *,
        catalog_context: dict[str, Any],
        governance_context: dict[str, Any] | None,
        target_subjects: list[Subject] | None,
        policy_intent: str | None,
    ) -> PolicyReasoningResult:
        subjects_dump = (
            json.dumps([s.model_dump() for s in target_subjects])
            if target_subjects
            else "No explicit caller subjects"
        )
        prompt = (
            "You are a Data Governance Policy Reasoning Agent.\n"
            "Output logical policy intent only; never native Ranger JSON.\n"
            "Only USER and GROUP subjects are valid.\n"
            "When explicit target subjects are supplied, use only those identities.\n"
            "Do not invent privileged usernames/groups. If no explicit target subjects are "
            "available and a concrete subject is required, return proposal=null and explain.\n"
            "Backend R5 supports exactly one mask intent: MASK. Do not emit MASK_HASH, "
            "MASK_NULL, MASK_SHOW_LAST_4, or any other mask value.\n"
            "Do not activate, rollback, update mapping, or request mutation.\n"
            "Backend/Ranger/Trino context is diagnostic, not authority.\n\n"
            f"Target Subjects:\n{subjects_dump}\n\n"
            f"Policy Intent:\n{policy_intent or 'Reason about an appropriate policy.'}\n\n"
            "Catalog Context:\n"
            f"{json.dumps(catalog_context, default=str)[:20000]}\n\n"
            "Backend Governance Context:\n"
            f"{json.dumps(governance_context or {}, default=str)[:10000]}"
        )
        return PolicyReasoningResult.model_validate(self._llm.invoke(prompt))


__all__ = [
    "StructuredClassifier",
    "PolicyClassifier",
    "OpenAIStructuredClassifier",
    "OpenAIPolicyClassifier",
    "AgentDecision",
    "AgentTagSuggestion",
]
