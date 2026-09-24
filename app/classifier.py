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
    ChatIntent,
    CopilotRecommendation,
    PolicyLLMOutput,
    PolicyReasoningResult,
    Subject,
    TagReasoningResult,
)


class StructuredClassifier(Protocol):
    model_name: str
    prompt_version: str

    def classify(
        self,
        *,
        catalog_context: dict[str, Any],
        allowed_tags: list[str],
    ) -> TagReasoningResult: ...


class PolicyClassifier(Protocol):
    model_name: str
    prompt_version: str

    def reason_policy(
        self,
        *,
        catalog_context: dict[str, Any],
        governance_context: dict[str, Any] | None,
        target_subjects: list[Subject] | None,
        policy_intent: str | None,
    ) -> PolicyReasoningResult: ...


def _build_base_llm(
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    use_responses_api: bool,
    disable_thinking: bool,
    temperature: float = 0,
):
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai package is missing")

    options: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
    }
    if base_url:
        options["base_url"] = base_url
    if use_responses_api:
        options["use_responses_api"] = True
    if disable_thinking:
        # Verified live against DeepSeek's deepseek-flash model (2026-09-24):
        # its default "thinking mode" rejects a forced tool_choice, which
        # with_structured_output(method="function_calling") requires --
        # "400 Thinking mode does not support this tool_choice". Disabling
        # thinking mode via this provider-specific extra_body param is the
        # fix; harmless to omit for providers that don't recognize the key.
        options["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**options)


def _build_structured_llm(
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    use_responses_api: bool,
    structured_output_method: str | None,
    disable_thinking: bool,
    schema: type,
):
    llm = _build_base_llm(
        model=model,
        api_key=api_key,
        base_url=base_url,
        use_responses_api=use_responses_api,
        disable_thinking=disable_thinking,
    )
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
        disable_thinking: bool = False,
    ) -> None:
        self.model_name = model
        self.prompt_version = prompt_version
        self._llm = _build_structured_llm(
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_responses_api=use_responses_api,
            structured_output_method=structured_output_method,
            disable_thinking=disable_thinking,
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
        disable_thinking: bool = False,
    ) -> None:
        self.model_name = model
        self.prompt_version = prompt_version
        self._llm = _build_structured_llm(
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_responses_api=use_responses_api,
            structured_output_method=structured_output_method,
            disable_thinking=disable_thinking,
            # Bug fix (2026-09-24): must be the LLM-only PolicyLLMOutput
            # schema, not the full PolicyReasoningResult -- see
            # PolicyLLMOutput's docstring in schemas.py for why.
            schema=PolicyLLMOutput,
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
        llm_output = PolicyLLMOutput.model_validate(self._llm.invoke(prompt))
        # Wrap into the full result. Every code-owned field (reason_code,
        # backend_context, backend_logical_policy, conflict, preview, draft,
        # audit_ref) is left at its default here -- the LLM never sees or
        # sets these; graph_policy.py is the only writer for any of them.
        return PolicyReasoningResult(**llm_output.model_dump())


class OpenAICopilotChatModel:
    """TASK-09 review copilot: free-text `answer()` plus a separate
    structured `recommend()` call. Two distinct LLM configurations on
    purpose -- see `app/review_copilot.py:CopilotChatModel`'s docstring for
    why free-text chat and structured output are never mixed in one call
    here (the exact bug class `PolicyLLMOutput` already fixed once)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        prompt_version: str = "v1",
        use_responses_api: bool = False,
        structured_output_method: str | None = None,
        disable_thinking: bool = False,
    ) -> None:
        self.model_name = model
        self.prompt_version = prompt_version
        # Non-zero temperature for the free-text answer path -- a review
        # copilot chatting with a human benefits from natural phrasing;
        # every classifier above uses temperature=0 because their output is
        # parsed as structured data, not read as prose.
        self._chat_llm = _build_base_llm(
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_responses_api=use_responses_api,
            disable_thinking=disable_thinking,
            temperature=0.3,
        )
        self._recommend_llm = _build_structured_llm(
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_responses_api=use_responses_api,
            structured_output_method=structured_output_method,
            disable_thinking=disable_thinking,
            schema=CopilotRecommendation,
        )

    def answer(self, *, prompt: str) -> str:
        response = self._chat_llm.invoke(prompt)
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    def recommend(self, *, prompt: str) -> CopilotRecommendation:
        return CopilotRecommendation.model_validate(self._recommend_llm.invoke(prompt))


class OpenAIChatRouter:
    """Classifies a free-text chat message into a structured intent
    (`ChatIntent`) -- the entry point for Agent Chat UI's plain chat input,
    where a human types "hi" or a natural request instead of posting a
    structured `AgentRunRequest`. Single structured call, same discipline as
    every other classifier in this file (only the fields the LLM should
    produce are in the schema)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        prompt_version: str = "v1",
        use_responses_api: bool = False,
        structured_output_method: str | None = None,
        disable_thinking: bool = False,
    ) -> None:
        self.model_name = model
        self.prompt_version = prompt_version
        self._llm = _build_structured_llm(
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_responses_api=use_responses_api,
            structured_output_method=structured_output_method,
            disable_thinking=disable_thinking,
            schema=ChatIntent,
        )

    def classify(
        self,
        *,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> ChatIntent:
        history_text = (
            "\n".join(f"{m['role']}: {m['content']}" for m in conversation_history)
            if conversation_history
            else "(no prior messages)"
        )
        prompt = (
            "You are the entry point of a Data Governance Agent. A human "
            "typed a free-text message in a chat UI. Classify it into one "
            "of these 5 structured capabilities, or ask for clarification "
            "if there isn't enough information yet:\n\n"
            "- TAG: suggest/review classification tags for a specific "
            "OpenMetadata table (needs entity_fqn).\n"
            "- POLICY: reason about/propose a Ranger access policy for a "
            "specific table (needs entity_fqn).\n"
            "- DQ: create a data quality test case (needs entity_fqn and "
            "specific test details -- almost always needs clarification "
            "since a chat message rarely has all the required fields).\n"
            "- VERIFICATION: gather read-only audit/Ranger evidence for a "
            "specific table (needs entity_fqn).\n"
            "- COPILOT: answer a question or ask for an approve/reject "
            "opinion about a pending change on a specific table (needs "
            "entity_fqn and the question).\n\n"
            "If the message is a greeting or small talk with no topic at "
            "all, set needs_clarification=true and reply naturally and "
            "briefly -- greet back, or ask what they need. Never invent an "
            "entity_fqn that wasn't mentioned.\n\n"
            "If the user named or implied a topic (e.g. 'khách hàng', 'tài "
            "chính', 'giao dịch', 'CRM') but you don't know the exact table "
            "FQN, OR the user explicitly asked you to search/explore/find "
            "it yourself (e.g. 'tìm giúp', 'tự tìm đi', 'khám phá', "
            "'search', 'bạn tự tìm đi') -- even if that topic was mentioned "
            "several messages ago, not just the latest one -- set "
            "needs_clarification=true AND fill in search_query with a "
            "short keyword capturing that topic (translate to English if "
            "the OpenMetadata catalog is in English, e.g. 'tài chính' -> "
            "'financial', 'khách hàng' -> 'customer'; when unsure, keep the "
            "user's own word). Read the WHOLE conversation history for "
            "this, not just the latest message -- a follow-up like 'bạn tự "
            "tìm đi' has no topic of its own, the topic is in an earlier "
            "message. Do not also write reply_message in this case; the "
            "chat node will call search_metadata and reply with real "
            "results instead.\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"Latest message:\n{message}"
        )
        return ChatIntent.model_validate(self._llm.invoke(prompt))


__all__ = [
    "StructuredClassifier",
    "PolicyClassifier",
    "OpenAIStructuredClassifier",
    "OpenAIPolicyClassifier",
    "OpenAICopilotChatModel",
    "OpenAIChatRouter",
    "AgentDecision",
    "AgentTagSuggestion",
]
