"""Shared LLM runtime configuration for runner and Celery worker."""
from __future__ import annotations

from dataclasses import dataclass
import os

from app.classifier import OpenAIPolicyClassifier, OpenAIStructuredClassifier

_ALLOWED_STRUCTURED_METHODS = frozenset({"json_schema", "function_calling", "json_mode"})


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMRuntimeConfig:
    model: str
    api_key: str
    base_url: str | None
    use_responses_api: bool
    structured_output_method: str | None

    @classmethod
    def from_env(cls) -> "LLMRuntimeConfig":
        model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
        base_url = (os.getenv("LLM_BASE_URL") or "").strip() or None
        use_responses_api = _env_bool("LLM_USE_RESPONSES_API", False)

        raw_method = (os.getenv("LLM_STRUCTURED_OUTPUT_METHOD") or "").strip()
        structured_output_method = raw_method or (
            "json_schema" if use_responses_api else None
        )
        if (
            structured_output_method is not None
            and structured_output_method not in _ALLOWED_STRUCTURED_METHODS
        ):
            allowed = ", ".join(sorted(_ALLOWED_STRUCTURED_METHODS))
            raise ValueError(
                "LLM_STRUCTURED_OUTPUT_METHOD must be one of: "
                f"{allowed}"
            )

        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_responses_api=use_responses_api,
            structured_output_method=structured_output_method,
        )

    def tag_classifier(self) -> OpenAIStructuredClassifier:
        return OpenAIStructuredClassifier(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            use_responses_api=self.use_responses_api,
            structured_output_method=self.structured_output_method,
        )

    def policy_classifier(self) -> OpenAIPolicyClassifier:
        return OpenAIPolicyClassifier(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            use_responses_api=self.use_responses_api,
            structured_output_method=self.structured_output_method,
        )


__all__ = ["LLMRuntimeConfig"]
