from __future__ import annotations

from typing import Any

from app import classifier
from app.schemas import TagReasoningResult


class _FakeChatOpenAI:
    init_options: dict[str, Any]
    schema: type[TagReasoningResult]

    def __init__(self, **kwargs: Any) -> None:
        type(self).init_options = kwargs

    def with_structured_output(self, schema: type[TagReasoningResult]) -> _FakeChatOpenAI:
        type(self).schema = schema
        return self


def test_openai_compatible_classifier_uses_configured_base_url(monkeypatch) -> None:
    monkeypatch.setattr(classifier, "ChatOpenAI", _FakeChatOpenAI)

    result = classifier.OpenAIStructuredClassifier(
        model="9router-model",
        api_key="machine-key",
        base_url="https://router.example/v1",
    )

    assert result.model_name == "9router-model"
    assert _FakeChatOpenAI.init_options == {
        "model": "9router-model",
        "api_key": "machine-key",
        "temperature": 0,
        "base_url": "https://router.example/v1",
    }
    assert _FakeChatOpenAI.schema is TagReasoningResult
