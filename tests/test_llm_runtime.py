from __future__ import annotations

from unittest.mock import MagicMock

import app.classifier as classifier_module
from app.llm_runtime import LLMRuntimeConfig


def test_responses_runtime_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "DeepSeek-V4-Flash")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv(
        "LLM_BASE_URL",
        "https://aiportalapi.stu-platform.live/jpe/v2",
    )
    monkeypatch.setenv("LLM_USE_RESPONSES_API", "true")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "json_schema")

    config = LLMRuntimeConfig.from_env()

    assert config.model == "DeepSeek-V4-Flash"
    assert config.api_key == "test-key"
    assert config.base_url == "https://aiportalapi.stu-platform.live/jpe/v2"
    assert config.use_responses_api is True
    assert config.structured_output_method == "json_schema"


def test_responses_api_defaults_to_json_schema(monkeypatch) -> None:
    monkeypatch.delenv("LLM_STRUCTURED_OUTPUT_METHOD", raising=False)
    monkeypatch.setenv("LLM_USE_RESPONSES_API", "true")

    config = LLMRuntimeConfig.from_env()

    assert config.structured_output_method == "json_schema"


def test_tag_classifier_wires_responses_api_and_json_schema(monkeypatch) -> None:
    structured = MagicMock(name="structured")
    client = MagicMock(name="chat-openai")
    client.with_structured_output.return_value = structured
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(classifier_module, "ChatOpenAI", factory)

    config = LLMRuntimeConfig(
        model="DeepSeek-V4-Flash",
        api_key="test-key",
        base_url="https://aiportalapi.stu-platform.live/jpe/v2",
        use_responses_api=True,
        structured_output_method="json_schema",
    )
    built = config.tag_classifier()

    assert built._llm is structured
    options = factory.call_args.kwargs
    assert options["model"] == "DeepSeek-V4-Flash"
    assert options["api_key"] == "test-key"
    assert options["temperature"] == 0
    assert options["base_url"] == "https://aiportalapi.stu-platform.live/jpe/v2"
    assert options["use_responses_api"] is True

    schema = client.with_structured_output.call_args.args[0]
    assert schema.__name__ == "TagReasoningResult"
    assert client.with_structured_output.call_args.kwargs == {
        "method": "json_schema"
    }


def test_policy_classifier_uses_same_transport(monkeypatch) -> None:
    structured = MagicMock(name="structured")
    client = MagicMock(name="chat-openai")
    client.with_structured_output.return_value = structured
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(classifier_module, "ChatOpenAI", factory)

    config = LLMRuntimeConfig(
        model="DeepSeek-V4-Flash",
        api_key="test-key",
        base_url="https://aiportalapi.stu-platform.live/jpe/v2",
        use_responses_api=True,
        structured_output_method="json_schema",
    )
    built = config.policy_classifier()

    assert built._llm is structured
    assert factory.call_args.kwargs["use_responses_api"] is True
    schema = client.with_structured_output.call_args.args[0]
    assert schema.__name__ == "PolicyReasoningResult"
    assert client.with_structured_output.call_args.kwargs["method"] == "json_schema"


def test_invalid_structured_method_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "invented-mode")

    try:
        LLMRuntimeConfig.from_env()
    except ValueError as exc:
        assert "LLM_STRUCTURED_OUTPUT_METHOD" in str(exc)
    else:
        raise AssertionError("invalid structured output method must be rejected")
