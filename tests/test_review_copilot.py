from __future__ import annotations

from unittest.mock import MagicMock

from app.review_copilot import ReviewCopilotService, gather_review_context
from app.schemas import CopilotRecommendation


def test_gather_review_context_includes_catalog_and_audit_evidence() -> None:
    om = MagicMock()
    om.get_entity_context.return_value = {"details": {"name": "customers"}}
    gov = MagicMock()
    gov.get_audit_summary.return_value = {"entries": []}

    context = gather_review_context(
        om_gateway=om,
        gov_gateway=gov,
        entity_type="table",
        entity_fqn="financial.crm.customers",
        policy_key=None,
    )
    assert context["catalog_context"] == {"details": {"name": "customers"}}
    assert context["audit_summary"] == {"entries": []}
    assert "existing_policy" not in context  # no policy_key -> no policy history
    assert context["warnings"] == []


def test_gather_review_context_includes_policy_history_when_policy_key_given() -> None:
    om = MagicMock()
    om.get_entity_context.return_value = {"details": {}}
    gov = MagicMock()
    gov.get_audit_summary.return_value = {}
    gov.get_policy.return_value = {"status": "ACTIVE"}
    gov.list_policy_versions.return_value = [{"version": 1}]
    gov.get_ranger_sync_status.return_value = {"projections": []}

    context = gather_review_context(
        om_gateway=om,
        gov_gateway=gov,
        entity_type="table",
        entity_fqn="financial.crm.customers",
        policy_key="policy-1",
    )
    assert context["existing_policy"] == {"status": "ACTIVE"}
    assert context["policy_versions"] == [{"version": 1}]
    assert context["ranger_sync_status"] == {"projections": []}


def test_gather_review_context_degrades_gracefully_on_gateway_failure() -> None:
    om = MagicMock()
    om.get_entity_context.side_effect = RuntimeError("OM unreachable")
    gov = MagicMock()
    gov.get_audit_summary.side_effect = RuntimeError("Backend unreachable")

    context = gather_review_context(
        om_gateway=om,
        gov_gateway=gov,
        entity_type="table",
        entity_fqn="financial.crm.customers",
        policy_key=None,
    )
    assert "catalog_context" not in context
    assert "audit_summary" not in context
    assert len(context["warnings"]) == 2


def test_service_ask_calls_answer_only_when_question_given() -> None:
    chat_model = MagicMock()
    chat_model.answer.return_value = "The answer."
    service = ReviewCopilotService(chat_model=chat_model)

    result = service.ask(context={}, question="Why?", want_recommendation=False)
    assert result["answer"] == "The answer."
    assert result["recommendation"] is None
    chat_model.recommend.assert_not_called()


def test_service_ask_calls_recommend_only_when_requested() -> None:
    chat_model = MagicMock()
    chat_model.recommend.return_value = CopilotRecommendation(
        suggestion="REJECT",
        rationale="Too risky.",
        confidence=0.9,
        key_risks=["PII exposure"],
    )
    service = ReviewCopilotService(chat_model=chat_model)

    result = service.ask(context={}, question=None, want_recommendation=True)
    assert result["answer"] is None
    assert result["recommendation"]["suggestion"] == "REJECT"
    assert result["recommendation"]["key_risks"] == ["PII exposure"]
    chat_model.answer.assert_not_called()


def test_service_ask_can_do_both_question_and_recommendation() -> None:
    chat_model = MagicMock()
    chat_model.answer.return_value = "Here's why."
    chat_model.recommend.return_value = CopilotRecommendation(
        suggestion="NEEDS_MORE_INFO",
        rationale="Missing context.",
        confidence=0.4,
    )
    service = ReviewCopilotService(chat_model=chat_model)

    result = service.ask(context={}, question="Explain?", want_recommendation=True)
    assert result["answer"] == "Here's why."
    assert result["recommendation"]["suggestion"] == "NEEDS_MORE_INFO"
