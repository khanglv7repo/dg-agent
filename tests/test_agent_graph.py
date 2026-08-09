from __future__ import annotations

from unittest.mock import MagicMock
from app.clients.mcp import OpenMetadataMCPClient
from app.graph import run_classification_graph
from app.schemas import AgentDecision, AgentTagSuggestion


def test_agent_graph_execution() -> None:
    mcp_mock = MagicMock(spec=OpenMetadataMCPClient)
    mcp_mock.entity_context.return_value = {
        "details": {"name": "customer_table", "columns": [{"name": "email"}]},
        "lineage": {},
    }

    classifier_mock = MagicMock()
    classifier_mock.model_name = "gpt-4o-mini"
    classifier_mock.classify.return_value = AgentDecision(
        suggestions=[
            AgentTagSuggestion(
                tag="PII.Email",
                confidence=0.95,
                rationale="Column email contains customer email addresses",
                field_path="customer_table.email",
            )
        ],
        summary="Found PII email column",
    )

    decision, context = run_classification_graph(
        mcp=mcp_mock,
        classifier=classifier_mock,
        entity_type="table",
        entity_fqn="service.db.schema.customer_table",
        allowed_tags=["PII.Email", "PII.Phone"],
        include_lineage=True,
    )

    assert len(decision.suggestions) == 1
    assert decision.suggestions[0].tag == "PII.Email"
    assert "details" in context
