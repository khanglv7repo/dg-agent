from __future__ import annotations

from unittest.mock import MagicMock

from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway
from app.graph import run_governance_graph
from app.schemas import TagRecommendation, TagReasoningResult


def test_tag_path_preserves_fail_closed_taxonomy_boundary() -> None:
    om = MagicMock(spec=OpenMetadataGateway)
    om.get_entity_context.return_value = {"details": {"columns": [{"name": "email"}]}}
    om.get_taxonomies.return_value = []
    classifier = MagicMock()
    classifier.classify.return_value = TagReasoningResult(
        recommendations=[
            TagRecommendation(
                tag="INVENTED.Tag",
                confidence=0.99,
                rationale="invented",
                action_recommendation="APPLY",
            )
        ]
    )
    gov = MagicMock(spec=GovernanceGateway)
    tag_result, policy_result, _ = run_governance_graph(
        om_gateway=om,
        gov_gateway=gov,
        tag_classifier=classifier,
        policy_classifier=MagicMock(),
        request_type="TAG",
        entity_fqn="service.db.schema.table",
    )
    assert policy_result is None
    assert tag_result.recommendations == []
    gov.inspect_ranger_state.assert_not_called()
