from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.gateways.openmetadata import OpenMetadataGateway, OpenMetadataMutationError


def _gateway() -> OpenMetadataGateway:
    gateway = OpenMetadataGateway(
        endpoint="http://localhost:8585/mcp",
        token="token",
        fallback_mcp=MagicMock(),
    )
    gateway._sdk = None
    return gateway


def test_existing_confirmed_entity_tag_is_no_change() -> None:
    gateway = _gateway()
    gateway.get_taxonomies = MagicMock(return_value=["PII.Email"])
    gateway.get_entity_native = MagicMock(return_value={
        "id": "1",
        "tags": [{
            "tagFQN": "PII.Email",
            "source": "Classification",
            "labelType": "Automated",
            "state": "Confirmed",
        }],
        "columns": [],
    })
    gateway._apply_patch = MagicMock()
    result = gateway.apply_tag_authoritative(
        entity_type="table",
        entity_fqn="financial.crm.customers",
        tag_fqn="PII.Email",
    )
    assert result["status"] == "NO_CHANGE"
    assert result["mutation_count"] == 0
    gateway._apply_patch.assert_not_called()


def test_valid_new_column_tag_mutates_and_reads_back() -> None:
    gateway = _gateway()
    gateway.get_taxonomies = MagicMock(return_value=["PII.Email"])
    before = {
        "id": "1",
        "tags": [],
        "columns": [{
            "name": "email",
            "fullyQualifiedName": "financial.crm.customers.email",
            "tags": [],
        }],
    }
    after = {
        "id": "1",
        "tags": [],
        "columns": [{
            "name": "email",
            "fullyQualifiedName": "financial.crm.customers.email",
            "tags": [{
                "tagFQN": "PII.Email",
                "source": "Classification",
                "labelType": "Automated",
                "state": "Confirmed",
            }],
        }],
    }
    gateway.get_entity_native = MagicMock(side_effect=[before, after])
    gateway._apply_patch = MagicMock(return_value="NATIVE_API")

    result = gateway.apply_tag_authoritative(
        entity_type="table",
        entity_fqn="financial.crm.customers",
        tag_fqn="PII.Email",
        field_path="financial.crm.customers.email",
    )
    assert result["status"] == "APPLIED"
    assert result["mutation_count"] == 1
    patch = gateway._apply_patch.call_args.kwargs["patch"]
    assert patch[0]["path"] == "/columns/0/tags"
    assert patch[0]["value"][-1] == {
        "tagFQN": "PII.Email",
        "source": "Classification",
        "labelType": "Automated",
        "state": "Confirmed",
    }


def test_invented_tag_cannot_mutate() -> None:
    gateway = _gateway()
    gateway.get_taxonomies = MagicMock(return_value=["PII.Email"])
    gateway._apply_patch = MagicMock()
    with pytest.raises(OpenMetadataMutationError, match="not present"):
        gateway.apply_tag_authoritative(
            entity_type="table",
            entity_fqn="financial.crm.customers",
            tag_fqn="INVENTED.Tag",
        )
    gateway._apply_patch.assert_not_called()


def test_bad_field_path_cannot_mutate() -> None:
    gateway = _gateway()
    gateway.get_taxonomies = MagicMock(return_value=["PII.Email"])
    gateway.get_entity_native = MagicMock(return_value={
        "id": "1",
        "tags": [],
        "columns": [{
            "name": "email",
            "fullyQualifiedName": "financial.crm.customers.email",
            "tags": [],
        }],
    })
    gateway._apply_patch = MagicMock()
    with pytest.raises(OpenMetadataMutationError, match="not a column"):
        gateway.apply_tag_authoritative(
            entity_type="table",
            entity_fqn="financial.crm.customers",
            tag_fqn="PII.Email",
            field_path="other.table.password",
        )
    gateway._apply_patch.assert_not_called()
