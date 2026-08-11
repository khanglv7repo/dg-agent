from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.adapters.policy import to_backend_logical_policy
from app.schemas import (
    ColumnMask,
    LogicalPolicyProposal,
    PolicyResource,
    RowFilter,
    Subject,
)


def test_exact_backend_policy_adapter() -> None:
    proposal = LogicalPolicyProposal(
        subjects=[Subject(subject_type="USER", name="alice")],
        resource=PolicyResource(
            catalog="financial",
            schema="crm",
            table="customers",
        ),
        access={"select": "ALLOW"},
        masks=[ColumnMask(column="phone", mask_type="MASK")],
        row_filter=RowFilter(expression="customer_id <= 10"),
    )
    assert to_backend_logical_policy(proposal) == {
        "subjects": [{"type": "USER", "name": "alice"}],
        "resource": {
            "catalog": "financial",
            "schema": "crm",
            "table": "customers",
        },
        "access": {"select": "ALLOW"},
        "masks": {"phone": "MASK"},
        "row_filter": "customer_id <= 10",
    }


def test_unsupported_mask_is_rejected_by_production_schema() -> None:
    with pytest.raises(ValidationError):
        ColumnMask(column="phone", mask_type="MASK_HASH")
