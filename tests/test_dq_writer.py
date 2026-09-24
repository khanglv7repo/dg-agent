from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.clients.backend_rest import BackendRestError
from app.services.dq_writer import DQSpecDraft, DQValidationError, DQWriterService


def valid_draft(**overrides) -> DQSpecDraft:
    defaults = dict(
        target_asset_fqn="financial.crm.customers",
        test_definition_fqn="columnValuesToBeNotNull",
        rule_id="rule-pii-1",
        worker_id="agent-worker-1",
        parameter_values={},
        test_key=None,
        column_name="email",
        rationale="PII column should never be null",
    )
    defaults.update(overrides)
    return DQSpecDraft(**defaults)


def writer(backend: MagicMock) -> DQWriterService:
    return DQWriterService(backend=backend, model_name="test-model", prompt_version="v1")


def test_validate_passes_for_complete_draft() -> None:
    # Should not raise.
    DQWriterService.validate(valid_draft())


@pytest.mark.parametrize(
    "field",
    ["target_asset_fqn", "test_definition_fqn", "rule_id", "worker_id"],
)
def test_validate_rejects_missing_required_field(field: str) -> None:
    draft = valid_draft(**{field: None})
    with pytest.raises(DQValidationError) as exc_info:
        DQWriterService.validate(draft)
    assert field in exc_info.value.missing_fields


def test_write_master_kill_switch_skips_validation_and_backend_call() -> None:
    backend = MagicMock()
    service = writer(backend)
    result = service.write(valid_draft(), agent_write_to_om_enabled=False)

    assert result["status"] == "SKIPPED"
    assert result["reason_code"] == "KILL_SWITCH_DISABLED"
    assert result["audit_ref"]["validated_at"] is None
    backend.create_dq_test_case.assert_not_called()


def test_write_incomplete_draft_fails_validation_before_backend_call() -> None:
    backend = MagicMock()
    service = writer(backend)
    result = service.write(valid_draft(rule_id=None))

    assert result["status"] == "VALIDATION_FAILED"
    assert result["reason_code"] == "VALIDATION_FAILED"
    assert "rule_id" in result["missing_fields"]
    assert result["audit_ref"]["validated_at"] is None
    backend.create_dq_test_case.assert_not_called()


def test_write_success_calls_backend_with_validated_at_before_call() -> None:
    backend = MagicMock()
    backend.create_dq_test_case.return_value = {
        "id": "tc-1",
        "natural_key_hash": "dg_abc",
        "om_testcase_id": None,
        "status": "STAGED",
    }
    service = writer(backend)
    result = service.write(valid_draft())

    assert result["status"] == "STAGED"
    assert result["reason_code"] == "RULE_TRUSTED_AUTO_APPLY"
    assert result["natural_key_hash"] == "dg_abc"
    assert result["om_testcase_id"] is None
    assert result["audit_ref"]["validated_at"] is not None
    assert result["audit_ref"]["model_fingerprint"]
    assert result["audit_ref"]["prompt_version"] == "v1"
    backend.create_dq_test_case.assert_called_once_with(
        target_asset_fqn="financial.crm.customers",
        test_definition_fqn="columnValuesToBeNotNull",
        parameter_values={},
        rule_id="rule-pii-1",
        worker_id="agent-worker-1",
        test_key=None,
        column_name="email",
        rationale="PII column should never be null",
    )


def test_write_409_conflict_is_reported_as_benign_conflict_not_failure() -> None:
    backend = MagicMock()
    backend.create_dq_test_case.side_effect = BackendRestError(
        status_code=409,
        code="CONFLICT",
        message="already reserved",
    )
    service = writer(backend)
    result = service.write(valid_draft())

    assert result["status"] == "CONFLICT"
    assert result["reason_code"] == "STALE_GENERATION"
    assert result["backend_error"]["status_code"] == 409


def test_write_422_from_backend_is_reported_as_failed_validation() -> None:
    backend = MagicMock()
    backend.create_dq_test_case.side_effect = BackendRestError(
        status_code=422,
        code="VALIDATION_ERROR",
        message="bad field",
    )
    service = writer(backend)
    result = service.write(valid_draft())

    assert result["status"] == "FAILED"
    assert result["reason_code"] == "VALIDATION_FAILED"


def test_write_other_backend_error_is_reported_as_failed() -> None:
    backend = MagicMock()
    backend.create_dq_test_case.side_effect = BackendRestError(
        status_code=503,
        code="EXTERNAL_SYSTEM_ERROR",
        message="OM down",
    )
    service = writer(backend)
    result = service.write(valid_draft())

    assert result["status"] == "FAILED"
    assert result["reason_code"] == "NO_MATCH"
    assert result["backend_error"]["retryable"] is True
