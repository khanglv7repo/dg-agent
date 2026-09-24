"""Agent DQ governance writer: SPEC_DRAFT -> VALIDATED -> BACKEND_STAGED.

Per `planning/03-CODE-EDIT-MANIFEST.md`'s Agent "DQ writer" row and
`docs/13_IMPLEMENTATION_SPEC.md`'s "DQ Writer (Agent) CONDITIONAL ADD"
section (resolved: B2 PASS + B3 PASS -> Agent direct-create-TestCase is
viable). This module owns none of the durable idempotency/coordination
logic -- that is entirely Backend's `DQService.create_staged_test_case`
(natural_key_hash reservation, deterministic FQN, I1 crash/race safety).
This module's job is the Agent-side half of the boundary:

- SPEC_DRAFT: assemble a proposed TestCase spec from reasoning/rule output.
- VALIDATED (I8, validate-before-write): reject an incomplete/malformed spec
  BEFORE calling Backend -- mirrors the same required-field checks
  `DQService.create_staged_test_case` itself enforces, so a bad request
  fails fast on the Agent side instead of round-tripping to Backend only to
  get a 422.
- BACKEND_STAGED: `POST /api/v1/dq/test-cases` persists governance intent in
  Backend only. It does not create an OpenMetadata TestCase. A separate human
  approval + Backend materialization flow owns the later OM write.

Every write carries an `AIWriteAuditRef` (I9) and a `reason_code`, gated by
the same I11 kill switches as the classification/policy write boundaries
(`agent_write_to_om_enabled` is the master switch here too -- DQ TestCase
staging is an Agent-authored Backend write; OpenMetadata materialization is
owned later by Backend after human/operator approval).
"""
from __future__ import annotations

from typing import Any, Protocol

from app.audit_fingerprint import input_fingerprint, model_fingerprint, utc_now_iso
from app.clients.backend_rest import BackendRestError
from app.rate_limiter import RateLimiter, dq_write_rate_limiter


class DQTestCaseWriter(Protocol):
    """Bounded write channel. Only Backend's REST DQ endpoint is exposed;
    the LLM never sees this as a generic tool and cannot choose arbitrary
    Backend routes."""

    def create_dq_test_case(
        self,
        *,
        target_asset_fqn: str,
        test_definition_fqn: str,
        parameter_values: dict[str, Any],
        rule_id: str,
        worker_id: str,
        test_key: str | None = None,
        column_name: str | None = None,
        rationale: str | None = None,
    ) -> dict[str, Any]: ...


class DQSpecDraft:
    """SPEC_DRAFT: a proposed DQ TestCase before validation. Plain data
    holder -- deliberately not a pydantic model, since a draft is allowed to
    be incomplete (that's exactly what VALIDATED checks for)."""

    def __init__(
        self,
        *,
        target_asset_fqn: str | None,
        test_definition_fqn: str | None,
        rule_id: str | None,
        worker_id: str | None,
        parameter_values: dict[str, Any] | None = None,
        test_key: str | None = None,
        column_name: str | None = None,
        rationale: str | None = None,
    ) -> None:
        self.target_asset_fqn = target_asset_fqn
        self.test_definition_fqn = test_definition_fqn
        self.rule_id = rule_id
        self.worker_id = worker_id
        self.parameter_values = parameter_values or {}
        self.test_key = test_key
        self.column_name = column_name
        self.rationale = rationale


class DQValidationError(RuntimeError):
    """VALIDATED step failed -- write never attempted (I8 fail-closed)."""

    def __init__(self, missing_fields: list[str]) -> None:
        self.missing_fields = missing_fields
        super().__init__(
            "DQ TestCase spec failed validate-before-write: missing "
            f"{', '.join(missing_fields)}"
        )


class DQWriterService:
    def __init__(
        self,
        *,
        backend: DQTestCaseWriter,
        model_name: str,
        prompt_version: str,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.prompt_version = prompt_version
        # I10: rate limiting on repeated create_test_case-class calls, per
        # docs/13_IMPLEMENTATION_SPEC.md section 8. Lazily built (not at
        # import time) so tests that never call write() don't need a live
        # Redis; RateLimiter itself fails open if Redis is unreachable.
        self.rate_limiter = rate_limiter or dq_write_rate_limiter()

    @staticmethod
    def validate(draft: DQSpecDraft) -> None:
        """I8 validate-before-write. Mirrors the exact required-field set
        Backend's own `DQService.create_staged_test_case` and
        `DQTestCaseCreateRequest` enforce, so a bad draft fails on the Agent
        side rather than round-tripping to a 422."""
        missing = [
            name
            for name, value in (
                ("target_asset_fqn", draft.target_asset_fqn),
                ("test_definition_fqn", draft.test_definition_fqn),
                ("rule_id", draft.rule_id),
                ("worker_id", draft.worker_id),
            )
            if not value
        ]
        if missing:
            raise DQValidationError(missing)

    def _audit_ref(self, *, reason_code: str, draft: DQSpecDraft, validated_at: str | None) -> dict[str, Any]:
        return {
            "model_fingerprint": model_fingerprint(
                model_name=self.model_name, prompt_version=self.prompt_version
            ),
            "prompt_version": self.prompt_version,
            "input_fingerprint": input_fingerprint(
                {
                    "target_asset_fqn": draft.target_asset_fqn,
                    "test_definition_fqn": draft.test_definition_fqn,
                    "rule_id": draft.rule_id,
                    "test_key": draft.test_key,
                }
            ),
            "reason_code": reason_code,
            "validated_at": validated_at,
        }

    def write(
        self,
        draft: DQSpecDraft,
        *,
        agent_write_to_om_enabled: bool = True,
    ) -> dict[str, Any]:
        """Run SPEC_DRAFT -> VALIDATED -> STAGED. Returns a dict with
        `status` (STAGED/FAILED/SKIPPED/VALIDATION_FAILED),
        `reason_code`, `audit_ref`, and (on success) Backend's own response
        fields (`id`, `natural_key_hash`, `om_testcase_id`)."""
        # I11 master kill switch: fail safe before validation or any write.
        if not agent_write_to_om_enabled:
            return {
                "status": "SKIPPED",
                "reason_code": "KILL_SWITCH_DISABLED",
                "audit_ref": self._audit_ref(
                    reason_code="KILL_SWITCH_DISABLED", draft=draft, validated_at=None
                ),
            }

        try:
            self.validate(draft)
        except DQValidationError as exc:
            return {
                "status": "VALIDATION_FAILED",
                "reason_code": "VALIDATION_FAILED",
                "missing_fields": exc.missing_fields,
                "audit_ref": self._audit_ref(
                    reason_code="VALIDATION_FAILED", draft=draft, validated_at=None
                ),
            }

        # I10 rate limiting: checked after validation (a malformed request
        # shouldn't consume rate budget) but before the Backend call. Keyed
        # by target entity, matching I10's "repeated create_test_case-class
        # calls" scope -- this bounds runaway writes against the SAME asset,
        # not global throughput.
        rate_result = self.rate_limiter.check_and_increment(
            f"dq_write:{draft.target_asset_fqn}"
        )
        if not rate_result.allowed:
            return {
                "status": "RATE_LIMITED",
                "reason_code": "RATE_LIMIT_EXCEEDED",
                "rate_limit": {
                    "limit": rate_result.limit,
                    "window_seconds": rate_result.window_seconds,
                    "current_count": rate_result.current_count,
                },
                "audit_ref": self._audit_ref(
                    reason_code="RATE_LIMIT_EXCEEDED", draft=draft, validated_at=None
                ),
            }

        # Validate-before-write (I8) complete; timestamp strictly before the
        # first (only) write call, per Hard Invariant #13.
        validated_at = utc_now_iso()
        audit_ref = self._audit_ref(
            reason_code="DQ_DRAFT_STAGED", draft=draft, validated_at=validated_at
        )

        try:
            response = self.backend.create_dq_test_case(
                target_asset_fqn=draft.target_asset_fqn,  # type: ignore[arg-type]
                test_definition_fqn=draft.test_definition_fqn,  # type: ignore[arg-type]
                parameter_values=draft.parameter_values,
                rule_id=draft.rule_id,  # type: ignore[arg-type]
                worker_id=draft.worker_id,  # type: ignore[arg-type]
                test_key=draft.test_key,
                column_name=draft.column_name,
                rationale=draft.rationale,
            )
        except BackendRestError as exc:
            # CONFLICT (409, I1 race case) is not a failure of this write --
            # another worker already owns this natural_key_hash. Surface it
            # distinctly so a caller can treat it as a benign no-op, not an
            # error to retry/alert on.
            if exc.status_code == 409:
                return {
                    "status": "CONFLICT",
                    "reason_code": "STALE_GENERATION",
                    "audit_ref": audit_ref,
                    "backend_error": exc.as_dict(),
                }
            return {
                "status": "FAILED",
                "reason_code": "VALIDATION_FAILED"
                if exc.status_code == 422
                else "NO_MATCH",
                "audit_ref": audit_ref,
                "backend_error": exc.as_dict(),
            }

        return {
            "status": response.get("status", "STAGED"),
            "reason_code": "DQ_DRAFT_STAGED",
            "audit_ref": audit_ref,
            "id": response.get("id"),
            "natural_key_hash": response.get("natural_key_hash"),
            "om_testcase_id": response.get("om_testcase_id"),
        }


__all__ = ["DQWriterService", "DQSpecDraft", "DQValidationError", "DQTestCaseWriter"]
