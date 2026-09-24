"""Domain models for TAG and POLICY reasoning."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class TagRecommendation(BaseModel):
    tag: str = Field(description="Fully qualified tag name verified against OpenMetadata")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2000)
    field_path: str | None = Field(default=None, max_length=1024)
    action_recommendation: Literal["APPLY", "REVIEW", "NO_ACTION"] = "APPLY"


class TagReasoningResult(BaseModel):
    recommendations: list[TagRecommendation] = Field(default_factory=list, max_length=200)
    summary: str = Field(default="", max_length=4000)


class AgentTagSuggestion(BaseModel):
    tag: str
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=2000)
    field_path: str | None = Field(default=None, max_length=1024)


class AgentDecision(BaseModel):
    suggestions: list[AgentTagSuggestion] = Field(default_factory=list, max_length=200)
    summary: str = Field(default="", max_length=4000)


class Subject(BaseModel):
    subject_type: Literal["USER", "GROUP"]
    name: str = Field(min_length=1, max_length=255)


class PolicyResource(BaseModel):
    catalog: str = Field(min_length=1, max_length=255)
    schema_name: str = Field(alias="schema", min_length=1, max_length=255)
    table: str = Field(min_length=1, max_length=255)

    model_config = {"populate_by_name": True}


class ColumnMask(BaseModel):
    column: str = Field(min_length=1, max_length=255)
    mask_type: Literal["MASK"] = Field(
        default="MASK",
        description="Frozen Backend R5 supports exactly logical mask intent MASK",
    )


class RowFilter(BaseModel):
    expression: str | None = None


class LogicalPolicyProposal(BaseModel):
    subjects: list[Subject] = Field(min_length=1)
    resource: PolicyResource
    access: dict[str, Literal["ALLOW", "DENY"]] = Field(
        default_factory=lambda: {"select": "ALLOW"}
    )
    masks: list[ColumnMask] = Field(default_factory=list)
    row_filter: RowFilter | None = None


class PolicyLLMOutput(BaseModel):
    """Bug found live 2026-09-24: `PolicyReasoningResult` used to be passed
    directly to `with_structured_output()` for the LLM's own reasoning call
    (classifier.py's `OpenAIPolicyClassifier.reason_policy`). Every field on
    that model -- including ones meant to be set exclusively by code
    afterward (`reason_code`, `backend_context`, `backend_logical_policy`,
    `conflict`, `preview`, `draft`, `audit_ref`) -- was therefore part of the
    JSON schema handed to the LLM, free for it to fill with its own
    unconstrained guess. Observed live on a read-only run
    (persist_draft=False, so graph_policy.py's optional_create_draft returns
    before ever touching `reason_code`): the LLM invented
    `reason_code="ALLOW_WITH_COLUMN_MASK"`, which is not one of the frozen
    `PolicyReasonCode` values.

    This model is the fix: it is the ONLY schema ever handed to
    `with_structured_output()` for policy reasoning, containing exclusively
    the fields the LLM should legitimately produce. `classifier.py` maps
    this into a full `PolicyReasoningResult` afterward, leaving every
    code-owned field (reason_code and the others named above) at its default
    until graph_policy.py explicitly sets it.
    """

    proposal: LogicalPolicyProposal | None = None
    rationale: str = Field(min_length=1, max_length=2000)
    expected_impact: str = Field(default="", max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    warnings: list[str] = Field(default_factory=list)


class PolicyReasoningResult(BaseModel):
    proposal: LogicalPolicyProposal | None = None
    rationale: str = Field(min_length=1, max_length=2000)
    expected_impact: str = Field(default="", max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    warnings: list[str] = Field(default_factory=list)
    backend_context: dict[str, Any] = Field(default_factory=dict)
    backend_logical_policy: dict[str, Any] | None = None
    conflict: dict[str, Any] | None = None
    preview: dict[str, Any] | None = None
    draft: dict[str, Any] | None = None
    # TASK-09: write-boundary-only field, set exclusively by
    # graph_policy.py's optional_create_draft. See PolicyLLMOutput's
    # docstring above for why this can no longer be filled by the LLM.
    reason_code: str | None = None
    audit_ref: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# TASK-09: Frozen write-boundary schemas
# These satisfy I8 (validate-before-write), I9 (audit fingerprinting),
# and I11 (kill switches). Every AI-originated write MUST carry
# model_fingerprint + prompt_version; every write MUST complete validation
# before the first mutation call (validated_at != created_at per Hard
# Invariant #13).
# ---------------------------------------------------------------------------

# Reason codes for classification write paths (SUGGEST / APPLY / SKIP / BLOCKED)
ClassificationReasonCode = Literal[
    "RULE_TRUSTED_AUTO_APPLY",     # deterministic, high-confidence exact rule, auto-applied
    "RULE_UNTRUSTED_SUGGEST",      # rule match, confidence below auto-apply threshold
    "AGENT_SUGGEST",               # LLM-originated suggestion, SUGGEST only
    "KILL_SWITCH_DISABLED",        # feature flag OFF, write skipped
    "MANUAL_OVERRIDE_LOCKED",      # human override locked, write skipped
    "RATE_LIMIT_EXCEEDED",         # per-asset rate limit hit, write deferred
    "VALIDATION_FAILED",           # validate-before-write check failed, write blocked
    "STALE_GENERATION",            # generation fence fired, write skipped
    "COMPLETION_BOUND_EXCEEDED",   # LLM produced more than Backend limit, blocked
    "NO_MATCH",                    # no classification rule matched, no write
]

# Reason codes for policy write paths
PolicyReasonCode = Literal[
    "DRAFT_CREATED",               # DRAFT version persisted (human must approve)
    "DRAFT_SKIPPED_NO_KEY",        # persist_draft=True but no policy_key provided
    "DRAFT_SKIPPED_NO_SUBJECTS",   # persist_draft=True but no target_subjects
    "DRAFT_SKIPPED_NORMALIZATION", # policy normalization failed, DRAFT not persisted
    "DRAFT_SKIPPED_MAPPING",       # service mapping missing/mismatched, DRAFT blocked
    "CONFLICT_CHECK_FAILED",       # conflict check missing, DRAFT not persisted
    "KILL_SWITCH_DISABLED",        # feature flag OFF
]


class AIWriteAuditRef(BaseModel):
    """Attached to every AI-originated write per I9 (audit actor/timestamp/
    reason/model-fingerprint) and Hard Invariant #13 (validated_at != created_at).
    """
    model_fingerprint: str = Field(
        description="sha256[:12] of model name + version, for audit trail",
        min_length=1,
        max_length=64,
    )
    prompt_version: str = Field(
        description="Prompt template version identifier",
        min_length=1,
        max_length=128,
    )
    input_fingerprint: str | None = Field(
        default=None,
        description="sha256[:12] of the serialised LLM input context",
        max_length=64,
    )
    reason_code: str = Field(
        description="One of ClassificationReasonCode or PolicyReasonCode",
        min_length=1,
        max_length=64,
    )
    validated_at: str | None = Field(
        default=None,
        description=(
            "ISO-8601 UTC timestamp when validate-before-write completed "
            "(must differ from created_at per Hard Invariant #13)"
        ),
        max_length=32,
    )


class AgentRunRequest(BaseModel):
    request_type: Literal["TAG", "POLICY", "DQ", "VERIFICATION"] = "TAG"
    event_id: str = Field(default="req-local", min_length=1, max_length=255)
    entity_type: str = Field(default="table", min_length=1, max_length=64)
    entity_fqn: str = Field(min_length=1, max_length=1024)
    allowed_tags: list[str] = Field(default_factory=list, max_length=200)
    include_lineage: bool = True
    correlation_id: str | None = Field(default=None, max_length=128)

    target_subjects: list[Subject] | None = None
    policy_intent: str | None = Field(default=None, max_length=2000)
    policy_key: str | None = Field(default=None, min_length=1, max_length=512)
    persist_draft: bool = False
    environment: str = Field(default="local", min_length=1, max_length=64)

    # TASK-09: kill switch fields (I11 — independent per-path disable)
    # agent_write_to_om_enabled: master kill switch for ALL Agent→OM writes.
    # When False the graph completes but skips every OM mutation.
    agent_write_to_om_enabled: bool = True
    # TASK-09: DQ request fields (request_type="DQ" only). No LLM reasoning
    # in this path -- the caller supplies the already-decided TestCase spec;
    # see app/graph_dq.py / app/services/dq_writer.py.
    dq_test_definition_fqn: str | None = Field(default=None, max_length=512)
    dq_rule_id: str | None = Field(default=None, max_length=255)
    dq_worker_id: str | None = Field(default=None, max_length=255)
    dq_parameter_values: dict[str, Any] = Field(default_factory=dict)
    dq_test_key: str | None = Field(default=None, max_length=255)
    dq_column_name: str | None = Field(default=None, max_length=255)
    dq_rationale: str | None = Field(default=None, max_length=2000)

    # TASK-09: VERIFICATION request fields (request_type="VERIFICATION" only).
    # Read-only evidence gathering; see app/graph_verification.py.
    verification_audit_limit: int = Field(default=20, ge=1, le=200)
    verification_trino_check_sql: str | None = Field(
        default=None,
        max_length=4000,
        description=(
            "Caller-supplied, pre-built read-only SQL only -- never "
            "LLM-generated. See app/graph_verification.py's boundary note."
        ),
    )


class AgentRunResponse(BaseModel):
    status: str
    request_type: str = "TAG"
    decision: AgentDecision = Field(default_factory=AgentDecision)
    tag_result: TagReasoningResult | None = None
    policy_result: PolicyReasoningResult | None = None
    openmetadata_suggestion_ids: list[str] = Field(
        default_factory=list,
        description="Deprecated legacy field; not used by authoritative R6-B flow.",
    )
    # TASK-09: reason code + audit ref on every response
    reason_code: str | None = None
    audit_ref: AIWriteAuditRef | None = None
    # TASK-09: DQ / VERIFICATION results (populated only for their own
    # request_type; both are plain dicts since their shapes are Backend's
    # own DQTestCaseResponse / read-only evidence bundles, not Agent-owned
    # reasoning schemas).
    dq_result: dict[str, Any] | None = None
    verification_result: dict[str, Any] | None = None
