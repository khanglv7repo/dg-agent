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


class AgentRunRequest(BaseModel):
    request_type: Literal["TAG", "POLICY"] = "TAG"
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
