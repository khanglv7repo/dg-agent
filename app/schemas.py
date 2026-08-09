from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# --- TAG DOMAIN SCHEMAS ---

class TagRecommendation(BaseModel):
    tag: str = Field(description="Fully qualified name of the recommended tag (must be in allowed_tags)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    rationale: str = Field(min_length=1, max_length=2000, description="Concise rationale citing metadata evidence")
    field_path: str | None = Field(default=None, max_length=1024, description="Target column FQN or None for entity level")
    action_recommendation: str = Field(default="APPLY", description="Action recommendation e.g. APPLY, REVIEW, NO_ACTION")


class TagReasoningResult(BaseModel):
    recommendations: list[TagRecommendation] = Field(default_factory=list, max_length=200)
    summary: str = Field(default="", max_length=4000)


# Backward-compatibility aliases for legacy code
class AgentTagSuggestion(BaseModel):
    tag: str
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=2000)
    field_path: str | None = Field(default=None, max_length=1024)


class AgentDecision(BaseModel):
    suggestions: list[AgentTagSuggestion] = Field(default_factory=list, max_length=200)
    summary: str = Field(default="", max_length=4000)


# --- POLICY DOMAIN SCHEMAS ---

class Subject(BaseModel):
    subject_type: Literal["USER", "GROUP"] = Field(description="Subject category: USER or GROUP")
    name: str = Field(min_length=1, max_length=255, description="Username or group name")


class PolicyResource(BaseModel):
    database: str = Field(min_length=1, max_length=255, description="Catalog / database name")
    schema_name: str = Field(min_length=1, max_length=255, description="Schema name")
    table: str = Field(min_length=1, max_length=255, description="Table name")


class ColumnMask(BaseModel):
    column: str = Field(min_length=1, max_length=255)
    mask_type: str = Field(min_length=1, max_length=64, description="Mask intent e.g., MASK_HASH, MASK_NULL, MASK_SHOW_LAST_4")


class RowFilter(BaseModel):
    expression: str | None = Field(default=None, description="Row filter SQL expression or null")


class LogicalPolicyProposal(BaseModel):
    subjects: list[Subject] = Field(min_length=1)
    resource: PolicyResource
    access: list[str] = Field(default_factory=lambda: ["SELECT"], description="Logical access intents e.g. ['SELECT']")
    masks: list[ColumnMask] = Field(default_factory=list)
    row_filter: RowFilter | None = Field(default=None)


class PolicyReasoningResult(BaseModel):
    proposal: LogicalPolicyProposal | None = Field(default=None)
    rationale: str = Field(min_length=1, max_length=2000)
    expected_impact: str = Field(default="", max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    warnings: list[str] = Field(default_factory=list)


# --- ORCHESTRATION / RUNNER SCHEMAS ---

class AgentRunRequest(BaseModel):
    request_type: Literal["TAG", "POLICY"] = Field(default="TAG", description="Explicit reasoning domain: TAG or POLICY")
    event_id: str = Field(default="req-local", min_length=1, max_length=255)
    entity_type: str = Field(default="table", min_length=1, max_length=64)
    entity_fqn: str = Field(min_length=1, max_length=1024)
    allowed_tags: list[str] = Field(default_factory=list, max_length=200)
    include_lineage: bool = True
    correlation_id: str | None = Field(default=None, max_length=128)

    # Optional policy inputs
    target_subjects: list[Subject] | None = Field(default=None)
    policy_intent: str | None = Field(default=None, max_length=2000)


class AgentRunResponse(BaseModel):
    status: str
    request_type: str = "TAG"
    decision: AgentDecision = Field(default_factory=AgentDecision)  # Backward compatibility field
    tag_result: TagReasoningResult | None = None
    policy_result: PolicyReasoningResult | None = None
    openmetadata_suggestion_ids: list[str] = Field(default_factory=list)

