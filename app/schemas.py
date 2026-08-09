from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=255)
    entity_type: str = Field(default="table", min_length=1, max_length=64)
    entity_fqn: str = Field(min_length=1, max_length=1024)
    allowed_tags: list[str] = Field(min_length=1, max_length=200)
    include_lineage: bool = True
    correlation_id: str | None = Field(default=None, max_length=128)


class AgentTagSuggestion(BaseModel):
    tag: str
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=2000)
    field_path: str | None = Field(default=None, max_length=1024)


class AgentDecision(BaseModel):
    suggestions: list[AgentTagSuggestion] = Field(default_factory=list, max_length=200)
    summary: str = Field(default="", max_length=4000)


class AgentRunResponse(BaseModel):
    status: str
    decision: AgentDecision
    openmetadata_suggestion_ids: list[str] = Field(default_factory=list)
