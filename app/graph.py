"""Deterministic LangGraph flows for TAG and POLICY reasoning."""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.adapters.policy import PolicyAdapterError, to_backend_logical_policy
from app.classifier import PolicyClassifier, StructuredClassifier
from app.clients.backend_mcp import BackendMCPError
from app.gateways.governance import GovernanceGateway
from app.gateways.openmetadata import OpenMetadataGateway
from app.schemas import PolicyReasoningResult, Subject, TagReasoningResult

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    request_type: str
    entity_type: str
    entity_fqn: str
    include_lineage: bool
    allowed_tags: list[str]
    target_subjects: list[dict[str, Any]] | None
    policy_intent: str | None
    policy_key: str | None
    persist_draft: bool
    environment: str
    catalog_context: dict[str, Any]
    governance_context: dict[str, Any]
    tag_result: dict[str, Any]
    policy_result: dict[str, Any]
    backend_logical_policy: dict[str, Any] | None


def compute_effective_allowed_tags(
    actual_om_tags: list[str],
    caller_allowed_tags: list[str],
) -> list[str]:
    if not actual_om_tags:
        return []
    if caller_allowed_tags:
        actual_set = set(actual_om_tags)
        return [tag for tag in caller_allowed_tags if tag in actual_set]
    return list(actual_om_tags)


def _details_dict(context: dict[str, Any]) -> dict[str, Any]:
    details = context.get("details")
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    if isinstance(details, dict) and isinstance(details.get("data"), dict):
        details = details["data"]
    return details if isinstance(details, dict) else {}


def _om_service_name(context: dict[str, Any]) -> str | None:
    details = _details_dict(context)
    service = details.get("service")
    if hasattr(service, "model_dump"):
        service = service.model_dump()
    if isinstance(service, dict):
        for key in ("name", "fullyQualifiedName"):
            value = service.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(service, str) and service.strip():
        return service.strip()
    return None


def _append_policy_warning(state: AgentState, message: str) -> dict[str, Any]:
    current = dict(state.get("policy_result") or {})
    warnings = list(current.get("warnings") or [])
    if message not in warnings:
        warnings.append(message)
    current["warnings"] = warnings
    return current


def build_governance_graph(
    *,
    om_gateway: OpenMetadataGateway,
    gov_gateway: GovernanceGateway,
    tag_classifier: StructuredClassifier,
    policy_classifier: PolicyClassifier,
):
    def route_intent(state: AgentState) -> str:
        return "POLICY" if (state.get("request_type") or "TAG").upper() == "POLICY" else "TAG"

    def load_om_context(state: AgentState) -> AgentState:
        return {
            "catalog_context": om_gateway.get_entity_context(
                entity_type=state.get("entity_type", "table"),
                entity_fqn=state["entity_fqn"],
                include_lineage=bool(state.get("include_lineage", True)),
            )
        }

    def tag_reasoning(state: AgentState) -> AgentState:
        try:
            actual_om_tags = om_gateway.get_taxonomies()
        except Exception as exc:
            logger.warning("Failed to fetch OM taxonomy: %s", exc)
            actual_om_tags = []
        effective = compute_effective_allowed_tags(
            actual_om_tags,
            state.get("allowed_tags", []),
        )
        raw = tag_classifier.classify(
            catalog_context=state.get("catalog_context", {}),
            allowed_tags=effective,
        )
        allowed = set(effective)
        raw.recommendations = [rec for rec in raw.recommendations if rec.tag in allowed]
        return {"tag_result": raw.model_dump(mode="json")}

    def load_backend_policy_context(state: AgentState) -> AgentState:
        context: dict[str, Any] = {}
        warnings: list[str] = []

        try:
            context["ranger_health"] = gov_gateway.inspect_ranger_state(kind="health")
        except Exception as exc:
            warnings.append(f"Ranger diagnostic unavailable: {exc}")

        service_name = _om_service_name(state.get("catalog_context", {}))
        if service_name:
            try:
                context["service_mapping"] = gov_gateway.resolve_resource_mapping(
                    om_service_name=service_name,
                    environment=state.get("environment", "local"),
                )
            except Exception as exc:
                warnings.append(
                    f"Exact service mapping unresolved for {service_name!r}: {exc}"
                )
        else:
            warnings.append("OpenMetadata service identity unavailable for exact mapping")

        policy_key = state.get("policy_key")
        if policy_key:
            try:
                context["existing_policy"] = gov_gateway.get_policy(policy_key)
                context["versions"] = gov_gateway.list_policy_versions(policy_key)
                context["sync_status"] = gov_gateway.get_ranger_sync_status(
                    policy_key=policy_key
                )
            except BackendMCPError as exc:
                if exc.code == "NOT_FOUND":
                    context["existing_policy"] = None
                    context["versions"] = []
                else:
                    raise

        context["warnings"] = warnings
        return {"governance_context": context}

    def policy_reasoning(state: AgentState) -> AgentState:
        raw_subjects = state.get("target_subjects")
        subjects = [Subject.model_validate(s) for s in raw_subjects] if raw_subjects else None
        result = policy_classifier.reason_policy(
            catalog_context=state.get("catalog_context", {}),
            governance_context=state.get("governance_context", {}),
            target_subjects=subjects,
            policy_intent=state.get("policy_intent"),
        )
        result.backend_context = dict(state.get("governance_context", {}))
        return {"policy_result": result.model_dump(mode="json")}

    def normalize_backend_policy(state: AgentState) -> AgentState:
        result = dict(state.get("policy_result") or {})
        proposal_raw = result.get("proposal")
        if not proposal_raw:
            return {"policy_result": result, "backend_logical_policy": None}

        proposal = PolicyReasoningResult.model_validate(result).proposal
        assert proposal is not None
        raw_subjects = state.get("target_subjects")
        explicit = [Subject.model_validate(s) for s in raw_subjects] if raw_subjects else None
        try:
            document = to_backend_logical_policy(
                proposal,
                explicit_subjects=explicit,
                require_explicit_subjects=bool(state.get("persist_draft", False)),
            )
        except PolicyAdapterError as exc:
            result = _append_policy_warning(state, str(exc))
            result["backend_logical_policy"] = None
            return {"policy_result": result, "backend_logical_policy": None}

        result["backend_logical_policy"] = document
        return {"policy_result": result, "backend_logical_policy": document}

    def check_conflict(state: AgentState) -> AgentState:
        result = dict(state.get("policy_result") or {})
        document = state.get("backend_logical_policy")
        policy_key = state.get("policy_key")
        if not document or not policy_key:
            if document and not policy_key:
                result = _append_policy_warning(
                    state,
                    "policy_key is required for Backend conflict/preview and DRAFT persistence",
                )
            return {"policy_result": result}
        conflict = gov_gateway.check_policy_conflict(
            policy_key=policy_key,
            logical_policy=document,
        )
        result["conflict"] = conflict
        return {"policy_result": result}

    def preview_change(state: AgentState) -> AgentState:
        result = dict(state.get("policy_result") or {})
        document = state.get("backend_logical_policy")
        policy_key = state.get("policy_key")
        if not document or not policy_key:
            return {"policy_result": result}
        preview = gov_gateway.preview_policy_change(
            policy_key=policy_key,
            logical_policy=document,
        )
        result["preview"] = preview
        return {"policy_result": result}

    def optional_create_draft(state: AgentState) -> AgentState:
        result = dict(state.get("policy_result") or {})
        if not state.get("persist_draft", False):
            return {"policy_result": result}

        document = state.get("backend_logical_policy")
        policy_key = state.get("policy_key")
        if not policy_key:
            result = _append_policy_warning(
                state, "persist_draft=true requires explicit policy_key"
            )
            return {"policy_result": result}
        if not state.get("target_subjects"):
            result = _append_policy_warning(
                state, "persist_draft=true requires explicit target_subjects"
            )
            return {"policy_result": result}
        if not document:
            result = _append_policy_warning(
                state, "DRAFT was not persisted because policy normalization failed"
            )
            return {"policy_result": result}

        mapping = (state.get("governance_context") or {}).get("service_mapping")
        if not isinstance(mapping, dict):
            result = _append_policy_warning(
                state,
                "persist_draft=true requires an exact resolved OpenMetadata service mapping",
            )
            return {"policy_result": result}
        mapped_catalog = str(mapping.get("trino_catalog") or "").strip()
        document_catalog = str(
            (document.get("resource") or {}).get("catalog") or ""
        ).strip()
        if not mapped_catalog:
            result = _append_policy_warning(
                state,
                "resolved service mapping is missing trino_catalog",
            )
            return {"policy_result": result}
        if document_catalog != mapped_catalog:
            result = _append_policy_warning(
                state,
                (
                    "policy resource catalog does not match exact service mapping: "
                    f"{document_catalog!r} != {mapped_catalog!r}"
                ),
            )
            return {"policy_result": result}

        if result.get("preview") is None or result.get("conflict") is None:
            result = _append_policy_warning(
                state, "DRAFT was not persisted because normalization/preview/conflict failed"
            )
            return {"policy_result": result}

        draft = gov_gateway.create_policy_version(
            policy_key=policy_key,
            logical_policy=document,
            reason=state.get("policy_intent"),
        )
        if draft.get("status") != "DRAFT":
            raise RuntimeError("Backend create_policy_version did not return DRAFT")
        if draft.get("authority_changed") is not False:
            raise RuntimeError("Backend DRAFT unexpectedly changed authority")
        if draft.get("dispatched") is not False:
            raise RuntimeError("Backend DRAFT unexpectedly dispatched reconciliation")
        result["draft"] = draft
        return {"policy_result": result}

    graph = StateGraph(AgentState)
    graph.add_node("load_om_context", load_om_context)
    graph.add_node("tag_reasoning", tag_reasoning)
    graph.add_node("load_backend_policy_context", load_backend_policy_context)
    graph.add_node("policy_reasoning", policy_reasoning)
    graph.add_node("normalize_backend_policy", normalize_backend_policy)
    graph.add_node("check_policy_conflict", check_conflict)
    graph.add_node("preview_policy_change", preview_change)
    graph.add_node("optional_create_draft", optional_create_draft)

    graph.add_edge(START, "load_om_context")
    graph.add_conditional_edges(
        "load_om_context",
        route_intent,
        {"TAG": "tag_reasoning", "POLICY": "load_backend_policy_context"},
    )
    graph.add_edge("tag_reasoning", END)
    graph.add_edge("load_backend_policy_context", "policy_reasoning")
    graph.add_edge("policy_reasoning", "normalize_backend_policy")
    graph.add_edge("normalize_backend_policy", "check_policy_conflict")
    graph.add_edge("check_policy_conflict", "preview_policy_change")
    graph.add_edge("preview_policy_change", "optional_create_draft")
    graph.add_edge("optional_create_draft", END)
    return graph.compile()


def run_governance_graph(
    *,
    om_gateway: OpenMetadataGateway,
    gov_gateway: GovernanceGateway,
    tag_classifier: StructuredClassifier,
    policy_classifier: PolicyClassifier,
    request_type: str = "TAG",
    entity_type: str = "table",
    entity_fqn: str,
    allowed_tags: list[str] | None = None,
    include_lineage: bool = True,
    target_subjects: list[Subject] | None = None,
    policy_intent: str | None = None,
    policy_key: str | None = None,
    persist_draft: bool = False,
    environment: str = "local",
) -> tuple[TagReasoningResult | None, PolicyReasoningResult | None, dict[str, Any]]:
    graph = build_governance_graph(
        om_gateway=om_gateway,
        gov_gateway=gov_gateway,
        tag_classifier=tag_classifier,
        policy_classifier=policy_classifier,
    )
    result = graph.invoke(
        {
            "request_type": request_type,
            "entity_type": entity_type,
            "entity_fqn": entity_fqn,
            "allowed_tags": allowed_tags or [],
            "include_lineage": include_lineage,
            "target_subjects": (
                [s.model_dump(mode="json") for s in target_subjects]
                if target_subjects
                else None
            ),
            "policy_intent": policy_intent,
            "policy_key": policy_key,
            "persist_draft": persist_draft,
            "environment": environment,
        }
    )
    tag_result = (
        TagReasoningResult.model_validate(result["tag_result"])
        if result.get("tag_result")
        else None
    )
    policy_result = (
        PolicyReasoningResult.model_validate(result["policy_result"])
        if result.get("policy_result")
        else None
    )
    return tag_result, policy_result, dict(result.get("catalog_context", {}))
