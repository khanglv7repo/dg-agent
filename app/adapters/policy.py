"""Adapter from Agent logical policy proposal to frozen Backend R5 document."""
from __future__ import annotations

from typing import Iterable

from app.schemas import LogicalPolicyProposal, Subject


class PolicyAdapterError(ValueError):
    pass


def _subject_document(subject: Subject) -> dict[str, str]:
    return {
        "type": subject.subject_type,
        "name": subject.name.strip(),
    }


def to_backend_logical_policy(
    proposal: LogicalPolicyProposal,
    *,
    explicit_subjects: Iterable[Subject] | None = None,
    require_explicit_subjects: bool = False,
) -> dict:
    bound_subjects = list(explicit_subjects or [])
    if require_explicit_subjects and not bound_subjects:
        raise PolicyAdapterError(
            "persisted Backend DRAFT requires explicit caller target_subjects"
        )
    subjects = bound_subjects or list(proposal.subjects)
    if not subjects:
        raise PolicyAdapterError("policy must contain at least one subject")

    masks: dict[str, str] = {}
    for item in proposal.masks:
        if item.mask_type != "MASK":
            raise PolicyAdapterError(
                f"unsupported Backend mask intent {item.mask_type!r}; only 'MASK' is supported"
            )
        column = item.column.strip()
        if not column:
            raise PolicyAdapterError("mask column must not be empty")
        masks[column] = "MASK"

    row_filter = None
    if proposal.row_filter is not None:
        expression = (proposal.row_filter.expression or "").strip()
        if expression:
            row_filter = expression

    return {
        "subjects": [_subject_document(item) for item in subjects],
        "resource": proposal.resource.model_dump(mode="json", by_alias=True),
        "access": dict(proposal.access),
        "masks": masks,
        "row_filter": row_filter,
    }
