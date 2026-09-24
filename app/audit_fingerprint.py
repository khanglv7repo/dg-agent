"""I9 audit-fingerprint helpers.

Per `docs/13_IMPLEMENTATION_SPEC.md` section 7, every AI-originated write must
carry a `model_prompt_fingerprint` distinct from a `validated_at` timestamp
that proves Validate-before-Write ordering (Hard Invariant #13:
`validated_at != created_at`). This module is the single place that computes
those values so every write boundary uses the same derivation.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any


def env_bool(name: str, default: bool = True) -> bool:
    """I11 kill-switch reader. Defaults to True (feature ON) unless the env var
    is explicitly set to a falsy value -- matches AgentRunRequest's own
    `agent_write_to_om_enabled=True` / `auto_apply_tag_enabled=True` defaults.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def model_fingerprint(*, model_name: str, prompt_version: str) -> str:
    """sha256[:12] of `<model_name>:<prompt_version>`, per AIWriteAuditRef docstring."""
    raw = f"{model_name}:{prompt_version}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def input_fingerprint(payload: Any) -> str:
    """sha256[:12] of the serialised LLM input context (order-stable JSON)."""
    raw = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, second precision, explicit `Z` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["env_bool", "model_fingerprint", "input_fingerprint", "utc_now_iso"]
