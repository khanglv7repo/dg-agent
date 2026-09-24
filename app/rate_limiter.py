"""I10 rate limiting for Agent write-boundary calls.

Per `docs/13_IMPLEMENTATION_SPEC.md` section 8: "Agent Gateway / Backend MCP
contract rate limiting on repeated create_test_case-class calls." This is a
bounded, per-key sliding-window counter backed by the same Redis instance
already used for Celery (`CELERY_BROKER_URL`/`REDIS_URL`) -- no new
infrastructure, matches the existing architecture rather than adding a
parallel rate-limit store.

Fails OPEN, not closed, when Redis is unreachable: a transient broker outage
must not silently block every write forever. This mirrors the same
fail-safe-vs-fail-open judgment call already made elsewhere in this codebase
(e.g. `openmetadata_context.py`'s fallback chain never treats "can't reach
the primary" as "deny the read") -- rate limiting is a defense-in-depth
guard against runaway writes, not the sole enforcement mechanism (Backend's
own `testcase_registry` reservation and OM's own duplicate-name rejection
are the real, unconditional safety nets per B3).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

try:
    import redis
except ImportError:
    redis = None


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    window_seconds: int
    current_count: int


class RateLimiter:
    """Fixed-window counter per key, e.g. `dq_write:<entity_fqn>`."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        limit: int = 10,
        window_seconds: int = 60,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._client = None
        if redis is not None:
            url = redis_url or os.getenv(
                "CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
            )
            try:
                self._client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
            except Exception:
                self._client = None

    def check_and_increment(self, key: str) -> RateLimitResult:
        """Atomically increments the counter for `key` and reports whether
        this call is within the allowed rate. Uses a fixed window keyed by
        the current window boundary (simpler and cheaper than a true sliding
        log for this use case -- I10's own scope is "repeated calls", not
        precise per-second fairness)."""
        if self._client is None:
            # Redis unavailable or the `redis` package is missing -- fail
            # open per this module's own documented policy above.
            return RateLimitResult(
                allowed=True, limit=self.limit, window_seconds=self.window_seconds, current_count=0
            )

        window_id = int(time.time()) // self.window_seconds
        redis_key = f"dg:agent:ratelimit:{key}:{window_id}"
        try:
            pipe = self._client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, self.window_seconds * 2)
            count, _ = pipe.execute()
        except Exception:
            # Redis reachable-but-erroring mid-call -- same fail-open policy.
            return RateLimitResult(
                allowed=True, limit=self.limit, window_seconds=self.window_seconds, current_count=0
            )

        return RateLimitResult(
            allowed=count <= self.limit,
            limit=self.limit,
            window_seconds=self.window_seconds,
            current_count=count,
        )


def dq_write_rate_limiter() -> RateLimiter:
    """The one rate limiter instance for DQ TestCase creation, per I10's
    explicit "create_test_case-class calls" scope. Limits are env-tunable;
    defaults are deliberately generous (10/60s per entity) -- this guards
    against a runaway loop, not normal usage."""
    limit = int(os.getenv("DQ_WRITE_RATE_LIMIT", "10"))
    window = int(os.getenv("DQ_WRITE_RATE_LIMIT_WINDOW_SECONDS", "60"))
    return RateLimiter(limit=limit, window_seconds=window)


__all__ = ["RateLimiter", "RateLimitResult", "dq_write_rate_limiter"]
