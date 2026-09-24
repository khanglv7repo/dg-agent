"""LangGraph checkpoint persistence for the Agent.

Per `docs/DG_FINAL_SPEC.md` section 10, the Agent's checkpointer must use a
dedicated Postgres instance/database, explicitly NOT shared with Backend's
`governance_db` ("không chung"). `planning/03-CODE-EDIT-MANIFEST.md`'s
"Checkpoint persistence" row was BLOCKED on this database not existing;
TASK-10 provisioned `agent_checkpoint_db` (idempotent creation in
`infrastructure/docker/postgres-init/01-init.sh`), which unblocks this row.

This module owns the connection string construction and a single shared
`PostgresSaver` setup helper -- callers (the graph builder, the Celery
worker, the runner) get one checkpointer instance per process, not one per
request, since `PostgresSaver` manages its own connection pool internally.
"""
from __future__ import annotations

import os

from psycopg import Connection
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres import PostgresSaver


def checkpoint_connection_string() -> str:
    """Builds the Agent checkpointer's own Postgres DSN from env vars.

    Deliberately separate from Backend's DATABASE_URL / governance_db --
    this database is Agent-owned only, per the "không chung" requirement.
    """
    host = os.getenv("AGENT_CHECKPOINT_DB_HOST", "localhost")
    port = os.getenv("AGENT_CHECKPOINT_DB_PORT", "5432")
    db = os.getenv("AGENT_CHECKPOINT_DB", "agent_checkpoint_db")
    user = os.getenv("AGENT_CHECKPOINT_DB_USER", "agent_checkpoint_db_user")
    password = os.getenv("AGENT_CHECKPOINT_DB_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode=disable"


def checkpointer_enabled() -> bool:
    """I11-style kill switch: checkpointing can be disabled independently of
    every other Agent write path. Defaults to True once the DSN is
    configured; explicit AGENT_CHECKPOINT_ENABLED=false always wins."""
    raw = os.getenv("AGENT_CHECKPOINT_ENABLED")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(os.getenv("AGENT_CHECKPOINT_DB_PASSWORD"))


def build_checkpointer() -> PostgresSaver:
    """Opens a PostgresSaver against the dedicated Agent checkpoint DB and
    ensures its schema exists (idempotent -- safe to call on every process
    start, mirrors the same idempotent-setup pattern already used by
    infrastructure/docker/postgres-init/01-init.sh and
    infrastructure/scripts/ranger/bootstrap.py elsewhere in this repo).

    Deliberately does NOT use `PostgresSaver.from_conn_string(...)`'s
    `@contextmanager` wrapper here -- calling `.__enter__()` on that
    generator-based context manager without keeping the generator object
    alive lets it get garbage-collected, which closes the underlying
    connection out from under the saver (observed live: `psycopg.
    OperationalError: the connection is closed` on the very next graph
    invocation after construction). Opening the `psycopg.Connection`
    directly and holding it as a real object avoids that -- the connection
    stays open for the lifetime of this process, matching how every other
    long-lived client in this codebase (BackendMCPClient, OpenMetadataGateway)
    is constructed once per process, not per request.
    """
    conn_string = checkpoint_connection_string()
    conn = Connection.connect(
        conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
    )
    saver = PostgresSaver(conn)
    saver.setup()
    return saver


__all__ = ["checkpoint_connection_string", "checkpointer_enabled", "build_checkpointer"]
