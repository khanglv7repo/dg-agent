from __future__ import annotations

from pathlib import Path

# TASK-09/TASK-10: app/checkpointer.py is a deliberate, narrow exception to
# the "no direct Postgres access" rule below. It connects to the Agent's OWN
# dedicated checkpoint database (agent_checkpoint_db, per docs/DG_FINAL_SPEC.md
# section 10's "không chung" requirement) -- never Backend's governance_db,
# never through Backend's own repositories/models. This is a different
# database, a different credential, and a different purpose (LangGraph's own
# checkpoint persistence, not governance business data) from what this test
# guards against (the Agent reaching into Backend's authoritative store).
_ALLOWED_PSYCOPG_FILES = frozenset({"checkpointer.py"})


def _production_files():
    root = Path(__file__).parents[1] / "app"
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def _production_text(*, exclude: frozenset[str] = frozenset()) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in _production_files()
        if path.name not in exclude
    )


def test_agent_production_has_no_direct_governance_backends() -> None:
    production = _production_text(exclude=_ALLOWED_PSYCOPG_FILES)
    forbidden_import_fragments = (
        "app.repositories",
        "sqlalchemy",
        "psycopg",
        "ranger_client",
        "trino.dbapi",
        "trino.auth",
    )
    forbidden_credentials = (
        "RANGER_PASSWORD",
        "RANGER_USERNAME",
        "GOVERNANCE_DATABASE_URL",
        "TRINO_PASSWORD",
        "TRINO_ADMIN",
    )

    for fragment in forbidden_import_fragments:
        assert fragment not in production, fragment
    for secret_name in forbidden_credentials:
        assert secret_name not in production, secret_name


def test_checkpointer_psycopg_usage_is_scoped_to_its_own_dedicated_database() -> None:
    """The one allowed psycopg import must only ever reference the Agent's
    own AGENT_CHECKPOINT_DB_* env vars for its connection string -- never
    Backend's DATABASE_URL or GOVERNANCE_DATABASE_URL, which would defeat
    the point of this exception. (`governance_db` itself may appear in
    comments/docstrings explaining what this module deliberately does NOT
    connect to -- only the actual credential-source env var names matter
    here, not every substring.)"""
    checkpointer_files = [
        path for path in _production_files() if path.name in _ALLOWED_PSYCOPG_FILES
    ]
    assert checkpointer_files, "expected app/checkpointer.py to exist"
    text = "\n".join(p.read_text(encoding="utf-8") for p in checkpointer_files)
    assert "psycopg" in text
    # Only inspect actual os.getenv(...) calls -- the module docstring
    # legitimately mentions Backend's DATABASE_URL/governance_db by name to
    # explain what this file deliberately does NOT read from.
    getenv_lines = [line for line in text.splitlines() if "os.getenv(" in line]
    assert any("AGENT_CHECKPOINT_DB" in line for line in getenv_lines)
    forbidden_backend_terms = ("DATABASE_URL", "GOVERNANCE_DATABASE_URL")
    for line in getenv_lines:
        for term in forbidden_backend_terms:
            assert term not in line, f"{term} in getenv call: {line.strip()}"


def test_backend_mcp_endpoint_is_frozen_r5_default() -> None:
    from app.clients.backend_mcp import BackendMCPClient
    from app.gateways.governance import GovernanceGateway

    assert BackendMCPClient().endpoint == "http://127.0.0.1:8001/mcp"
    assert GovernanceGateway().endpoint == "http://127.0.0.1:8001/mcp"
