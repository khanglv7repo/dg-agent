from __future__ import annotations

from pathlib import Path


def _production_text() -> str:
    root = Path(__file__).parents[1] / "app"
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def test_agent_production_has_no_direct_governance_backends() -> None:
    production = _production_text()
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


def test_backend_mcp_endpoint_is_frozen_r5_default() -> None:
    from app.clients.backend_mcp import BackendMCPClient
    from app.gateways.governance import GovernanceGateway

    assert BackendMCPClient().endpoint == "http://127.0.0.1:8001/mcp"
    assert GovernanceGateway().endpoint == "http://127.0.0.1:8001/mcp"
