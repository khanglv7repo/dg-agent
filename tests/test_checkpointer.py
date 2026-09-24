from __future__ import annotations

from unittest.mock import MagicMock, patch

from app import checkpointer


def test_checkpoint_dsn_uses_dedicated_agent_database(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://backend:secret@db/governance_db")
    monkeypatch.setenv("AGENT_CHECKPOINT_DB_HOST", "postgres")
    monkeypatch.setenv("AGENT_CHECKPOINT_DB_PORT", "5432")
    monkeypatch.setenv("AGENT_CHECKPOINT_DB", "agent_checkpoint_db")
    monkeypatch.setenv("AGENT_CHECKPOINT_DB_USER", "agent_checkpoint_db_user")
    monkeypatch.setenv("AGENT_CHECKPOINT_DB_PASSWORD", "agent-secret")

    dsn = checkpointer.checkpoint_connection_string()

    assert "agent_checkpoint_db_user:agent-secret@postgres:5432/agent_checkpoint_db" in dsn
    assert "governance_db" not in dsn
    assert "backend:secret" not in dsn


def test_checkpointer_defaults_off_without_password(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_CHECKPOINT_ENABLED", raising=False)
    monkeypatch.delenv("AGENT_CHECKPOINT_DB_PASSWORD", raising=False)

    assert checkpointer.checkpointer_enabled() is False


def test_explicit_checkpointer_kill_switch_wins(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CHECKPOINT_DB_PASSWORD", "configured")
    monkeypatch.setenv("AGENT_CHECKPOINT_ENABLED", "false")

    assert checkpointer.checkpointer_enabled() is False

    monkeypatch.setenv("AGENT_CHECKPOINT_ENABLED", "true")
    assert checkpointer.checkpointer_enabled() is True


def test_build_checkpointer_keeps_real_connection_and_runs_setup(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CHECKPOINT_DB_PASSWORD", "agent-secret")
    connection = MagicMock()
    saver = MagicMock()

    with patch.object(
        checkpointer.Connection,
        "connect",
        return_value=connection,
    ) as connect, patch.object(
        checkpointer,
        "PostgresSaver",
        return_value=saver,
    ) as saver_cls:
        result = checkpointer.build_checkpointer()

    assert result is saver
    connect.assert_called_once()
    kwargs = connect.call_args.kwargs
    assert kwargs["autocommit"] is True
    assert kwargs["prepare_threshold"] == 0
    saver_cls.assert_called_once_with(connection)
    saver.setup.assert_called_once_with()
