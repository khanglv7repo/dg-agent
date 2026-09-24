"""Chat entry node: lets Agent Chat UI's plain free-text chat input drive
the same 5 structured domains (TAG/POLICY/DQ/VERIFICATION/COPILOT) that a
direct structured `AgentRunRequest` already can.

Found live: Agent Chat UI posts `{"messages": [...]}`, not a structured
request -- the graph previously required `entity_fqn` unconditionally in
every domain's first node, so a plain "hi" crashed with `KeyError:
'entity_fqn'` instead of getting a reply. This node is the fix: it runs
FIRST (before the existing `route_from_start` routing), classifies the
latest human message via `OpenAIChatRouter` (see `app/classifier.py`), and
either:
- replies conversationally in `messages` and stops (a greeting, small
  talk, or a request missing required fields like entity_fqn) -- the
  fail-closed default, matching how a human colleague would respond to
  "hi" rather than crashing, per the user's own explicit expectation; or
- calls OpenMetadata's `search_metadata` (via `intent.search_query`, see
  below) when the user named a topic but not an exact FQN, or explicitly
  asked the agent to search/explore itself; or
- fills in `request_type`/`entity_fqn`/`copilot_question` from the
  classified intent and lets the graph continue into the existing routing
  exactly as if the caller had sent a structured request directly.

Only activates when the caller actually used the chat/messages path --
callers who already send a structured `request_type` (runner.py, the
Celery worker, every existing test) skip this node entirely, so none of
that existing behavior changes.

Design note (2026-09-24 rewrite): an earlier version of this file tried to
detect "the user wants me to search" and extract a search keyword using
hand-written regex/keyword-list matching (`_ENTITY_HINTS`, a fixed
English-only word list, a 6-message lookback window). Found live, with the
user's own test transcript: this broke as soon as the user said "tài
chính" instead of "financial", or when the topic was mentioned more than 6
messages back -- the agent then falsely claimed "I don't have a search
tool" even though search_metadata was working correctly and had already
been demonstrated working in the same session. Root cause was the general
one regex-based intent detection always has: it can't understand novel
phrasing or long-range context. Replaced with `ChatIntent.search_query`
(see schemas.py) -- the LLM (which already reads the whole conversation
history for `needs_clarification`) decides when to search and what to
search for; this code only decides WHETHER to act on that field, same
"code decides whether to call a tool, LLM decides what to look for -- never
gets a generic tool-choice loop" boundary as every other write path in
this codebase.
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol
import urllib.error
import urllib.request

from langchain_core.messages import AIMessage, BaseMessage

from app.schemas import ChatIntent


class ChatRouter(Protocol):
    model_name: str
    prompt_version: str

    def classify(
        self,
        *,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> ChatIntent: ...


def _latest_human_text(messages: list[BaseMessage]) -> str | None:
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None)
        if msg_type == "human":
            content = getattr(msg, "content", None)
            return content if isinstance(content, str) else str(content)
    return None


def _history_as_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for msg in messages[:-1]:  # exclude the latest message, passed separately
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        history.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return history


def _entity_fqn_was_user_supplied(entity_fqn: str | None, messages: list[BaseMessage]) -> bool:
    if not entity_fqn:
        return False
    needle = entity_fqn.casefold()
    for msg in messages:
        if getattr(msg, "type", None) != "human":
            continue
        content = getattr(msg, "content", "")
        if needle in (content if isinstance(content, str) else str(content)).casefold():
            return True
    return False


def _metadata_search_reply(keyword: str) -> str:
    """Calls OpenMetadata MCP's `search_metadata` tool for real (part of
    the 15-tool whitelist verified live in TASK-03's A1 POC) and formats
    the results as a chat reply. `keyword` comes only from
    `ChatIntent.search_query` -- an LLM-produced text field, never a
    tool-choice decision (the LLM cannot choose to call this itself)."""
    token = os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", "")
    if not token:
        return "OpenMetadata MCP search_metadata is unavailable: OPENMETADATA_AGENT_BOT_TOKEN is not set."

    om_mcp = os.getenv("OPENMETADATA_MCP_URL", "http://localhost:8585/mcp")
    try:
        data = _openmetadata_mcp_call_tool(om_mcp, "search_metadata", {"query": keyword})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        return f"OpenMetadata MCP search_metadata failed: HTTP {exc.code}: {body}"
    except Exception as exc:
        return f"OpenMetadata MCP search_metadata failed: {type(exc).__name__}: {str(exc)[:300]}"

    parsed = _parse_mcp_text_json(data)
    if not isinstance(parsed, dict):
        return f"OpenMetadata MCP search_metadata returned an unexpected response for query {keyword!r}."

    results = parsed.get("results") or []
    if not isinstance(results, list) or not results:
        return (
            f"OpenMetadata MCP search_metadata ran for query {keyword!r}, but returned no "
            "results. Try a different keyword, or give me the exact table FQN."
        )

    lines = [f"OpenMetadata MCP search_metadata('{keyword}') returned {parsed.get('totalFound', len(results))} results; showing first {len(results)}:"]
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        entity_type = item.get("entityType", "entity")
        fqn = item.get("fullyQualifiedName") or item.get("name") or "(missing FQN)"
        display = item.get("displayName") or item.get("name")
        suffix = f" ({display})" if display and display != fqn.rsplit(".", 1)[-1] else ""
        lines.append(f"- {entity_type}: {fqn}{suffix}")
    if parsed.get("hasMore"):
        lines.append("There are more matches; narrow by service/schema/table name if you want a shorter list.")
    lines.append("\nTell me the exact table FQN from this list and I'll check its tags/policy/DQ status.")
    return "\n".join(lines)


def _openmetadata_mcp_call_tool(endpoint: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    token = os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("OPENMETADATA_AGENT_BOT_TOKEN is not set")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    data, _ = _mcp_rpc(endpoint, payload, headers={"Authorization": f"Bearer {token}"})
    if "error" in data:
        raise RuntimeError(f"OpenMetadata MCP error: {data['error']}")
    result = data.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("OpenMetadata MCP returned no result object")
    return result


def _mcp_rpc(endpoint: str, payload: dict[str, Any], *, headers: dict[str, str]) -> tuple[dict[str, Any], str | None]:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **headers,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        session_id = response.headers.get("mcp-session-id")
        body = response.read().decode("utf-8", errors="replace")

    if body.startswith("event:"):
        data_lines = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")]
        body = "\n".join(data_lines)
    try:
        return json.loads(body), session_id
    except json.JSONDecodeError as exc:
        raise ValueError(f"non-JSON MCP response: {body[:300]}") from exc


def _parse_mcp_text_json(result: dict[str, Any]) -> Any:
    content = result.get("content")
    if not isinstance(content, list):
        return result
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return result


def build_chat_nodes(*, chat_router: ChatRouter) -> dict[str, Any]:
    """Returns the chat entry node, ready to register on a
    `StateGraph(AgentState)`."""

    def chat_entry(state: dict[str, Any]) -> dict[str, Any]:
        messages: list[BaseMessage] = state.get("messages") or []
        latest = _latest_human_text(messages)
        if not latest:
            # No human message to classify (e.g. a structured request that
            # also happened to route through here) -- nothing to do, let
            # existing routing proceed with whatever fields are already set.
            return {}

        history = _history_as_dicts(messages)
        intent = chat_router.classify(message=latest, conversation_history=history)

        if intent.search_query:
            return {
                "messages": [AIMessage(content=_metadata_search_reply(intent.search_query))],
                "request_type": "CHAT_REPLY_ONLY",
            }

        if intent.needs_clarification or not intent.request_type or not intent.entity_fqn:
            reply = intent.reply_message or (
                "Mình có thể giúp bạn classify tags, đề xuất policy, tạo DQ test, "
                "hoặc kiểm tra verification evidence cho một bảng cụ thể.\n\n"
                "Cho mình biết FQN đầy đủ, hoặc một từ khóa để mình tìm trong "
                "OpenMetadata (ví dụ: khách hàng, tài chính, giao dịch)."
            )
            return {
                "messages": [AIMessage(content=reply)],
                "request_type": "CHAT_REPLY_ONLY",
            }

        if not _entity_fqn_was_user_supplied(intent.entity_fqn, messages):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Mình sẽ không tự bịa FQN. Cho mình FQN chính xác, hoặc "
                            "nói mình tìm giúp theo tên/chủ đề."
                        )
                    )
                ],
                "request_type": "CHAT_REPLY_ONLY",
            }

        result: dict[str, Any] = {
            "request_type": intent.request_type,
            "entity_fqn": intent.entity_fqn,
            "entity_type": intent.entity_type,
        }
        if intent.request_type == "COPILOT":
            result["copilot_question"] = intent.copilot_question or latest
            result["copilot_want_recommendation"] = True
        return result

    return {"chat_entry": chat_entry}


__all__ = ["build_chat_nodes"]
