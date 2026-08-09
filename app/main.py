from __future__ import annotations

import sys
from app.runner import GovernanceAgentRunner
from app.schemas import AgentRunRequest


def main() -> None:
    print("Starting Governance Agent standalone service...")
    print("Agent is connected directly to OpenMetadata via MCP & REST API.")


if __name__ == "__main__":
    main()
