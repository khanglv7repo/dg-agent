from __future__ import annotations

from app.runner import GovernanceAgentRunner
from app.schemas import AgentRunRequest


def main() -> None:
    print("Starting Governance Agent standalone service...")
    print("Agent is connected to OpenMetadataGateway and Backend GovernanceGateway.")


if __name__ == "__main__":
    main()
