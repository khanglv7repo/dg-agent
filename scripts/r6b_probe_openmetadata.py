import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.gateways.openmetadata import OpenMetadataGateway


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.getenv("OPENMETADATA_AGENT_BOT_TOKEN", "")
    print("OPENMETADATA_AGENT_BOT_TOKEN set:", bool(token))
    if not token:
        raise SystemExit(
            "Missing OPENMETADATA_AGENT_BOT_TOKEN. Export it or put it in agent/.env."
        )
    gateway = OpenMetadataGateway(
        endpoint=os.getenv("OPENMETADATA_MCP_URL", "http://localhost:8585/mcp"),
        token=token,
    )
    try:
        contract = gateway.patch_entity_contract()
        print("PATCH_ENTITY:", json.dumps(contract, indent=2, default=str))
        print(
            "PATCH_ENTITY_SUPPORTED:",
            gateway._patch_entity_schema_supported(contract),
        )
        print("TAXONOMY_COUNT:", len(gateway.get_taxonomies()))
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
