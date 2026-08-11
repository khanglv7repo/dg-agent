from app.clients.backend_mcp import BackendMCPClient, EXPECTED_BACKEND_TOOLS


def main() -> None:
    client = BackendMCPClient()
    probe = client.validate_frozen_contract()
    print("SERVER:", probe["server"])
    print("TOOLS:", [item["name"] for item in probe["tools"]])
    health = client.call_tool("inspect_ranger_state", {"kind": "health"})
    print("RANGER HEALTH:", health)
    trino = client.call_tool(
        "query_trino_readonly",
        {"sql": "SELECT count(*) AS n FROM financial.crm.customers"},
    )
    print("TRINO:", trino)


if __name__ == "__main__":
    main()
