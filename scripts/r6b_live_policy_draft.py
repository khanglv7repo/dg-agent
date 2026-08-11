from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from app.clients.backend_mcp import BackendMCPError
from app.gateways.governance import GovernanceGateway

POLICY_KEY = "r6b-live-draft-policy-20260811"
BASELINE_POLICY_KEY = "r4-absolute-final-combined-mask-row-20260809-2330"

LOGICAL_POLICY = {
    "subjects": [{"type": "USER", "name": "alice"}],
    "resource": {
        "catalog": "financial",
        "schema": "crm",
        "table": "customers",
    },
    "access": {"select": "ALLOW"},
    "masks": {"phone": "MASK"},
    "row_filter": "customer_id <= 10",
}


def dump(label: str, value) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    gateway = GovernanceGateway()
    try:
        contract = gateway.validate_contract()
        print("BACKEND CONTRACT: PASS")
        print("SERVER:", contract["server"])
        print("TOOL COUNT:", len(contract["tools"]))

        baseline_before = gateway.get_policy(BASELINE_POLICY_KEY)
        dump("BASELINE ACTIVE BEFORE", baseline_before)

        mapping = gateway.resolve_resource_mapping(
            om_service_name="financial_postgres",
            environment="local",
        )
        dump("SERVICE MAPPING", mapping)

        if mapping.get("trino_catalog") != "financial":
            raise AssertionError(
                f"expected trino_catalog=financial, got {mapping.get('trino_catalog')!r}"
            )

        try:
            existing = gateway.get_policy(POLICY_KEY)
        except BackendMCPError as exc:
            if exc.code != "NOT_FOUND":
                raise
            existing = None

        if existing is not None:
            raise SystemExit(
                f"ABORT: isolated R6-B test policy already exists: {POLICY_KEY}"
            )

        conflict = gateway.check_policy_conflict(
            policy_key=POLICY_KEY,
            logical_policy=LOGICAL_POLICY,
        )
        dump("CONFLICT", conflict)

        preview = gateway.preview_policy_change(
            policy_key=POLICY_KEY,
            logical_policy=LOGICAL_POLICY,
        )
        dump("PREVIEW", preview)

        projection_types = {
            str(item.get("projection_type"))
            for item in preview.get("projections", [])
        }
        expected = {"ACCESS", "MASK", "ROW_FILTER"}
        if projection_types != expected:
            raise AssertionError(
                f"expected preview projections {sorted(expected)}, "
                f"got {sorted(projection_types)}"
            )

        draft = gateway.create_policy_version(
            policy_key=POLICY_KEY,
            logical_policy=LOGICAL_POLICY,
            reason="R6-B live Agent integration acceptance; DRAFT only",
        )
        dump("CREATED DRAFT", draft)

        if draft.get("status") != "DRAFT":
            raise AssertionError("create_policy_version did not return DRAFT")
        if draft.get("authority_changed") is not False:
            raise AssertionError("DRAFT unexpectedly changed authority")
        if draft.get("dispatched") is not False:
            raise AssertionError("DRAFT unexpectedly dispatched reconciliation")

        version = int(draft["version"])
        persisted = gateway.get_policy(POLICY_KEY, version=version)
        dump("PERSISTED DRAFT", persisted)

        versions = gateway.list_policy_versions(POLICY_KEY)
        dump("VERSION HISTORY", versions)

        baseline_after = gateway.get_policy(BASELINE_POLICY_KEY)
        dump("BASELINE ACTIVE AFTER", baseline_after)

        if baseline_after.get("version") != baseline_before.get("version"):
            raise AssertionError("baseline ACTIVE version changed")
        if baseline_after.get("checksum") != baseline_before.get("checksum"):
            raise AssertionError("baseline ACTIVE checksum changed")

        print("\nDG-R6B-LIVE-POLICY-DRAFT: PASS")
        print("Created isolated immutable DRAFT only.")
        print("No activation, rollback, mapping update, or Ranger sync was requested.")
        print("Baseline R4 ACTIVE remained unchanged.")
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
