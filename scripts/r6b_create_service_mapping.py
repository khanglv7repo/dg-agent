from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from app.clients.backend_mcp import BackendMCPError
from app.gateways.governance import GovernanceGateway

OM_SERVICE = "financial_postgres"
ENVIRONMENT = "local"
TRINO_CATALOG = "financial"
RANGER_SERVICE = "dev_trino"
RANGER_TAG_SERVICE = "dev_tag"

BASELINE_POLICY_KEY = "r4-absolute-final-combined-mask-row-20260809-2330"


def dump(label: str, value) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    gateway = GovernanceGateway()

    try:
        gateway.validate_contract()
        print("BACKEND CONTRACT: PASS")

        baseline_before = gateway.get_policy(BASELINE_POLICY_KEY)
        dump("BASELINE BEFORE", baseline_before)

        try:
            existing = gateway.resolve_resource_mapping(
                om_service_name=OM_SERVICE,
                environment=ENVIRONMENT,
            )
        except BackendMCPError as exc:
            if exc.code != "NOT_FOUND":
                raise
            existing = None

        dump("EXISTING MAPPING", existing)

        expected = {
            "om_service_name": OM_SERVICE,
            "trino_catalog": TRINO_CATALOG,
            "ranger_service_name": RANGER_SERVICE,
            "ranger_tag_service_name": RANGER_TAG_SERVICE,
            "environment": ENVIRONMENT,
        }

        if existing is not None:
            mismatches = {
                key: (existing.get(key), value)
                for key, value in expected.items()
                if existing.get(key) != value
            }
            if mismatches:
                raise SystemExit(
                    "ABORT: mapping already exists but differs from R6-B expected "
                    f"contract: {mismatches}"
                )
            print("Mapping already matches expected contract; no write needed.")
        else:
            try:
                gateway.update_service_mapping(
                    om_service_name=OM_SERVICE,
                    trino_catalog=TRINO_CATALOG,
                    ranger_service_name=RANGER_SERVICE,
                    ranger_tag_service_name=RANGER_TAG_SERVICE,
                    environment=ENVIRONMENT,
                    confirmed=False,
                    enabled=True,
                    reason="R6-B local lab service mapping acceptance",
                )
            except BackendMCPError as exc:
                if exc.code != "CONFIRMATION_REQUIRED":
                    raise
                print("PASS confirmed=false -> CONFIRMATION_REQUIRED")
            else:
                raise AssertionError(
                    "confirmed=false unexpectedly changed service mapping authority"
                )

            # Read must still fail after rejected unconfirmed write.
            try:
                unexpected = gateway.resolve_resource_mapping(
                    om_service_name=OM_SERVICE,
                    environment=ENVIRONMENT,
                )
            except BackendMCPError as exc:
                if exc.code != "NOT_FOUND":
                    raise
                print("PASS unconfirmed write left mapping absent")
            else:
                raise AssertionError(
                    f"confirmed=false unexpectedly created mapping: {unexpected}"
                )

            created = gateway.update_service_mapping(
                om_service_name=OM_SERVICE,
                trino_catalog=TRINO_CATALOG,
                ranger_service_name=RANGER_SERVICE,
                ranger_tag_service_name=RANGER_TAG_SERVICE,
                environment=ENVIRONMENT,
                confirmed=True,
                enabled=True,
                reason="R6-B local lab exact OM->Trino/Ranger mapping",
            )
            dump("CREATED MAPPING", created)

            if created.get("authority_changed") is not True:
                raise AssertionError("confirmed mapping write did not change authority")
            if created.get("ranger_mutation") is not False:
                raise AssertionError("service mapping must not directly mutate Ranger")
            if created.get("reconciliation_enqueued") is not False:
                raise AssertionError("service mapping must not enqueue Ranger reconcile")

        observed = gateway.resolve_resource_mapping(
            om_service_name=OM_SERVICE,
            environment=ENVIRONMENT,
        )
        dump("READ-BACK MAPPING", observed)

        for key, value in expected.items():
            if observed.get(key) != value:
                raise AssertionError(
                    f"mapping read-back mismatch for {key}: "
                    f"{observed.get(key)!r} != {value!r}"
                )
        if observed.get("enabled") is not True:
            raise AssertionError("mapping must be enabled")

        baseline_after = gateway.get_policy(BASELINE_POLICY_KEY)
        dump("BASELINE AFTER", baseline_after)

        if baseline_after.get("version") != baseline_before.get("version"):
            raise AssertionError("R4 baseline ACTIVE version changed")
        if baseline_after.get("checksum") != baseline_before.get("checksum"):
            raise AssertionError("R4 baseline checksum changed")

        print("\nDG-R6B-SERVICE-MAPPING: PASS")
        print("financial_postgres + local -> financial / dev_trino / dev_tag")
        print("Ranger mutation: false")
        print("Reconciliation enqueued: false")
        print("R4 baseline ACTIVE unchanged")
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
