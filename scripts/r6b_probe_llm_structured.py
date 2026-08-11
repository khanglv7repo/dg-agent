"""Live R6-B probe using the exact production TAG classifier contract."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.llm_runtime import LLMRuntimeConfig

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

config = LLMRuntimeConfig.from_env()
if not config.api_key:
    raise SystemExit("LLM_API_KEY is not set in the Agent environment")

print("MODEL:", config.model)
print("BASE_URL:", config.base_url)
print("RESPONSES_API:", config.use_responses_api)
print("STRUCTURED_METHOD:", config.structured_output_method)
print("API_KEY_SET:", bool(config.api_key))

classifier = config.tag_classifier()
result = classifier.classify(
    catalog_context={
        "entity_type": "table",
        "entity_fqn": "r6b.probe.customers",
        "name": "customers",
        "columns": [
            {
                "name": "phone_number",
                "fullyQualifiedName": "r6b.probe.customers.phone_number",
                "dataType": "VARCHAR",
                "description": "Customer phone number",
            }
        ],
    },
    allowed_tags=["PII.Phone"],
)

payload = result.model_dump(mode="json")
print("RESULT:", payload)

invalid = [
    item["tag"]
    for item in payload.get("recommendations", [])
    if item.get("tag") != "PII.Phone"
]
if invalid:
    raise SystemExit(f"FAIL: model invented non-allowed tags: {invalid}")

print("R6B_REAL_LLM_STRUCTURED: PASS")
