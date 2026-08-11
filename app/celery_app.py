"""Celery consumer configuration for the external Agent classification worker."""
from __future__ import annotations

import os
from pathlib import Path

from celery import Celery
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")

app = Celery("governance-agent", broker=BROKER_URL, backend=BROKER_URL)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_default_queue="ai.classification",
    task_routes={
        "app.tasks.classification.ai_classify_entity": {
            "queue": "ai.classification",
        }
    },
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
app.conf.imports = ("app.tasks.classification",)
