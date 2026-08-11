from app.celery_app import app
from app.tasks.classification import ai_classify_entity


def test_exact_backend_handoff_task_and_queue_contract() -> None:
    assert ai_classify_entity.name == "app.tasks.classification.ai_classify_entity"
    route = app.conf.task_routes["app.tasks.classification.ai_classify_entity"]
    assert route["queue"] == "ai.classification"
    assert app.conf.task_default_queue == "ai.classification"
