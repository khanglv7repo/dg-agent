.PHONY: test worker-classification probe-llm

test:
	python -m pytest -q

worker-classification:
	python -m celery -A app.celery_app worker -Q ai.classification -c 1 --loglevel=info

probe-llm:
	python scripts/r6b_probe_llm_structured.py
