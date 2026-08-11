.PHONY: test worker-classification

test:
	python -m pytest -q

worker-classification:
	python -m celery -A app.celery_app worker -Q ai.classification -c 1 --loglevel=info
