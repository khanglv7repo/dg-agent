.PHONY: test worker-classification probe-llm langgraph-dev

test:
	python -m pytest -q

worker-classification:
	python -m celery -A app.celery_app worker -Q ai.classification -c 1 --loglevel=info

probe-llm:
	python scripts/r6b_probe_llm_structured.py

# LangGraph API server (dev, in-memory) -- Studio UI + HTTP access to the
# same graph.py used by GovernanceAgentRunner/the Celery worker. Requires
# `pip install -e '.[server]'`. See langgraph.json / app/langgraph_entry.py.
langgraph-dev:
	langgraph dev --no-browser
