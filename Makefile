.PHONY: test probe-llm

test:
	python -m pytest -q

probe-llm:
	python scripts/r6b_probe_llm_structured.py
