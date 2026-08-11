# R6-B Real LLM Overlay

Baseline repository: `khanglv7repo/dg-agent`

Required baseline commit:

`46cef8b7eccd4429654016c8885347d0b3f7821c` — `R6-B PRE-LLM CHECKPOINT`

This overlay is source-based. It contains complete replacement/new source files,
not search/replace mutations.

## Scope

- Configure `ChatOpenAI` for OpenAI-compatible Responses API when enabled.
- Explicit structured-output method support (`json_schema` verified live).
- One shared `LLMRuntimeConfig` for both synchronous runner and Celery worker.
- Celery `ai.classification` worker therefore uses the same model/base URL/API mode
  as the normal Agent runner.
- Add a live probe that invokes the production `TagReasoningResult` classifier.
- Preserve the existing fail-safe completion boundary: no authoritative OM write
  while Backend completion callback is absent.

## Verified external contract before this overlay

- Model: `DeepSeek-V4-Flash`
- Base URL: `https://aiportalapi.stu-platform.live/jpe/v2`
- Responses endpoint: `<base>/responses`
- Raw JSON Schema: PASS
- LangChain `ChatOpenAI(use_responses_api=True)`: PASS
- LangChain `with_structured_output(..., method="json_schema")`: PASS
- Pydantic structured result: PASS

## Files

- `app/classifier.py`
- `app/llm_runtime.py`
- `app/runner.py`
- `app/tasks/classification.py`
- `.env.example`
- `Makefile`
- `tests/test_llm_runtime.py`
- `scripts/r6b_probe_llm_structured.py`

## Deliberately not included

Backend generation-fenced completion callback. That is a separate Backend
authority change and must be implemented/tested in `dg-backend` before full
`DG-R6B-CLOSED` can be claimed.
