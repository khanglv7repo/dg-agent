# TASK-09: LangGraph dev-mode API server, self-hosted, free (no LangSmith
# account/license needed -- `langgraph build`'s own image requires
# LANGSMITH_API_KEY/LANGGRAPH_CLOUD_LICENSE_KEY, confirmed live: it refused
# to start without one). `langgraph dev` has no such requirement, at the
# cost of being explicitly the in-memory/dev-mode server (its own startup
# banner says so) -- the graph's own checkpointer (app/checkpointer.py,
# wired in app/langgraph_entry.py:graph()) still persists real graph state
# to the dedicated agent_checkpoint_db Postgres database regardless; only
# the LangGraph API server's own run/thread bookkeeping layer is in-memory.
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md langgraph.json ./

# psycopg's pure-Python fallback needs libpq, which python:3.13-slim doesn't
# ship -- found live ("no pq wrapper available ... libpq library not found").
# Installing the libpq5 system package (not switching to psycopg[binary],
# which triggered a slow/stuck pip dependency resolve against the already
# pinned plain `psycopg` from langgraph-checkpoint-postgres) is the minimal
# fix -- psycopg's pure-Python implementation just needs libpq present.
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

RUN python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); deps=p['project']['dependencies'] + p['project']['optional-dependencies']['server']; open('/tmp/requirements.txt','w').write('\n'.join(deps) + '\n')" \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY app ./app

# Keep dependency installation cached across normal source edits. After `app/`
# changes, only this lightweight editable install should rerun.
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 2024

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024", "--no-browser"]
