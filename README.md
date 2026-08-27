# Resume Tailor

An asynchronous API that tailors a LaTeX résumé to a specific job description
using a multi-agent LLM pipeline. It returns the rewritten LaTeX **plus a
structured, change-by-change audit trail** — every edit is labelled with the
section it touched, what changed, and why.

The design goal is not "call an LLM." It is to run an LLM safely inside a
production-shaped backend: authenticated, rate-limited, observable, processed
off the request thread, and deployed to Kubernetes.

---

## What it does

`POST` a résumé and a job description → get back a job ID immediately
(`202 Accepted`) → poll for the result. Behind that simple contract, a four-stage
agent pipeline analyses the gap, rewrites the résumé, validates the rewrite for
honesty and structural integrity, and scores the result.

```json
// GET /v1/jobs/{id}  →  when complete
{
  "status": "complete",
  "result": {
    "modified_latex": "\\documentclass{article}...",
    "changes": [
      { "section": "experience", "action": "added",
        "content": "Designed asynchronous services using Celery and Redis.",
        "reason": "JD requires message-queue experience not shown in the original." }
    ],
    "ats_keywords_added": ["Celery", "Redis", "Docker", "Kubernetes", "FastAPI"],
    "ats_keywords_missing": []
  }
}
```

---

## Architecture

```
              HTTP                     enqueue job                run pipeline
  Client ───────────────▶  FastAPI  ───────────────▶  Redis  ───────────────▶  Celery worker
  (Gradio /              (auth, validation,        (broker +               (LangGraph 4-agent
   curl / any)            rate-limit, 202)          result store)           pipeline → LLM)
                              ▲                                                     │
                              └──────────────  poll GET /v1/jobs/{id}  ────────────┘
```

The API never blocks on the LLM. A request is authenticated, validated, and
handed to Redis in milliseconds; the 10–60 s of model work happens in a separate
worker process. This is what keeps the API responsive under load and lets the
workers scale independently of the web tier.

| Component | Responsibility |
|-----------|----------------|
| **FastAPI** | API-key auth, request validation, sliding-window rate limiting, async job dispatch, health/readiness probes, Prometheus metrics |
| **Redis** | Celery broker (job queue) and result backend |
| **Celery worker** | Runs the LangGraph pipeline off the request thread |
| **LangGraph pipeline** | The four-agent tailoring graph (below) |
| **Gradio** | Optional thin browser client — pure HTTP, no business logic |

---

## The multi-agent pipeline

The heart of the project. A LangGraph `StateGraph` over a typed state object,
wiring four single-responsibility agents with a **self-correcting retry loop**:

```
                    ┌──────────────── retry with feedback ─────────────┐
                    ▼                                                   │
  analyzer ───▶ tailor ───▶ validator ──▶ (route)
                                            │  passed / max-retries ──▶ scorer ──▶ done
                                            │  error ──────────────────────────▶ done
```

- **analyzer** — gap analysis between résumé and JD (missing keywords, relevant
  sections, seniority and tone signals, prioritised changes). Produces a brief;
  makes no edits.
- **tailor** — rewrites the LaTeX from that brief under hard constraints: never
  fabricate, preserve LaTeX structure, return the complete document, and emit a
  structured list of every change.
- **validator** — a strict quality gate that checks for fabrication, LaTeX
  structural integrity, and change-list accuracy. On failure it routes back to
  the tailor **with specific feedback injected into the next prompt**, so the
  model fixes the named issues instead of repeating them.
- **scorer** — quantifies the result (ATS score, delta vs. the original,
  confidence, remaining gaps).

Design decisions worth calling out:

- **Bounded retries.** The retry count lives in graph state; a conditional edge
  caps total attempts so a systematic failure can't loop (and bill) forever.
- **Graceful degradation, two ways.** A validator exception is caught and turned
  into a recoverable "retry" rather than crashing the graph; the scorer is last
  and never fails the job — on error it returns a low-confidence default so the
  user still gets their tailored résumé.
- **Error vs. quality failure.** An upstream node *exception* short-circuits to
  the end (retrying the tailor with no analysis would only produce worse output),
  which is treated differently from a recoverable *validation* rejection.
- **Backend-agnostic nodes.** Nodes depend only on the LangChain `BaseChatModel`
  interface, so the LLM backend is swappable by config (see below) with zero
  changes to pipeline logic.

---

## Tech stack

**Python 3.11 · FastAPI · Celery · Redis · LangGraph · LangChain · Pydantic v2 ·
Docker · Kubernetes · Ollama / Anthropic Claude · Prometheus · structlog · pytest**

---

## Running it

Two supported paths. Both need [Docker](https://docs.docker.com/get-docker/) and
an LLM backend. By default the system uses a **self-hosted [Ollama](https://ollama.com)
model — no API key, no paid service.** Set `LLM_PROVIDER=anthropic` to use Claude.

```bash
# one-time: pull the default local model
ollama pull qwen2.5-coder:7b
```

### Option A — Docker Compose (fastest)

```bash
cp .env.example .env          # defaults work as-is for local Ollama
docker compose up --build
```

- API docs:  http://localhost:8000/docs
- Gradio UI: http://localhost:7860
- Health:    http://localhost:8000/health

### Option B — Kubernetes (minikube)

The full stack also runs on Kubernetes. Manifests live in [`infra/k8s/`](infra/k8s):
a Redis StatefulSet, the API Deployment (single replica, backed by a
PersistentVolumeClaim for its SQLite key store), a stateless worker Deployment, a
HorizontalPodAutoscaler on the worker, and an Ingress.

```bash
minikube start --driver=docker --cpus=4 --memory=5120
minikube addons enable metrics-server ingress

# build images straight into the cluster's docker daemon
eval $(minikube docker-env)
docker build -f Dockerfile.api    -t resume-tailor-api:latest    .
docker build -f Dockerfile.worker -t resume-tailor-worker:latest .

kubectl apply -f infra/k8s/

kubectl -n resume-tailor get pods
```

```
NAME                     READY   STATUS    RESTARTS   AGE
api-d9498bf6f-qvvkx      1/1     Running   0          89m
redis-0                  1/1     Running   0          116m
worker-6df768d8c8-wq9sh  1/1     Running   0          110m
```

The worker pods reach the host's Ollama server via `host.minikube.internal`
(run Ollama with `OLLAMA_HOST=0.0.0.0`). Issue an API key and try it:

```bash
KEY=$(kubectl -n resume-tailor exec deploy/api -- \
  python -c "import asyncio; from app.services import auth; print(asyncio.run(auth.create_api_key()))")

kubectl -n resume-tailor port-forward svc/api 8000:8000 &
curl -X POST http://localhost:8000/v1/jobs \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"resume_latex": "...", "job_description": "..."}'
```

---

## Testing

```bash
uv sync --extra dev
uv run pytest        # 79 tests
uv run ruff check .
```

The suite covers the Pydantic contracts and validators, every graph node and
edge, the conditional retry routing (verifying the tailor runs exactly the
capped number of times), the Celery worker wiring, and the job endpoints — with
the LLM and broker mocked so tests are fast and deterministic.

---

## Project layout

```
app/
├── main.py                 FastAPI app: lifespan, CORS, metrics, router
├── config.py               all settings from environment (12-factor)
├── dependencies.py         auth + rate-limit dependency chain
├── api/v1/endpoints/       health probes, job create/status
├── models/                 request/response Pydantic contracts
├── services/               API-key auth (salted SHA-256), rate limiter
└── worker/
    ├── celery_app.py       Celery configuration
    ├── tasks.py            the async task entry point
    └── llm/
        ├── provider.py     LLM backend factory (ollama | anthropic)
        ├── client.py       builds + runs the graph
        └── graph/          state, nodes, and edges of the pipeline
infra/k8s/                  Kubernetes manifests
gradio_client/              optional browser UI (thin HTTP client)
tests/                      unit + integration tests
```

---

## Configuration

Everything is driven by environment variables (see [`.env.example`](.env.example)).
Key settings:

| Variable | Default | Notes |
|----------|---------|-------|
| `LLM_PROVIDER` | `ollama` | `ollama` (self-hosted, free) or `anthropic` |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | any local Ollama model |
| `ANTHROPIC_API_KEY` | – | required only when `LLM_PROVIDER=anthropic` |
| `REDIS_URL` | `redis://localhost:6379/0` | broker + result backend |
| `RATE_LIMIT_PER_MINUTE` | `60` | per-key sliding window |
