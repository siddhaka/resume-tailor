# Resume Tailor

A production-grade API that tailors a LaTeX resume to a specific 
job description using LLM-based analysis. Returns modified LaTeX 
source with a structured diff of every change made and why.

## Architecture


Gradio -> FastAPI -> Redis -> Celery Worker -> LLM 

- **FastAPI** — authentication, validation, async job dispatch
- **Redis** — message broker (job queue) and result backend
- **Celery** — distributed task worker for LLM processing
- **Gradio** — browser-based demo client

## Stack

- Runtime: Python 3.11, FastAPI, Celery, Redis
- Infrastructure: Kubernetes (k3s), Docker, GitHub Actions
- Security: HashiCorp Vault, Snyk, API key auth with rate limiting
- Observability: Prometheus, Grafana, structured logging

## Getting started

\```bashuv sync --extra dev
cp .env.example .env
# fill in your values

docker compose up
\```

API docs available at http://localhost:8000/docs