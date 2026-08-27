from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1.router import router
from app.config import settings
from app.services import auth


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _configure_logging()
    log = structlog.get_logger(__name__)

    await auth.initialize_db()
    log.info("app.started", environment=settings.environment, version="1.0.0")

    yield

    log.info("app.shutdown")


app = FastAPI(
    title="Resume Tailor API",
    version="1.0.0",
    description="""
Tailors a LaTeX resume to a specific job description using LLM-based analysis.
Accepts resume LaTeX source and a job description, returns modified LaTeX with
a structured explanation of every change made.

## Authentication
All endpoints except `/health` require an `X-API-Key` header.

## Async Processing
Job submission returns immediately with a `job_id`.
Poll `GET /v1/jobs/{job_id}` for results.
""",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS: allow all origins in development so the Gradio client and local
# browser testing work without configuration. In production this MUST be
# restricted to the specific origin(s) that serve your frontend (e.g.
# ["https://yourdomain.com"]). CORS exists to prevent a malicious website
# from silently making authenticated requests to this API using a visitor's
# browser cookies or stored credentials. Allowing all origins in production
# defeats that protection entirely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation. The /metrics endpoint exposes counters,
# histograms, and gauges for every route (request counts, latency
# percentiles, status code distributions) in the Prometheus text format,
# ready for any Prometheus-compatible scraper to collect.
Instrumentator().instrument(app).expose(app)

app.include_router(router)
