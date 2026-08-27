from __future__ import annotations

# Set required environment variables BEFORE importing any app module.
# app/config.py runs `settings = get_settings()` at import time, which reads
# env vars immediately. If these aren't set first, Settings() raises
# ValidationError for missing required fields and the entire test suite fails
# to collect.
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
os.environ.setdefault("API_KEY_SALT", "test-salt-value-for-testing")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing")

from unittest.mock import MagicMock, patch

import httpx
import pytest
import redis.asyncio as aioredis

from app.config import Settings
from app.config import settings as _global_settings
from app.models.responses import (
    AnalysisResult,
    Change,
    ScoringResult,
    TailorResult,
    ValidationResult,
)
from app.services import auth

# ---------------------------------------------------------------------------
# Shared test data constants
# ---------------------------------------------------------------------------

VALID_RESUME_LATEX = r"""\documentclass[11pt]{article}
\begin{document}
\section*{Experience}
Senior Software Engineer at Acme Corp, 2020--2024.
Built and maintained Python-based REST APIs serving 10M+ requests per day.
Led migration of monolithic application to microservices architecture.
\section*{Skills}
Python, FastAPI, PostgreSQL, Docker, Redis, Git, Linux, AWS
\section*{Education}
B.S. Computer Science, State University, 2018
\end{document}"""

VALID_RESUME_LATEX_MODIFIED = r"""\documentclass[11pt]{article}
\begin{document}
\section*{Experience}
Senior Software Engineer at Acme Corp, 2020--2024.
Built and maintained Python-based REST APIs serving 10M+ requests per day.
Led migration of monolithic application to Kubernetes-based microservices
using distributed systems principles and gRPC service communication.
\section*{Skills}
Python, FastAPI, PostgreSQL, Docker, Redis, Kubernetes, gRPC, distributed systems
\section*{Education}
B.S. Computer Science, State University, 2018
\end{document}"""

VALID_JD = (
    "We are seeking a Senior Backend Engineer to join our distributed systems team. "
    "You will design and implement gRPC services and Kubernetes-based microservices. "
    "Required: Python, distributed systems, Kubernetes, Docker, PostgreSQL. "
    "Preferred experience with gRPC, service mesh, observability, and cloud infrastructure."
)


# ---------------------------------------------------------------------------
# Configuration fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    """Fresh Settings instance with test-safe values.

    Separate from the global settings singleton. Use monkeypatch on
    _global_settings when you need to change the live settings the app uses.
    """
    return Settings(
        anthropic_api_key="sk-ant-test-fake-key",
        api_key_salt="test-salt-value-for-testing",
        secret_key="test-secret-key-for-testing",
        redis_url="redis://localhost:6379/1",
        database_url="sqlite:///./test_resume_tailor.db",
    )


# ---------------------------------------------------------------------------
# Database isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    """Patch global settings to point to a per-test temp SQLite file.

    Both test_api_key and async_client depend on this fixture. Because pytest
    deduplicates fixture instances within a single test, they share the same
    tmp_path, so keys created by test_api_key are visible to the auth
    middleware in the running FastAPI app.
    """
    db_path = str(tmp_path / "test.db")
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setattr(_global_settings, "database_url", db_url)
    await auth.initialize_db()
    return db_url


# ---------------------------------------------------------------------------
# Redis fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def redis_client(settings):
    """Real async Redis client on DB 1 (separate from dev DB 0).

    We use real Redis rather than a mock because the sliding-window rate
    limiter relies on sorted-set semantics (ZREMRANGEBYSCORE, ZCARD, ZADD)
    that must behave correctly. A mock that records calls cannot reproduce
    the atomic ordering semantics the rate limiter depends on.
    """
    client: aioredis.Redis = aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=True
    )
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis not available — skipping Redis-dependent test")

    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


# ---------------------------------------------------------------------------
# LLM output mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        keywords_in_jd_not_in_resume=["Kubernetes", "distributed systems", "gRPC"],
        relevant_sections=["experience", "skills"],
        experience_level_signal="senior",
        tone_signal="technical and precise",
        priority_changes=[
            "Add Kubernetes to skills section",
            "Reframe experience bullets to emphasize scale",
        ],
    )


@pytest.fixture
def mock_tailor_output() -> dict:
    return {
        "modified_latex": VALID_RESUME_LATEX_MODIFIED,
        "changes": [
            {
                "section": "skills",
                "action": "added",
                "content": "Added Kubernetes, gRPC, distributed systems",
                "reason": "These are primary requirements listed in the job description",
            },
            {
                "section": "experience",
                "action": "modified",
                "content": "Reframed microservices bullet to mention Kubernetes and gRPC",
                "reason": "Directly addresses the distributed systems requirement",
            },
        ],
    }


@pytest.fixture
def mock_validation_result() -> ValidationResult:
    return ValidationResult(
        passed=True,
        fabrication_detected=False,
        latex_structure_intact=True,
        changes_verified=True,
        feedback=None,
    )


@pytest.fixture
def mock_scoring_result() -> ScoringResult:
    return ScoringResult(
        ats_score=78,
        ats_score_delta=15,
        confidence=0.85,
        remaining_gaps=["gRPC"],
    )


@pytest.fixture
def mock_llm_result(mock_tailor_output, mock_scoring_result) -> TailorResult:
    changes = [Change(**c) for c in mock_tailor_output["changes"]]
    return TailorResult(
        modified_latex=mock_tailor_output["modified_latex"],
        changes=changes,
        ats_keywords_added=["Kubernetes", "distributed systems"],
        ats_keywords_missing=mock_scoring_result.remaining_gaps,
    )


@pytest.fixture
def mock_node_responses(
    mock_analysis_result,
    mock_tailor_output,
    mock_validation_result,
    mock_scoring_result,
) -> dict:
    """Dict mapping node names to their expected state-update return values.

    Used in graph integration tests where individual nodes are patched to
    control routing and assembly logic independently of the LLM.
    """
    changes = [Change(**c) for c in mock_tailor_output["changes"]]
    return {
        "analyzer": {"analysis": mock_analysis_result},
        "tailor": {
            "modified_latex": mock_tailor_output["modified_latex"],
            "changes": changes,
            "tailor_retry_count": 1,
        },
        "validator": {"validation_passed": True, "validation_feedback": None},
        "scorer": {"scoring": mock_scoring_result},
    }


# ---------------------------------------------------------------------------
# Graph / LLM mock fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph(mock_llm_result):
    """Patch call_llm so no real Anthropic API calls happen in endpoint tests.

    We mock at the call_llm boundary rather than mocking individual graph
    nodes because endpoint tests care about the HTTP contract (status codes,
    response shapes, auth behaviour), not graph internals. This exercises
    the full Celery task code path while keeping tests fast and free of API costs.
    """
    with patch(
        "app.worker.llm.client.call_llm", return_value=mock_llm_result
    ) as mock:
        yield mock


# ---------------------------------------------------------------------------
# HTTP client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_client(mock_graph, temp_db):
    """Async HTTP client for FastAPI integration tests.

    Depends on mock_graph → all endpoint tests use mocked LLM automatically.
    Depends on temp_db → auth lookups use an isolated SQLite file per test.
    Also patches process_resume.delay to prevent Celery broker dispatch —
    endpoint tests verify the HTTP layer, not task execution.
    """
    from app.main import app

    with patch("app.api.v1.endpoints.jobs.process_resume") as mock_task:
        mock_task.delay.return_value = MagicMock()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client


# ---------------------------------------------------------------------------
# API key fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_api_key(temp_db) -> str:
    """Create a real API key through the real auth code path.

    Depends on temp_db so the key is stored in the same isolated SQLite file
    that async_client's auth middleware reads. Within a single test, pytest
    shares the temp_db instance between fixtures, so both see the same DB.
    """
    return await auth.create_api_key()
