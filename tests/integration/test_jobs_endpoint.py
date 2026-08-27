from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.conftest import VALID_JD, VALID_RESUME_LATEX

# ---------------------------------------------------------------------------
# POST /v1/jobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_job_returns_202(async_client, test_api_key):
    response = await async_client.post(
        "/v1/jobs",
        json={"resume_latex": VALID_RESUME_LATEX, "job_description": VALID_JD},
        headers={"X-API-Key": test_api_key},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert "job_id" in data
    assert "/v1/jobs/" in data["message"]


@pytest.mark.asyncio
async def test_create_job_without_api_key_returns_401(async_client):
    response = await async_client.post(
        "/v1/jobs",
        json={"resume_latex": VALID_RESUME_LATEX, "job_description": VALID_JD},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_job_with_invalid_api_key_returns_401(async_client):
    response = await async_client.post(
        "/v1/jobs",
        json={"resume_latex": VALID_RESUME_LATEX, "job_description": VALID_JD},
        headers={"X-API-Key": "this-key-does-not-exist"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_job_with_short_resume_returns_422(async_client, test_api_key):
    response = await async_client.post(
        "/v1/jobs",
        json={"resume_latex": r"\documentclass{a}", "job_description": VALID_JD},
        headers={"X-API-Key": test_api_key},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_job_with_short_jd_returns_422(async_client, test_api_key):
    response = await async_client.post(
        "/v1/jobs",
        json={"resume_latex": VALID_RESUME_LATEX, "job_description": "Too short."},
        headers={"X-API-Key": test_api_key},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_job_response_shape(async_client, test_api_key):
    response = await async_client.post(
        "/v1/jobs",
        json={"resume_latex": VALID_RESUME_LATEX, "job_description": VALID_JD},
        headers={"X-API-Key": test_api_key},
    )
    data = response.json()
    assert set(data.keys()) >= {"job_id", "status", "message"}
    assert data["status"] == "queued"


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_without_api_key_returns_401(async_client):
    response = await async_client.get("/v1/jobs/some-job-id")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_job_pending_returns_queued(async_client, test_api_key):
    with patch("app.api.v1.endpoints.jobs.AsyncResult") as mock_cls:
        mock_cls.return_value.state = "PENDING"

        response = await async_client.get(
            "/v1/jobs/nonexistent-job-id",
            headers={"X-API-Key": test_api_key},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["result"] is None


@pytest.mark.asyncio
async def test_get_job_started_returns_processing(async_client, test_api_key):
    with patch("app.api.v1.endpoints.jobs.AsyncResult") as mock_cls:
        mock_cls.return_value.state = "STARTED"

        response = await async_client.get(
            "/v1/jobs/running-job-id",
            headers={"X-API-Key": test_api_key},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "processing"


@pytest.mark.asyncio
async def test_get_job_success_returns_complete_with_result(
    async_client, test_api_key, mock_llm_result
):
    with patch("app.api.v1.endpoints.jobs.AsyncResult") as mock_cls:
        mock_cls.return_value.state = "SUCCESS"
        mock_cls.return_value.get.return_value = mock_llm_result.model_dump()

        response = await async_client.get(
            "/v1/jobs/done-job-id",
            headers={"X-API-Key": test_api_key},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "complete"
    assert data["result"] is not None
    assert "modified_latex" in data["result"]
    assert isinstance(data["result"]["changes"], list)


@pytest.mark.asyncio
async def test_get_job_failure_returns_failed_with_error(async_client, test_api_key):
    with patch("app.api.v1.endpoints.jobs.AsyncResult") as mock_cls:
        mock_cls.return_value.state = "FAILURE"
        mock_cls.return_value.info = RuntimeError("LLM call failed")

        response = await async_client.get(
            "/v1/jobs/failed-job-id",
            headers={"X-API-Key": test_api_key},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert data["error"] is not None


# ---------------------------------------------------------------------------
# Health endpoints (no auth required)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_liveness_probe_no_auth(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_liveness_probe_returns_environment(async_client):
    response = await async_client.get("/health")
    data = response.json()
    assert "environment" in data
