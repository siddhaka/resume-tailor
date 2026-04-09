from __future__ import annotations

import hashlib
from uuid import uuid4

import structlog
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import check_rate_limit, verify_api_key_header
from app.models.requests import TailorRequest
from app.models.responses import CreateJobResponse, JobResponse, TailorResult
from app.worker.tasks import process_resume

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.post("", status_code=202, response_model=CreateJobResponse)
async def create_job(
    request: TailorRequest,
    api_key: str = Depends(check_rate_limit),
) -> CreateJobResponse:
    """Accept a tailor request and immediately queue it for async processing.

    Returns 202 Accepted — not 200 OK — because the work has not completed
    yet. The entire point of this architecture is that the HTTP response is
    immediate: the LLM call (which can take 10–30 s) runs in the Celery
    worker, not in this thread. The client polls GET /v1/jobs/{job_id} for
    the result.
    """
    job_id = str(uuid4())
    trace_id = str(uuid4())

    # Hash the api_key for log correlation — never log the plaintext key.
    # The truncated hash is enough to tie log lines together without
    # exposing anything that could be used to authenticate as this client.
    key_hash_prefix = hashlib.sha256(api_key.encode()).hexdigest()[:12]

    log = logger.bind(job_id=job_id, trace_id=trace_id, api_key_hash=key_hash_prefix)
    log.info("job.created")

    process_resume.delay(
        job_id=job_id,
        resume_latex=request.resume_latex,
        job_description=request.job_description,
    )

    return CreateJobResponse(
        job_id=job_id,
        status="queued",
        message=f"Poll /v1/jobs/{job_id} for status",
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    api_key: str = Depends(verify_api_key_header),
) -> JobResponse:
    """Return the current status and result (if complete) for a job.

    Rate limiting is intentionally NOT applied to this endpoint — only to
    POST /v1/jobs. A client legitimately polls this endpoint every few
    seconds while waiting for their LLM job to finish. Applying a per-minute
    limit to polling would cause clients to hit 429s during normal usage,
    since a 20-second LLM job requires ~6-10 polls at 2-second intervals.
    The attack we're defending against with rate limiting is expensive job
    *creation* (each POST triggers an LLM call that costs money). Polling
    only reads a Redis state key — it is cheap and harmless to allow freely.

    On the PENDING state: Celery returns PENDING for a task ID that is
    genuinely queued (waiting for a worker) AND for a task ID that was never
    submitted at all. Celery has no registry of "known job IDs" — if you ask
    about a random UUID it returns PENDING, not 404. This is a known Celery
    limitation. In a hardened system you would store job IDs in Redis or a
    database at creation time and return 404 here for unknown IDs. For now,
    PENDING means "queued or unknown" and we surface it as status="queued".
    """
    task = AsyncResult(job_id)
    state = task.state

    if state == "PENDING":
        return JobResponse(job_id=job_id, status="queued")

    if state == "STARTED":
        return JobResponse(job_id=job_id, status="processing")

    if state == "SUCCESS":
        raw = task.get()
        try:
            result = TailorResult(**raw)
        except Exception as exc:
            logger.error("job.result_parse_error", job_id=job_id, error=str(exc))
            raise HTTPException(
                status_code=500, detail="Job result could not be parsed"
            ) from exc
        return JobResponse(job_id=job_id, status="complete", result=result)

    if state == "FAILURE":
        return JobResponse(job_id=job_id, status="failed", error=str(task.info))

    # REVOKED, RETRY, or any custom state
    return JobResponse(job_id=job_id, status="unknown")
