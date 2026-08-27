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
    """Queue a tailor request and return 202 immediately.

    The LLM work runs in the Celery worker, not this thread; the client polls
    GET /v1/jobs/{job_id} for the result.
    """
    job_id = str(uuid4())
    trace_id = str(uuid4())

    # Log a hash prefix for correlation, never the plaintext key.
    key_hash_prefix = hashlib.sha256(api_key.encode()).hexdigest()[:12]

    log = logger.bind(job_id=job_id, trace_id=trace_id, api_key_hash=key_hash_prefix)
    log.info("job.created")

    # Bind our job_id as Celery's task_id so GET /v1/jobs/{job_id} can look the
    # task up directly. Without task_id=job_id, Celery assigns its own random
    # task id and the status endpoint — which queries AsyncResult(job_id) — would
    # never resolve, leaving every job stuck reporting "queued".
    process_resume.apply_async(
        kwargs={
            "job_id": job_id,
            "resume_latex": request.resume_latex,
            "job_description": request.job_description,
        },
        task_id=job_id,
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
    """Return a job's current status and result (if complete).

    Not rate limited — clients poll this every few seconds; the limit guards
    the expensive POST, not cheap status reads. Celery reports PENDING for both
    a queued task and an unknown id (it keeps no id registry), so PENDING is
    surfaced as "queued".
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
