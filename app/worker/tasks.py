from __future__ import annotations

import structlog

from app.worker.celery_app import celery_app
from app.worker.llm.client import call_llm

logger = structlog.get_logger(__name__)


@celery_app.task(bind=True, name="resume_tailor.process_resume")
def process_resume(self, job_id: str, resume_latex: str, job_description: str) -> dict:  # type: ignore[override]
    """Run the pipeline; return a dict so Celery's JSON backend can store it.

    The status endpoint reads it back with TailorResult(**task.get()).
    """
    log = logger.bind(job_id=job_id)
    log.info("task.started")

    try:
        result = call_llm(
            resume_latex=resume_latex,
            job_description=job_description,
        )
        log.info("task.complete", changes=len(result.changes))
        return result.model_dump()
    except Exception as exc:
        log.exception("task.failed", error=str(exc))
        raise
