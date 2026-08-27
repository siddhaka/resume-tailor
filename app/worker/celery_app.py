from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "resume_tailor",
    broker=settings.redis_url,
    backend=settings.redis_url,
    # Import the task module on startup so @celery_app.task definitions are
    # registered. Without this the worker starts but rejects incoming jobs as
    # "unregistered task", and the API pod (which only sends by name) can enqueue
    # work that no worker will ever accept.
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
)
