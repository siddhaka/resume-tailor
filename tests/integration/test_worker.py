"""Worker wiring tests.

These guard the Celery app configuration rather than task behaviour. The
tailoring task itself is exercised through the graph tests; here we only
assert that the worker is wired up correctly, because a misconfigured worker
fails silently — it starts, connects to the broker, and then discards every
incoming job as an "unregistered task".
"""

from __future__ import annotations

from app.worker.celery_app import celery_app
from app.worker.tasks import process_resume


def test_process_resume_task_is_registered():
    # The API sends jobs by task name only; if the worker's Celery app does not
    # import the task module on startup, the name is unknown and jobs are
    # dropped. This asserts the task is registered under the expected name.
    assert "resume_tailor.process_resume" in celery_app.tasks


def test_task_module_is_included_on_the_app():
    # The include list is what triggers task-module import at worker startup.
    assert "app.worker.tasks" in celery_app.conf.include


def test_process_resume_has_expected_name():
    assert process_resume.name == "resume_tailor.process_resume"
