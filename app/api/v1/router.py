from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import health, jobs

# Aggregates the v1 endpoint routers; a future v2 gets its own router module.
router = APIRouter()

router.include_router(health.router)
router.include_router(jobs.router)
