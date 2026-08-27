from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import health, jobs

# The /v1 prefix lives here rather than on individual endpoints.
# When we add v2, we create app/api/v2/router.py with its own prefix and
# include it alongside this one in main.py — no existing endpoint code
# changes. Version negotiation is a routing concern, not an endpoint concern.
router = APIRouter()

router.include_router(health.router)
router.include_router(jobs.router)
