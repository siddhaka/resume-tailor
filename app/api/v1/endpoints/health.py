from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.dependencies import get_redis

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict[str, Any]:
    """Liveness probe: is the process alive? No dependency checks, no auth.

    Deliberately does not check Redis — restarting the pod can't fix a Redis
    outage, so a dependency check here would only churn healthy pods.
    """
    return {"status": "ok", "environment": settings.environment}


@router.get("/health/ready")
async def readiness(redis: aioredis.Redis = Depends(get_redis)) -> dict[str, Any]:
    """Readiness probe: can this pod serve traffic right now? No auth.

    Checks Redis (needed for queuing and rate limiting). On failure Kubernetes
    stops routing to the pod but does not restart it, so it rejoins the rotation
    automatically once Redis recovers.
    """
    try:
        await redis.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Redis unavailable") from exc

    return {"status": "ready", "redis": "connected"}
