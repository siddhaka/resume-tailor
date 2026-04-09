from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.dependencies import get_redis

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict[str, Any]:
    """Kubernetes liveness probe.

    A liveness probe answers one question: "is this process alive and able
    to handle requests at all?" It should never check external dependencies.
    If Redis is down, the *process* is still alive — killing and restarting
    the pod won't fix a Redis outage, it will just cause unnecessary churn.
    Kubernetes restarts a pod when its liveness probe fails, so a liveness
    probe that checks Redis would restart healthy API pods every time Redis
    has a blip — the opposite of what we want.

    No authentication required: Kubernetes kubelet calls this from outside
    the application's auth layer.
    """
    return {"status": "ok", "environment": settings.environment}


@router.get("/health/ready")
async def readiness(redis: aioredis.Redis = Depends(get_redis)) -> dict[str, Any]:
    """Kubernetes readiness probe.

    A readiness probe answers a different question: "is this pod ready to
    receive traffic right now?" Unlike liveness, readiness *should* check
    external dependencies. If Redis is unavailable this pod cannot usefully
    serve requests (rate limiting and job queuing both require Redis), so
    Kubernetes should stop routing traffic to it.

    When a readiness probe fails Kubernetes removes the pod from the Service
    endpoints — no new requests are routed to it — but does NOT restart it.
    This is the correct behavior: the pod itself is healthy, its dependency
    is not. Traffic shifts to other pods that are still ready. When Redis
    recovers, the probe passes and the pod is re-added to the rotation
    automatically, with no operator intervention required.

    No authentication required: Kubernetes kubelet calls this without headers.
    """
    try:
        await redis.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Redis unavailable") from exc

    return {"status": "ready", "redis": "connected"}
