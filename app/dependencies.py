from __future__ import annotations

from collections.abc import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Depends, Header, HTTPException

from app.config import settings
from app.services import auth
from app.services.rate_limiter import RateLimiter


async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """Yield a request-scoped async Redis client, closed in the finally block."""
    client: aioredis.Redis = aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=True
    )
    try:
        yield client
    finally:
        await client.aclose()


async def get_rate_limiter(
    redis: aioredis.Redis = Depends(get_redis),
) -> RateLimiter:
    """Return a RateLimiter wired to the request-scoped Redis client."""
    return RateLimiter(redis, settings.rate_limit_per_minute)


async def verify_api_key_header(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> str:
    """Validate the X-API-Key header and return the plaintext key.

    Declared Optional so a missing header yields 401, not FastAPI's default 422
    for a missing required header. Returns the key (not a bool) because
    check_rate_limit uses it as the per-key rate-limit identifier.
    """
    if not x_api_key or not await auth.verify_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def check_rate_limit(
    api_key: str = Depends(verify_api_key_header),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
) -> str:
    """Enforce per-key rate limiting, then return the API key.

    Depends on verify_api_key_header, so Depends(check_rate_limit) gives an
    endpoint both auth and rate limiting in one dependency.
    """
    allowed, remaining = await rate_limiter.is_allowed(api_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )
    return api_key
