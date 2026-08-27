from __future__ import annotations

from collections.abc import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Depends, Header, HTTPException

from app.config import settings
from app.services import auth
from app.services.rate_limiter import RateLimiter


async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """Yield an async Redis client for the duration of a single request.

    FastAPI generator dependencies split at the yield: everything before
    yield is setup (runs before the endpoint), and the finally block is
    teardown (runs after the response is sent, even if the endpoint raised).
    Using a generator here ensures the connection is always closed and
    returned to the pool — no leaks even on unhandled exceptions.
    """
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

    The header is declared Optional so that a missing header produces a 401
    rather than a 422. FastAPI returns 422 for a missing *required* Header
    parameter because it treats the missing value as a request validation
    error. By making it Optional and checking for None ourselves, we produce
    the semantically correct 401: the request is structurally valid, but
    authentication credentials are absent or wrong.

    We return the key rather than True because downstream dependencies need
    the value itself — check_rate_limit uses it as the rate-limiter
    identifier so each API key has its own independent quota.
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

    Dependency chaining in action: this function depends on
    verify_api_key_header, so any endpoint that declares
    Depends(check_rate_limit) automatically gets both authentication *and*
    rate limiting applied — without listing both dependencies explicitly.
    FastAPI resolves the full dependency tree, deduplicates shared
    sub-dependencies (get_redis is only called once even if multiple
    dependencies need it in the same request), and calls them in the correct
    order. Adding a new cross-cutting concern later (e.g., IP allowlisting)
    means adding one more layer here, not touching every endpoint.
    """
    allowed, remaining = await rate_limiter.is_allowed(api_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )
    return api_key
