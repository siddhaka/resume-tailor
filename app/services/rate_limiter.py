from __future__ import annotations

import hashlib
import time
import uuid

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Sliding-window rate limiter backed by a Redis sorted set.

    Each identifier (a hashed API key) gets one sorted set: members are request
    UUIDs, scores are millisecond timestamps. A ZREMRANGEBYSCORE + ZCARD pair
    counts requests in the trailing 60 s, which avoids the boundary burst a
    fixed-window counter allows. Millisecond scores plus UUID members avoid
    collisions; the key's 70 s TTL outlives a full window so it can't reset early.
    """

    def __init__(self, redis_client: aioredis.Redis, limit_per_minute: int) -> None:
        self._redis = redis_client
        self._limit = limit_per_minute

    async def is_allowed(self, identifier: str) -> tuple[bool, int]:
        """Return (is_allowed, requests_remaining) for this identifier.

        The identifier is hashed before use as the Redis key so raw keys are
        never stored.
        """
        key = "rate_limit:" + hashlib.sha256(identifier.encode()).hexdigest()
        now_ms = int(time.time() * 1000)
        window_start_ms = now_ms - 60_000
        member = str(uuid.uuid4())

        log = logger.bind(rate_limit_key=key[-12:])

        try:
            async with self._redis.pipeline(transaction=False) as pipe:
                # Remove requests older than 60 s
                pipe.zremrangebyscore(key, 0, window_start_ms)
                # Count requests in the current window
                pipe.zcard(key)
                # Add this request
                pipe.zadd(key, {member: now_ms})
                # Ensure the key expires so Redis memory is not leaked
                pipe.expire(key, 70)

                results = await pipe.execute()

            current_count: int = results[1]  # zcard result, before this request

            if current_count >= self._limit:
                log.warning(
                    "rate_limit.exceeded",
                    count=current_count,
                    limit=self._limit,
                )
                # Undo the ZADD — this request is rejected, don't count it
                await self._redis.zrem(key, member)
                return False, 0

            remaining = self._limit - current_count - 1
            log.debug(
                "rate_limit.allowed",
                count=current_count + 1,
                remaining=remaining,
            )
            return True, remaining

        except Exception:
            # Fail open: a Redis outage should not take down the API.
            # Log loudly so on-call is alerted, but let the request through.
            log.exception("rate_limit.redis_error")
            return True, self._limit
