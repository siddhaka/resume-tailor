from __future__ import annotations

import hashlib
import time
import uuid

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Sliding-window rate limiter backed by a Redis sorted set.

    Each unique identifier (typically a hashed API key) gets its own sorted
    set key. Members are individual request UUIDs; scores are request
    timestamps in milliseconds.

    Why sorted sets implement sliding window correctly
    --------------------------------------------------
    A sorted set lets us ask "how many requests arrived in the last 60 s?"
    with a single ZREMRANGEBYSCORE + ZCARD pair. The window slides with real
    time: a request made at T=0 expires at T=60 s regardless of how many
    other requests arrived between those moments. Every check reflects the
    true 60-second history, not an arbitrary clock boundary.

    Why fixed-window counters have a boundary exploit
    -------------------------------------------------
    A fixed window resets at wall-clock boundaries (e.g., :00 of every
    minute). A caller who sends 60 requests at 00:59 and another 60 requests
    at 01:00 has made 120 requests in 2 seconds while the counter shows
    60 in each window — double the intended limit. The sorted-set approach
    makes this impossible: those 120 requests all fall inside the same 60 s
    window regardless of which clock-minute they straddle.

    Why millisecond timestamps as scores
    ------------------------------------
    Two requests arriving in the same second would collide if scores were
    integer seconds — ZADD would treat them as the same entry and only one
    would be stored. Milliseconds make accidental score collision roughly
    1000× less likely. We also use a UUID as the member value so that even
    two requests with identical millisecond timestamps produce distinct
    members.

    Why TTL is 70 seconds, not 60
    ------------------------------
    After the last request in a window the sorted set will be empty but
    still exists in Redis until its TTL fires. A 60 s TTL risks the key
    expiring while entries from 59 s ago are still within the window if
    clock skew or pipeline latency adds a small delay. 70 s provides a
    10-second buffer: the key outlives its contents, and we avoid a race
    where a fresh request on an just-expired key sees an empty set and
    resets the window early.
    """

    def __init__(self, redis_client: aioredis.Redis, limit_per_minute: int) -> None:
        self._redis = redis_client
        self._limit = limit_per_minute

    async def is_allowed(self, identifier: str) -> tuple[bool, int]:
        """Check whether *identifier* may make another request right now.

        Returns (is_allowed, requests_remaining).  requests_remaining is 0
        when the limit has been reached; otherwise it reflects how many more
        requests are permitted within the current sliding window.

        The identifier is hashed before use as a Redis key so that raw API
        keys (or IP addresses) are never written to Redis storage.
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
