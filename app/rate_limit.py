"""Redis-backed rate limiter for OpenRouter's free-model quota.

OpenRouter allows 20 requests/minute and 50 requests/day (1000 with purchased credits) on
`:free` models. Every HubScout process and parallel scout shares the same quota, so the
counters live in Redis and are updated atomically by one Lua script:

- per-minute: a sliding 60 s window (sorted set of request timestamps). When full, the
  limiter waits until the oldest request leaves the window.
- per-day: a counter keyed by UTC date. When exhausted, the limiter raises
  `DailyQuotaExceededError` immediately; `.with_fallbacks()` then routes the call to Ollama.

If Redis is unreachable the limiter fails open (logs a warning and lets the call through):
OpenRouter still enforces its limits server-side, and a 429 also triggers the fallback.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import redis
import redis.asyncio as aioredis
from langchain_core.rate_limiters import BaseRateLimiter

logger = logging.getLogger(__name__)

WINDOW_MS = 60_000
DAY_TTL_S = 2 * 24 * 3600

# Returns 0 = acquired, -1 = daily quota exhausted, >0 = milliseconds to wait.
_ACQUIRE_LUA = """
local minute_key, day_key = KEYS[1], KEYS[2]
local now_ms = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local rpm = tonumber(ARGV[3])
local rpd = tonumber(ARGV[4])
local member = ARGV[5]
local day_ttl = tonumber(ARGV[6])

if tonumber(redis.call('GET', day_key) or '0') >= rpd then
  return -1
end

redis.call('ZREMRANGEBYSCORE', minute_key, '-inf', now_ms - window_ms)
if redis.call('ZCARD', minute_key) >= rpm then
  local oldest = redis.call('ZRANGE', minute_key, 0, 0, 'WITHSCORES')
  local wait = tonumber(oldest[2]) + window_ms - now_ms
  if wait < 1 then wait = 1 end
  return wait
end

redis.call('ZADD', minute_key, now_ms, member)
redis.call('PEXPIRE', minute_key, window_ms)
redis.call('INCR', day_key)
redis.call('EXPIRE', day_key, day_ttl)
return 0
"""


class RateLimitError(RuntimeError):
    """Base class: the primary provider cannot be called right now; callers should fall back."""


class DailyQuotaExceededError(RateLimitError):
    """The provider's daily request budget is spent."""


class RateLimitWaitTimeoutError(RateLimitError):
    """Waiting for a free slot in the per-minute window would exceed `max_wait_s`.

    Raised rather than returning False because LangChain chat models ignore `acquire()`'s
    return value; raising lets `.with_fallbacks()` route the call elsewhere.
    """


class RedisRateLimiter(BaseRateLimiter):
    """Shared per-minute + per-day limiter implementing LangChain's `BaseRateLimiter`."""

    def __init__(
        self,
        redis_url: str,
        *,
        rpm: int,
        rpd: int,
        key_prefix: str = "hubscout:ratelimit:openrouter",
        max_wait_s: float = 120.0,
        sync_client: redis.Redis | None = None,
        async_client: aioredis.Redis | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rpm <= 0 or rpd <= 0:
            raise ValueError("rpm and rpd must be positive")
        self.rpm = rpm
        self.rpd = rpd
        self.key_prefix = key_prefix
        self.max_wait_s = max_wait_s
        self._clock = clock
        self._sleep = sleep
        self._async_sleep = async_sleep
        self._sync = sync_client or redis.Redis.from_url(redis_url, socket_timeout=2)
        self._async = async_client or aioredis.Redis.from_url(redis_url, socket_timeout=2)
        self._sync_script = self._sync.register_script(_ACQUIRE_LUA)
        self._async_script = self._async.register_script(_ACQUIRE_LUA)

    # ------------------------------------------------------------------ helpers
    def _keys(self, now: datetime) -> list[str]:
        return [f"{self.key_prefix}:minute", f"{self.key_prefix}:day:{now:%Y-%m-%d}"]

    def _args(self, now: datetime) -> list[str | int]:
        return [
            int(now.timestamp() * 1000),
            WINDOW_MS,
            self.rpm,
            self.rpd,
            uuid.uuid4().hex,
            DAY_TTL_S,
        ]

    def _interpret(self, result: int, blocking: bool) -> float | bool:
        """Map a script result to True (acquired), False, or seconds to wait."""
        if result == 0:
            return True
        if result == -1:
            raise DailyQuotaExceededError(
                f"OpenRouter daily budget of {self.rpd} requests is used up (resets 00:00 UTC)"
            )
        return (result / 1000) if blocking else False

    # ------------------------------------------------------------------ API
    def acquire(self, *, blocking: bool = True) -> bool:
        waited = 0.0
        while True:
            now = self._clock()
            try:
                result = int(self._sync_script(keys=self._keys(now), args=self._args(now)))
            except redis.RedisError as exc:
                logger.warning("rate limiter: Redis unavailable, failing open (%s)", exc)
                return True
            outcome = self._interpret(result, blocking)
            if isinstance(outcome, bool):
                return outcome
            if waited + outcome > self.max_wait_s:
                raise RateLimitWaitTimeoutError(
                    f"per-minute limit ({self.rpm}/min) would need more than {self.max_wait_s}s"
                )
            self._sleep(outcome)
            waited += outcome

    async def aacquire(self, *, blocking: bool = True) -> bool:
        waited = 0.0
        while True:
            now = self._clock()
            try:
                result = int(await self._async_script(keys=self._keys(now), args=self._args(now)))
            except redis.RedisError as exc:
                logger.warning("rate limiter: Redis unavailable, failing open (%s)", exc)
                return True
            outcome = self._interpret(result, blocking)
            if isinstance(outcome, bool):
                return outcome
            if waited + outcome > self.max_wait_s:
                raise RateLimitWaitTimeoutError(
                    f"per-minute limit ({self.rpm}/min) would need more than {self.max_wait_s}s"
                )
            await self._async_sleep(outcome)
            waited += outcome

    def usage(self) -> dict[str, int]:
        """Current counts, for logs and health endpoints."""
        now = self._clock()
        minute_key, day_key = self._keys(now)
        self._sync.zremrangebyscore(minute_key, "-inf", int(now.timestamp() * 1000) - WINDOW_MS)
        return {
            "minute_used": int(self._sync.zcard(minute_key)),
            "day_used": int(self._sync.get(day_key) or 0),
            "rpm": self.rpm,
            "rpd": self.rpd,
        }
