"""Redis rate limiter: sliding minute window, daily budget, waiting, and failing open."""

from datetime import UTC, datetime, timedelta

import fakeredis
import pytest
import redis

from app.rate_limit import DailyQuotaExceededError, RateLimitWaitTimeoutError, RedisRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def make_limiter(
    clock: FakeClock, *, rpm: int = 3, rpd: int = 10, **kw: object
) -> RedisRateLimiter:
    server = fakeredis.FakeServer()
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.advance(seconds)

    limiter = RedisRateLimiter(
        "redis://unused",
        rpm=rpm,
        rpd=rpd,
        sync_client=fakeredis.FakeRedis(server=server),
        async_client=fakeredis.FakeAsyncRedis(server=server),
        clock=clock,
        sleep=sleep,
        **kw,  # type: ignore[arg-type]
    )
    limiter.slept = slept  # type: ignore[attr-defined]
    return limiter


def test_acquires_until_minute_window_full_then_refuses_when_non_blocking() -> None:
    limiter = make_limiter(FakeClock(), rpm=3)

    assert [limiter.acquire(blocking=False) for _ in range(3)] == [True, True, True]
    assert limiter.acquire(blocking=False) is False
    assert limiter.usage()["minute_used"] == 3


def test_blocking_waits_until_oldest_request_leaves_window() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=2)
    limiter.acquire()
    clock.advance(10)
    limiter.acquire()

    assert limiter.acquire() is True
    assert limiter.slept == [pytest.approx(50.0)]  # type: ignore[attr-defined]


def test_window_slides() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=1)
    limiter.acquire()
    clock.advance(61)

    assert limiter.acquire(blocking=False) is True


def test_daily_budget_raises_even_when_minute_window_is_free() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=100, rpd=2)
    limiter.acquire()
    limiter.acquire()

    with pytest.raises(DailyQuotaExceededError):
        limiter.acquire()
    assert limiter.usage()["day_used"] == 2


def test_daily_budget_resets_on_new_utc_day() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=100, rpd=1)
    limiter.acquire()
    clock.advance(13 * 3600)  # next UTC day

    assert limiter.acquire(blocking=False) is True


def test_wait_longer_than_max_wait_raises() -> None:
    limiter = make_limiter(FakeClock(), rpm=1, max_wait_s=5)
    limiter.acquire()

    with pytest.raises(RateLimitWaitTimeoutError):
        limiter.acquire()


async def test_async_acquire_shares_counters_with_sync() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=2)
    limiter.acquire()

    assert await limiter.aacquire(blocking=False) is True
    assert await limiter.aacquire(blocking=False) is False


def test_fails_open_when_redis_is_down() -> None:
    limiter = RedisRateLimiter(
        "redis://127.0.0.1:1/0", rpm=1, rpd=1, sync_client=redis.Redis(port=1, socket_timeout=0.2)
    )

    assert limiter.acquire() is True


def test_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        RedisRateLimiter("redis://unused", rpm=0, rpd=1)
