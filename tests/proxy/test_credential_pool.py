"""Round-robin assignment, shared cooldowns, and cancellation without real waiting."""

from asyncio import CancelledError, Event, create_task, gather, timeout
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from harness.proxy.constants import AUTH_TOKEN_ENV, CREDENTIAL_COOLDOWN_SECONDS
from harness.proxy.credential_pool import CredentialPool
from harness.proxy.models import ProviderConfig

from tests.constants import ASYNC_TEST_TIMEOUT_SECONDS
from tests.proxy.constants import POOL_NOW, RESET_DELAY


async def test_round_robin_assignments_keep_current_credentials(pool: CredentialPool, providers: tuple[ProviderConfig, ...]) -> None:
    """Concurrent runs rotate A/B/A/B, while an existing run retains its eligible credential."""
    assert await gather(*(pool.acquire(None) for _ in providers)) == list(providers)
    for provider in providers:
        assert await pool.acquire(provider) == provider
    assert await gather(*(pool.acquire(None) for _ in providers)) == list(providers)


async def test_shared_cooldown_skips_current_and_new_assignments(
    pool: CredentialPool, providers: tuple[ProviderConfig, ...], pool_wait: AsyncMock,
) -> None:
    """A rate limit discovered by one run affects every run using that same credential."""
    first, second = providers
    pool.mark_rate_limited(first, POOL_NOW + RESET_DELAY)
    assert await pool.acquire(first) == second
    assert await pool.acquire(None) == second
    pool_wait.assert_not_awaited()


@pytest.mark.parametrize("reset", [None, POOL_NOW, POOL_NOW - RESET_DELAY, float("nan"), float("inf")])
async def test_unusable_reset_uses_five_minutes(
    providers: tuple[ProviderConfig, ...], pool_clock: Mock, pool_wait: AsyncMock, reset: float | None,
) -> None:
    """Missing, expired, or invalid reset times cannot hot-loop a limited credential."""
    first, _ = providers
    pool = CredentialPool((first,), pool_clock, pool_wait)
    pool.mark_rate_limited(first, reset)
    assert await pool.acquire(first) == first
    pool_wait.assert_awaited_once_with(CREDENTIAL_COOLDOWN_SECONDS)


async def test_all_cooling_waits_for_earliest_without_shortening_resets(
    pool: CredentialPool, providers: tuple[ProviderConfig, ...], pool_wait: AsyncMock,
) -> None:
    """Out-of-order rate-limit reports cannot shorten cooldown; the earliest ready credential wins."""
    first, second = providers
    pool.mark_rate_limited(first, POOL_NOW + CREDENTIAL_COOLDOWN_SECONDS)
    pool.mark_rate_limited(first, POOL_NOW + RESET_DELAY)
    pool.mark_rate_limited(second, POOL_NOW + RESET_DELAY)
    assert await pool.acquire(first) == second
    pool_wait.assert_awaited_once_with(RESET_DELAY)


async def test_cooldown_wait_is_cancellable(providers: tuple[ProviderConfig, ...], pool_clock: Mock) -> None:
    """Stopping a batch must not wait five minutes for pool acquisition to finish."""
    entered, blocked = Event(), Event()

    async def wait(seconds: float) -> None:
        """Expose the acquisition wait to the cancelling caller."""
        entered.set()
        await blocked.wait()

    pool = CredentialPool(providers, pool_clock, wait)
    for provider in providers:
        pool.mark_rate_limited(provider, None)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(pool.acquire(None))
        await entered.wait()
        task.cancel()
        with pytest.raises(CancelledError):
            await task


async def test_duplicate_credentials_do_not_overweight_rotation(
    providers: tuple[ProviderConfig, ...], pool_clock: Mock, pool_wait: AsyncMock,
) -> None:
    """Repeated environment aliases represent one credential, not extra round-robin slots."""
    first, second = providers
    pool = CredentialPool((first, first, second), pool_clock, pool_wait)
    assert await pool.acquire(None) == first
    assert await pool.acquire(None) == second


def test_pool_rejects_mixed_routes(
    providers: tuple[ProviderConfig, ...], pool_clock: Mock, pool_wait: AsyncMock,
) -> None:
    """A run token must not change authentication mode during endpoint rotation."""
    first, second = providers
    with pytest.raises(ValueError):
        CredentialPool((first, replace(second, auth_variable=AUTH_TOKEN_ENV)), pool_clock, pool_wait)
    with pytest.raises(ValueError):
        CredentialPool((), pool_clock, pool_wait)


async def test_shared_gateway_key_keeps_endpoint_cooldowns_independent(
    providers: tuple[ProviderConfig, ...], pool_clock: Mock, pool_wait: AsyncMock,
) -> None:
    """Codex accounts have separate endpoints even when their local authentication key is shared."""
    first = providers[0]
    second = replace(first, url="https://second.example")
    pool = CredentialPool((first, second), pool_clock, pool_wait)
    assert await pool.acquire(None) == first
    assert await pool.acquire(None) == second
    pool.mark_rate_limited(first, None)
    assert await pool.acquire(first) == second
    assert not pool.is_available(first) and pool.is_available(second)
    pool_wait.assert_not_awaited()


async def test_disabled_key_stays_excluded_after_clock_advance_and_late_rate_limit(
    pool: CredentialPool, providers: tuple[ProviderConfig, ...], pool_clock: Mock, pool_wait: AsyncMock,
) -> None:
    """Exhaustion is shared, idempotent, and cannot be undone by an older in-flight cooldown response."""
    first, second = providers
    pool.disable(first)
    pool.disable(first)
    pool_clock.return_value += CREDENTIAL_COOLDOWN_SECONDS
    pool.mark_rate_limited(first, None)
    assert pool.is_disabled(first) and not pool.is_available(first)
    assert await pool.acquire(first) == second
    assert await pool.acquire(None) == second
    pool.disable(second)
    with pytest.raises(RuntimeError, match="All provider credentials"):
        await pool.acquire(first)
    pool_wait.assert_not_awaited()


async def test_disabling_all_keys_wakes_existing_cooldown_waiters(
    providers: tuple[ProviderConfig, ...], pool_clock: Mock,
) -> None:
    """A late spend-limit response fails a waiting batch immediately and cancels its old timer."""
    entered, cancelled = Event(), Event()

    async def wait(seconds: float) -> None:
        """Expose a long provider reset while another request exhausts its credential."""
        entered.set()
        try:
            await Event().wait()
        finally:
            cancelled.set()

    pool = CredentialPool(providers, pool_clock, wait)
    for provider in providers:
        pool.mark_rate_limited(provider, None)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(pool.acquire(None))
        await entered.wait()
        for provider in providers:
            pool.disable(provider)
        with pytest.raises(RuntimeError, match="All provider credentials"):
            await task
    assert cancelled.is_set()
