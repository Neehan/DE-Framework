"""Batch-wide credential rotation with shared cooldowns and exhausted-key exclusion."""

import logging
from asyncio import FIRST_COMPLETED, Event, create_task, ensure_future, gather, wait
from collections.abc import Awaitable, Callable
from math import inf, isfinite

from harness.proxy.constants import CREDENTIAL_COOLDOWN_SECONDS
from harness.proxy.models import ProviderConfig
from harness.utils.constants import COUNT_INCREMENT, INITIAL_COUNT


class CredentialPool:
    """Acquire eligible credentials, cool rate limits, and disable exhausted keys for the batch; no overrides required.

    Selection and cooldown updates never await, so concurrent tasks need no lock. Inject the clock and cancellable wait for deterministic tests.
    """

    def __init__(
        self, providers: tuple[ProviderConfig, ...], now: Callable[[], float],
        wait: Callable[[float], Awaitable[None]],
    ) -> None:
        """Deduplicate endpoint/credential pairs and require one authentication mode for the batch."""
        if not providers:
            raise ValueError("credential pool must not be empty")
        if len({provider.auth_variable for provider in providers}) != COUNT_INCREMENT:
            raise ValueError("credential pool requires one authentication mode")
        self._cool_until = dict.fromkeys(providers, float(INITIAL_COUNT))
        self._next_index = INITIAL_COUNT
        self._now = now
        self._wait = wait
        self._changed = Event()

    async def acquire(self, current: ProviderConfig | None) -> ProviderConfig:
        """Keep an eligible current credential, otherwise rotate; wait only when every credential is cooling."""
        providers = tuple(self._cool_until)
        while True:
            now = self._now()
            if current is not None and self._cool_until[current] <= now:
                return current
            for offset in range(len(providers)):
                index = (self._next_index + offset) % len(providers)
                provider = providers[index]
                if self._cool_until[provider] <= now:
                    self._next_index = (index + COUNT_INCREMENT) % len(providers)
                    return provider
            earliest = min(self._cool_until.values())
            if not isfinite(earliest):
                raise RuntimeError("All provider credentials reached their spend limits; add usable credentials or raise the provider limit")
            await self._wait_for_change(earliest - now)

    def is_available(self, provider: ProviderConfig) -> bool:
        """Check eligibility without waiting inside an active inference request."""
        return self._cool_until[provider] <= self._now()

    def mark_rate_limited(self, provider: ProviderConfig, resets_at: float | None) -> None:
        """Use a future reset time or five minutes; never shorten an existing cooldown."""
        if self.is_disabled(provider):
            return
        now = self._now()
        until = resets_at if resets_at is not None and isfinite(resets_at) and resets_at > now else now + CREDENTIAL_COOLDOWN_SECONDS
        self._cool_until[provider] = max(self._cool_until[provider], until)
        logging.getLogger(__name__).warning("Provider credential rate-limited; cooling for %.0f seconds", self._cool_until[provider] - now)

    def is_disabled(self, provider: ProviderConfig) -> bool:
        """Distinguish a permanently exhausted key from one awaiting a temporary reset."""
        return self._cool_until[provider] == inf

    def disable(self, provider: ProviderConfig) -> None:
        """Exclude an exhausted key for this batch and wake acquisitions waiting on its cooldown."""
        if self.is_disabled(provider):
            return
        self._cool_until[provider] = inf
        self._changed.set()
        self._changed = Event()
        logging.getLogger(__name__).warning("Provider credential disabled for this batch: spend limit reached")

    async def _wait_for_change(self, seconds: float) -> None:
        """Wake on cooldown expiry or disabled-key changes; cancel both waits when the caller exits."""
        pending = (ensure_future(self._wait(seconds)), create_task(self._changed.wait()))
        try:
            done, _ = await wait(pending, return_when=FIRST_COMPLETED)
            for completed in done:
                completed.result()
        finally:
            for task in pending:
                task.cancel()
            await gather(*pending, return_exceptions=True)
