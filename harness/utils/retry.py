"""Shared bounded connection-retry timing without owning task or checkpoint state."""

import logging
from collections.abc import Awaitable, Callable

from harness.utils.constants import (
    COUNT_INCREMENT,
    RECOVERY_BACKOFF_MULTIPLIER,
    RECOVERY_BASE_DELAY_SECONDS,
    RECOVERY_MAX_DELAY_SECONDS,
    RECOVERY_MAX_RETRIES,
)


async def wait_before_retry(error: Exception, retries: int, wait: Callable[[float], Awaitable[None]]) -> None:
    """Delay the next connection attempt; callers decide what progress to restore."""
    delay = min(RECOVERY_BASE_DELAY_SECONDS * RECOVERY_BACKOFF_MULTIPLIER ** retries, RECOVERY_MAX_DELAY_SECONDS)
    logging.getLogger(__name__).warning("Retrying connection in %s seconds (%s/%s): %s", delay, retries + COUNT_INCREMENT, RECOVERY_MAX_RETRIES, error)
    await wait(delay)
