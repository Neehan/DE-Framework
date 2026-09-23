"""Shared process termination and cancellation-safe resource creation."""

import asyncio
import signal
from collections.abc import Coroutine
from typing import Any, TypeVar

from harness.utils.constants import INTERRUPTED_EXIT_CODE

T = TypeVar("T")


def run_process(operation: Coroutine[Any, Any, int]) -> int:
    """Run an async entrypoint, allowing SIGINT and SIGTERM to cancel it and finish cleanup."""
    try:
        return asyncio.run(_run_with_termination(operation))
    except (KeyboardInterrupt, asyncio.CancelledError):
        return INTERRUPTED_EXIT_CODE


async def _run_with_termination(operation: Coroutine[Any, Any, int]) -> int:
    """Cancel the owning task on SIGTERM and remove its handler when execution ends."""
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await operation
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


async def finish_before_cancelling(operation: Coroutine[Any, Any, T], description: str) -> T:
    """Finish resource creation before cancellation allows its owner to perform cleanup."""
    creation = asyncio.create_task(operation)
    try:
        return await asyncio.shield(creation)
    except asyncio.CancelledError as cancelled:
        try:
            await creation
        except Exception as error:
            cancelled.add_note(f"{description} also failed: {error}")
        raise
