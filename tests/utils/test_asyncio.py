"""Termination reaches the running operation and completes its cleanup."""

import asyncio
import os
import signal

from harness.utils.asyncio import run_process
from harness.utils.constants import INTERRUPTED_EXIT_CODE


def test_sigterm_cancels_operation_and_runs_cleanup() -> None:
    """Send SIGTERM only after the process helper installs its handler; no child or provider is involved."""
    cleaned = []

    async def operation() -> int:
        """Wait for the scheduled termination and record cleanup before returning control."""
        asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)
        try:
            await asyncio.Event().wait()
            raise AssertionError("terminated operation must not complete normally")
        finally:
            cleaned.append(True)

    assert run_process(operation()) == INTERRUPTED_EXIT_CODE
    assert cleaned == [True]
