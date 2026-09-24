"""Small shared helpers for collecting agent events and creating subprocess doubles."""

from asyncio import sleep
from asyncio.subprocess import Process
from contextlib import aclosing
from time import time
from unittest.mock import MagicMock, create_autospec

from harness.proxy.credential_pool import CredentialPool
from harness.proxy.models import ProviderConfig
from harness.session.agent_session import AgentSession
from harness.session.models import SessionEvent

from tests.constants import SANDBOX_FAILURE


async def collect_events(session: AgentSession, prompt: str) -> list[SessionEvent]:
    """Consume a phase with explicit generator cleanup."""
    async with aclosing(session.run(prompt)) as events:
        return [event async for event in events]


def make_process(output: bytes, returncode: int) -> MagicMock:
    """Provide the asynchronous subprocess API without running a command."""
    process = create_autospec(Process, instance=True)
    process.returncode = returncode
    process.communicate.return_value = (output, SANDBOX_FAILURE)
    process.wait.return_value = returncode
    return process


def make_credential_pool(providers: tuple[ProviderConfig, ...]) -> CredentialPool:
    """Share production time and cancellable waits without repeating pool setup across integration fixtures."""
    return CredentialPool(providers, time, sleep)
