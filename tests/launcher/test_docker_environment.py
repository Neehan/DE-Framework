"""Network ownership during cancellation and setup failures."""

from asyncio import CancelledError, Event, create_task, timeout
from asyncio.subprocess import PIPE
from unittest.mock import create_autospec

import pytest
from harness.sandbox.docker import Docker
from launcher.docker_environment import docker_environment
from launcher.models import ProviderRoute

from tests.constants import ASYNC_TEST_TIMEOUT_SECONDS


async def test_cancelled_network_creation_finishes_before_cleanup(provider: ProviderRoute) -> None:
    """Do not race network removal against an in-flight Docker create."""
    creating, release = Event(), Event()
    calls = []

    async def run(arguments: list[str], content: bytes | None, stdout: int) -> bytes:
        """Hold only network creation while the owner is cancelled."""
        calls.append(arguments)
        if arguments[:2] == ["network", "create"]:
            creating.set()
            await release.wait()
        return b"owned-resource"

    docker = create_autospec(Docker, instance=True)
    docker.run.side_effect = run

    async def enter() -> None:
        """Enter the production network context under test-owned cancellation."""
        async with docker_environment(docker, provider, False):
            pytest.fail("cancelled setup must not launch attempts")

    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(enter())
        await creating.wait()
        task.cancel()
        release.set()
        with pytest.raises(CancelledError):
            await task
    build, create, remove = calls
    assert build[0] == "build"
    assert remove == ["network", "rm", create[-1]]


async def test_cleanup_failure_does_not_replace_batch_error(provider: ProviderRoute) -> None:
    """Retain the original failure when releasing its network also fails."""
    docker = create_autospec(Docker, instance=True)
    docker.run.side_effect = [b"image", b"network", OSError("cleanup failed")]
    failure = RuntimeError("batch interrupted")
    with pytest.raises(RuntimeError) as raised:
        async with docker_environment(docker, provider, False):
            raise failure
    assert raised.value is failure
    assert "Network cleanup also failed" in failure.__notes__[0]
    assert docker.run.await_args.args[-1] == PIPE
