"""Execute Docker commands and reap their client processes on cancellation."""

import os
from asyncio.subprocess import PIPE, Process
from collections.abc import Awaitable, Callable
from subprocess import CalledProcessError

from harness.sandbox.constants import DOCKER_EXECUTABLE, SUCCESS_EXIT_CODE


class Docker:
    """Run argument lists through an injected process launcher; no shell or overrides are required."""

    def __init__(self, start_process: Callable[..., Awaitable[Process]]) -> None:
        """Accept the production subprocess factory or a test double."""
        self._start_process = start_process

    async def run(self, arguments: list[str], content: bytes | None, stdout: int) -> bytes:
        """Raise on failure and terminate/reap the Docker client if its caller is cancelled."""
        command = [DOCKER_EXECUTABLE, *arguments]
        process = await self._start_process(*command, stdin=PIPE, stdout=stdout, stderr=PIPE, env=dict(os.environ))
        try:
            output, errors = await process.communicate(content)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.communicate()
            raise
        returncode = await process.wait()
        if returncode != SUCCESS_EXIT_CODE:
            raise CalledProcessError(returncode, command, output, errors)
        return output if stdout == PIPE else b""
