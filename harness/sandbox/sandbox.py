"""Host-side lifecycle for one container with selected stdin input and one run directory."""

import os
from asyncio import timeout
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from subprocess import CalledProcessError
from uuid import uuid4

from harness.proxy.constants import PROXY_DOCKER_HOST
from harness.proxy.proxy import Proxy
from harness.sandbox.constants import (
    BOOTSTRAP_CAPABILITIES,
    BOOTSTRAP_MODULE,
    CONTAINER_NAME_PREFIX,
    CONTAINER_RUN_DIRECTORY,
    CONTROL_TIMEOUT_SECONDS,
    MISSING_CONTAINER_ERROR_PREFIX,
    PYTHON_EXECUTABLE,
    ROOT_USER_ID,
    RUN_DIRECTORY_LABEL,
    RUN_GID_ENV,
    RUN_UID_ENV,
    SUCCESS_EXIT_CODE,
)
from harness.sandbox.docker import Docker
from harness.sandbox.models import SandboxConfig
from harness.utils.asyncio import finish_before_cancelling
from harness.utils.constants import (
    RATE_LIMIT_EXIT_CODE,
    RATE_LIMIT_RESET_PREFIX,
    SPEND_LIMIT_EXIT_CODE,
    TEXT_ENCODING,
)


class Sandbox:
    """Run selected input behind a filtering proxy and clean up the gateway and container.

    Inject the Docker executor and proxy factory. Use an image containing bootstrap.py and firewall tools; no overrides are required.
    """

    def __init__(
        self, config: SandboxConfig, docker: Docker,
        proxy_factory: Callable[[], Proxy],
    ) -> None:
        """Accept container settings and an injectable Docker executor."""
        self._config = config
        self._docker = docker
        self._proxy_factory = proxy_factory
        self._running = False

    async def run(self, directory: Path, request: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Mount this run under RunLock, or injected nullcontext when the launcher already owns it."""
        if self._running:
            raise RuntimeError("sandbox is already running")
        if not request.strip():
            raise ValueError("sandbox request must not be empty")
        if os.getuid() == ROOT_USER_ID:
            raise PermissionError("sandbox must be launched by a non-root host user")
        directory = directory.resolve()
        if "," in str(directory):
            raise ValueError("sandbox directory must not contain a Docker mount separator")
        directory.mkdir(parents=True, exist_ok=True)
        self._running = True
        try:
            with ownership(directory):
                await self._remove_abandoned_containers(directory)
                await self._run_with_credentials(directory, request)
        finally:
            self._running = False

    async def _run_with_credentials(self, directory: Path, request: str) -> None:
        """Restart workers with eligible credentials after cleanup while retaining seed ownership."""
        proxy = self._proxy_factory()
        async with proxy as proxy_env:
            while True:
                exit_code, resets_at = await self._execute_container(directory, request, proxy_env)
                if exit_code == SUCCESS_EXIT_CODE:
                    return
                await proxy.recover_credential(exit_code == SPEND_LIMIT_EXIT_CODE, resets_at)

    async def _remove_abandoned_containers(self, directory: Path) -> None:
        """After acquiring host ownership, stop workers left behind by a killed launcher before restoring files."""
        async with timeout(CONTROL_TIMEOUT_SECONDS):
            containers = await self._docker.run([
                "ps", "--all", "--quiet", "--filter", f"label={RUN_DIRECTORY_LABEL}={directory}",
            ], None, PIPE)
        for container in containers.decode(TEXT_ENCODING).split():
            await self._remove_container(container)

    async def _execute_container(self, directory: Path, request: str, proxy_env: dict[str, str]) -> tuple[int, float | None]:
        """Clean up one worker and report whether it needs a credential change."""
        container = f"{CONTAINER_NAME_PREFIX}{uuid4()}"
        try:
            await finish_before_cancelling(self._create_container(directory, container, proxy_env), "Container creation")
            outcome = await self._start_container(container, request)
        except BaseException as error:
            try:
                await self._remove_container(container)
            except Exception as cleanup_error:
                error.add_note(f"Container cleanup also failed: {cleanup_error}")
            raise
        else:
            await self._remove_container(container)
            return outcome

    async def _start_container(self, container: str, request: str) -> tuple[int, float | None]:
        """Return success or an explicit credential-failure code; propagate every other worker failure."""
        try:
            await self._docker.run(["start", "--attach", "--interactive", container], request.encode(TEXT_ENCODING), DEVNULL)
        except CalledProcessError as error:
            if error.returncode == SPEND_LIMIT_EXIT_CODE:
                return error.returncode, None
            if error.returncode == RATE_LIMIT_EXIT_CODE:
                reset = error.stderr.decode(TEXT_ENCODING).strip().split("\n").pop()
                if not reset.startswith(RATE_LIMIT_RESET_PREFIX):
                    raise ValueError("rate-limited worker omitted its reset report") from error
                value = reset.removeprefix(RATE_LIMIT_RESET_PREFIX)
                return error.returncode, float(value) if value else None
            raise
        return SUCCESS_EXIT_CODE, None

    async def _create_container(self, directory: Path, container: str, proxy_env: dict[str, str]) -> None:
        """Grant only bootstrap capabilities; the entrypoint removes them before running the agent."""
        command = [
            "create", "--interactive", "--init", "--name", container,
            "--network", self._config.network, "--cap-drop", "ALL",
            "--label", f"{RUN_DIRECTORY_LABEL}={directory}",
            "--add-host", f"{PROXY_DOCKER_HOST}:host-gateway",
            "--security-opt", "no-new-privileges", "--user", f"{ROOT_USER_ID}:{ROOT_USER_ID}",
            "--entrypoint", PYTHON_EXECUTABLE,
            "--env", f"{RUN_UID_ENV}={os.getuid()}", "--env", f"{RUN_GID_ENV}={os.getgid()}",
            "--env", f"HOME={CONTAINER_RUN_DIRECTORY}",
            "--log-driver", "none", "--tmpfs", "/tmp:rw,nosuid,nodev",
            "--mount", f"type=bind,source={directory},target={CONTAINER_RUN_DIRECTORY}",
        ]
        for capability in BOOTSTRAP_CAPABILITIES:
            command.extend(["--cap-add", capability])
        for name, value in proxy_env.items():
            command.extend(["--env", f"{name}={value}"])
        command.extend([self._config.image, "-m", BOOTSTRAP_MODULE, *self._config.command])
        async with timeout(CONTROL_TIMEOUT_SECONDS):
            output = await self._docker.run(command, None, PIPE)
        if not output.decode(TEXT_ENCODING).strip():
            raise RuntimeError("Docker create returned no container ID")

    async def _remove_container(self, container: str) -> None:
        """Remove this run's unique name, tolerating only explicit absence of that container."""
        try:
            async with timeout(CONTROL_TIMEOUT_SECONDS):
                await self._docker.run(["rm", "--force", container], None, DEVNULL)
        except CalledProcessError as error:
            if error.stderr.decode(TEXT_ENCODING).strip() != f"{MISSING_CONTAINER_ERROR_PREFIX}{container}":
                raise
