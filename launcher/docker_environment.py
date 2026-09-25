"""Own the batch's Docker image and network; supply fresh sandboxes until cleanup."""

from asyncio import sleep, timeout
from asyncio.subprocess import PIPE
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from time import time
from uuid import uuid4

from harness.proxy.credential_pool import CredentialPool
from harness.proxy.proxy import Proxy
from harness.sandbox.constants import CONTROL_TIMEOUT_SECONDS, PYTHON_EXECUTABLE
from harness.sandbox.docker import Docker
from harness.sandbox.models import SandboxConfig
from harness.sandbox.sandbox import Sandbox
from harness.utils.asyncio import finish_before_cancelling

from launcher.constants import (
    AUDIT_DOCKER_IMAGE,
    AUDIT_DOCKERFILE,
    AUDIT_WORKER_MODULE,
    DOCKER_IMAGE,
    IMPLEMENTATION_DIRECTORY,
    NETWORK_PREFIX,
    RUN_DOCKERFILE,
    WORKER_MODULE,
)
from launcher.models import ProviderRoute


@asynccontextmanager
async def docker_environment(docker: Docker, route: ProviderRoute, audit: bool) -> AsyncIterator[Callable[[], Sandbox]]:
    """Prepare Docker only for pending work and remove its network after all sandboxes have closed."""
    await build_images(docker, audit)
    network = f"{NETWORK_PREFIX}{uuid4().hex}"
    try:
        await finish_before_cancelling(_run_network_command(docker, "create", network), "Network creation")
        config = SandboxConfig(AUDIT_DOCKER_IMAGE if audit else DOCKER_IMAGE, network,
                               (PYTHON_EXECUTABLE, "-m", AUDIT_WORKER_MODULE if audit else WORKER_MODULE))
        pool = CredentialPool(route.providers, time, sleep)
        proxy_factory = partial(Proxy, pool, route.use_anthropic_resets)
        yield partial(Sandbox, config, docker, proxy_factory)
    except BaseException as error:
        try:
            await _run_network_command(docker, "rm", network)
        except Exception as cleanup_error:
            error.add_note(f"Network cleanup also failed: {cleanup_error}")
        raise
    else:
        await _run_network_command(docker, "rm", network)


async def build_images(docker: Docker, audit: bool) -> None:
    """Build solver and optional audit images using the same definitions during setup and launch."""
    images = [(DOCKER_IMAGE, RUN_DOCKERFILE)]
    if audit:
        images.append((AUDIT_DOCKER_IMAGE, AUDIT_DOCKERFILE))
    for image, dockerfile in images:
        await docker.run(["build", "-t", image, "-f", str(dockerfile),
                          str(IMPLEMENTATION_DIRECTORY)], None, PIPE)


async def _run_network_command(docker: Docker, action: str, network: str) -> None:
    """Bound creation or cleanup of the unique network owned by this invocation."""
    async with timeout(CONTROL_TIMEOUT_SECONDS):
        await docker.run(["network", action, network], None, PIPE)
