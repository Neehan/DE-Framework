"""Injected Docker process fixtures for sandbox lifecycle tests."""

from unittest.mock import AsyncMock, Mock

import pytest
from harness.sandbox.constants import SUCCESS_EXIT_CODE
from harness.sandbox.docker import Docker
from harness.sandbox.models import SandboxConfig
from harness.sandbox.sandbox import Sandbox
from harness.utils.constants import TEXT_ENCODING

from tests.constants import (
    SANDBOX_COMMAND,
    SANDBOX_CONTAINER_ID,
    SANDBOX_IMAGE,
    SANDBOX_NETWORK,
    SANDBOX_PROXY_ENV,
)
from tests.support.helpers import make_process
from tests.support.models import SandboxProcesses


@pytest.fixture
def processes() -> SandboxProcesses:
    """Create the four independently controllable Docker subprocess doubles."""
    return SandboxProcesses(
        make_process(b"", SUCCESS_EXIT_CODE),
        make_process(SANDBOX_CONTAINER_ID.encode(TEXT_ENCODING), SUCCESS_EXIT_CODE),
        make_process(b"", SUCCESS_EXIT_CODE),
        make_process(b"", SUCCESS_EXIT_CODE),
    )


@pytest.fixture
def start_process(processes: SandboxProcesses) -> AsyncMock:
    """Return the injected Docker processes in creation, execution, and removal order."""
    return AsyncMock(side_effect=[processes.inspection, processes.creation, processes.execution, processes.removal])


@pytest.fixture
def proxy_factory() -> Mock:
    """Replace only the network server while checking gateway lifetime around Docker."""
    manager = AsyncMock()
    manager.__aenter__.return_value = SANDBOX_PROXY_ENV
    return Mock(return_value=manager)


@pytest.fixture
def sandbox(start_process: AsyncMock, proxy_factory: Mock) -> Sandbox:
    """Exercise the real wrapper without a Docker daemon."""
    return Sandbox(SandboxConfig(SANDBOX_IMAGE, SANDBOX_NETWORK, SANDBOX_COMMAND), Docker(start_process), proxy_factory)
