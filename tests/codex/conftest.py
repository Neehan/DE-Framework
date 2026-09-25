"""Injected Docker and HTTP dependencies for account lifecycle tests."""

from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from aiohttp import ClientSession
from codex.gateway import Gateway
from harness.sandbox.docker import Docker


@pytest.fixture
def docker_client() -> MagicMock:
    """Capture exact Docker operations without touching local accounts."""
    return create_autospec(Docker, instance=True)


@pytest.fixture
def gateway(docker_client: MagicMock) -> Gateway:
    """Serve immediate authenticated readiness with injectable I/O."""
    client = create_autospec(ClientSession, instance=True)
    response = client.get.return_value.__aenter__.return_value
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value={"data": [{"id": "gpt-*"}]})
    return Gateway(docker_client, client, AsyncMock())
