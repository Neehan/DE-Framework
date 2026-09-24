"""Local HTTP fixtures exercising the production gateway without a paid provider."""

from asyncio import CancelledError, Event
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientSession, web
from harness.proxy.constants import (
    API_KEY_ENV,
    PROXY_DOCKER_HOST,
    PROXY_EPHEMERAL_PORT,
    PROXY_PORT_INDEX,
)
from harness.proxy.credential_pool import CredentialPool
from harness.proxy.models import ProviderConfig
from harness.proxy.proxy import Proxy
from harness.sandbox.constants import PROVIDER_BASE_URL_ENV
from harness.utils.constants import INITIAL_COUNT

from tests.proxy.constants import (
    LIMIT_RESPONSES,
    LOOPBACK,
    POOL_NOW,
    SECOND_KEY,
    STREAM_CHUNKS,
    TEST_KEY,
)
from tests.support.helpers import make_credential_pool


@pytest.fixture
def upstream_cancelled() -> Event:
    """Expose cancellation of a streaming provider request without polling."""
    return Event()


@pytest.fixture
async def upstream(upstream_cancelled: Event) -> AsyncIterator[tuple[str, list[tuple[dict[str, Any], dict[str, str]]]]]:
    """Record accepted requests and serve success, redirect, failure, or interrupted streams."""
    calls: list[tuple[dict[str, Any], dict[str, str]]] = []

    async def respond(request: web.Request) -> web.StreamResponse:
        """Select deterministic transport behavior using the test model field."""
        body = await request.json()
        calls.append((body, {name.lower(): value for name, value in request.headers.items()}))
        if body["model"] in LIMIT_RESPONSES:
            status, headers = LIMIT_RESPONSES[body["model"]]
            return web.Response(status=status, headers=headers, text=TEST_KEY)
        if body["model"] == "redirect":
            raise web.HTTPTemporaryRedirect(location="http://127.0.0.1/forbidden")
        if body["model"] == "failure":
            return web.Response(status=HTTPStatus.TOO_MANY_REQUESTS, text=TEST_KEY)
        if body["model"] == "spent":
            return web.Response(status=HTTPStatus.BAD_REQUEST, text=f"Credit balance is too low: {TEST_KEY}")
        if not body.get("stream"):
            return web.json_response({"content": "proof"})
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in STREAM_CHUNKS:
            await response.write(chunk)
        if body["model"] == "slow":
            try:
                await Event().wait()
            except CancelledError:
                upstream_cancelled.set()
                raise
        if body["model"] == "disconnect":
            assert request.transport is not None
            request.transport.close()
        else:
            await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/{path:.*}", respond)
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    try:
        await web.TCPSite(runner, LOOPBACK, PROXY_EPHEMERAL_PORT).start()
        port = runner.addresses[INITIAL_COUNT][PROXY_PORT_INDEX]
        yield f"http://{LOOPBACK}:{port}", calls
    finally:
        await runner.cleanup()


@pytest.fixture
async def gateway(upstream: tuple[str, list[tuple[dict[str, Any], dict[str, str]]]]) -> AsyncIterator[tuple[str, str]]:
    """Use the real run-scoped proxy, accessing its listener from the test host."""
    url, _calls = upstream
    async with Proxy(make_credential_pool((ProviderConfig(url, API_KEY_ENV, TEST_KEY),)), True) as env:
        yield env[PROVIDER_BASE_URL_ENV].replace(PROXY_DOCKER_HOST, LOOPBACK), env[API_KEY_ENV]


@pytest.fixture
async def client() -> AsyncIterator[ClientSession]:
    """Close test HTTP connections even when an assertion fails."""
    async with ClientSession() as session:
        yield session


@pytest.fixture
def pool_clock() -> Mock:
    """Control credential reset time without waiting on wall-clock time."""
    return Mock(return_value=POOL_NOW)


@pytest.fixture
def pool_wait(pool_clock: Mock) -> AsyncMock:
    """Advance the injected clock by exactly the requested cooldown."""
    async def advance(seconds: float) -> None:
        """Make the earliest cooling credential eligible on the next selection."""
        pool_clock.return_value += seconds
    return AsyncMock(side_effect=advance)


@pytest.fixture
def providers() -> tuple[ProviderConfig, ...]:
    """Use two distinct credentials for one fixed provider."""
    return tuple(ProviderConfig("https://provider.example", API_KEY_ENV, key) for key in (TEST_KEY, SECOND_KEY))


@pytest.fixture
def pool(providers: tuple[ProviderConfig, ...], pool_clock: Mock, pool_wait: AsyncMock) -> CredentialPool:
    """Exercise the real shared pool with deterministic time and cancellation points."""
    return CredentialPool(providers, pool_clock, pool_wait)
