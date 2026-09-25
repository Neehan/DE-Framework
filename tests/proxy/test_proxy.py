"""Actual HTTP checks for authentication, upstream isolation, streaming, and fail-closed forwarding."""

from asyncio import CancelledError, Event, create_task, timeout
from http import HTTPStatus
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientPayloadError, ClientSession
from harness.proxy.constants import (
    API_KEY_ENV,
    AUTH_ENV_HEADERS,
    AUTH_TOKEN_ENV,
    CREDENTIAL_COOLDOWN_SECONDS,
    OAUTH_TOKEN_ENV,
    PROXY_DOCKER_HOST,
)
from harness.proxy.credential_pool import CredentialPool
from harness.proxy.models import ProviderConfig
from harness.proxy.proxy import Proxy
from harness.sandbox.constants import PROVIDER_BASE_URL_ENV

from tests.constants import ASYNC_TEST_TIMEOUT_SECONDS, ONE_ROUND
from tests.proxy.constants import (
    LIMIT_RESPONSES,
    LOOPBACK,
    POOL_NOW,
    REQUEST_BODY,
    RESET_DELAY,
    SECOND_KEY,
    STREAM_CHUNKS,
    TEST_KEY,
)
from tests.support.helpers import make_credential_pool


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens?beta=true"])
async def test_forwarding_replaces_credentials_and_drops_arbitrary_headers(
    gateway: tuple[str, str], upstream: tuple[str, list[tuple[dict[str, Any], dict[str, str]]]], client: ClientSession, path: str,
) -> None:
    """Only the fixed upstream receives its real credential, regardless of attacker-supplied headers."""
    url, token = gateway
    async with client.post(url + path, json=REQUEST_BODY, headers={
        "x-api-key": token, "Authorization": "Bearer attacker-value", "X-Forwarded-Host": "forbidden.example",
        "Host": "forbidden.example", "anthropic-version": "attacker-version",
    }) as response:
        assert response.status == HTTPStatus.OK
        assert TEST_KEY not in await response.text()
    _url, calls = upstream
    body, headers = calls.pop()
    assert body == REQUEST_BODY
    assert headers["x-api-key"] == TEST_KEY
    assert "authorization" not in headers and "x-forwarded-host" not in headers
    assert headers["host"] != "forbidden.example"
    assert token != TEST_KEY and token not in headers.values()


@pytest.mark.parametrize(("method", "path", "headers", "body", "status"), [
    ("GET", "/v1/messages", {}, REQUEST_BODY, HTTPStatus.FORBIDDEN),
    ("POST", "/v1/files", {}, REQUEST_BODY, HTTPStatus.FORBIDDEN),
    ("POST", "/v1/messages?target=https://forbidden.example", {}, REQUEST_BODY, HTTPStatus.FORBIDDEN),
    ("POST", "/v1/messages", {"x-api-key": "another-run-token"}, REQUEST_BODY, HTTPStatus.UNAUTHORIZED),
    ("POST", "/v1/messages", {"anthropic-beta": "future-search"}, REQUEST_BODY, HTTPStatus.BAD_REQUEST),
    ("POST", "/v1/messages", {"Content-Encoding": "gzip"}, REQUEST_BODY, HTTPStatus.UNSUPPORTED_MEDIA_TYPE),
    ("POST", "/v1/messages", {}, {**REQUEST_BODY, "mcp_servers": []}, HTTPStatus.BAD_REQUEST),
    ("POST", "/v1/messages", {}, {**REQUEST_BODY, "tools": [{"type": "web_search_20250305", "name": "web_search"}]}, HTTPStatus.BAD_REQUEST),
])
async def test_rejected_requests_never_reach_upstream(
    gateway: tuple[str, str], upstream: tuple[str, list[tuple[dict[str, Any], dict[str, str]]]], client: ClientSession,
    method: str, path: str, headers: dict[str, str], body: dict[str, Any], status: HTTPStatus,
) -> None:
    """Reject paths, cross-run access, encodings, beta flags, and body capabilities before forwarding."""
    url, token = gateway
    async with client.request(method, url + path, headers={"x-api-key": token, **headers}, json=body) as response:
        assert response.status == status
    assert not upstream[ONE_ROUND]


async def test_streaming_bytes_are_preserved(gateway: tuple[str, str], client: ClientSession) -> None:
    """SSE framing reaches the SDK unchanged instead of becoming a buffered JSON response."""
    url, token = gateway
    async with client.post(url + "/v1/messages", headers={"x-api-key": token}, json={**REQUEST_BODY, "stream": True}) as response:
        assert response.content_type == "text/event-stream"
        assert await response.read() == b"".join(STREAM_CHUNKS)


async def test_broken_upstream_stream_stays_broken(gateway: tuple[str, str], client: ClientSession) -> None:
    """A truncated stream must fail at the client so Recovery cannot mistake it for completion."""
    url, token = gateway
    async with client.post(url + "/v1/messages", headers={"x-api-key": token}, json={**REQUEST_BODY, "model": "disconnect", "stream": True}) as response:
        with pytest.raises(ClientPayloadError):
            await response.read()


@pytest.mark.parametrize(("model", "status"), [("redirect", HTTPStatus.BAD_GATEWAY), ("failure", HTTPStatus.TOO_MANY_REQUESTS)])
async def test_redirects_and_provider_errors_do_not_leak_credentials(
    gateway: tuple[str, str], upstream: tuple[str, list[tuple[dict[str, Any], dict[str, str]]]], client: ClientSession,
    model: str, status: HTTPStatus,
) -> None:
    """Never follow provider redirects or return error bodies that may echo authentication headers."""
    url, token = gateway
    async with client.post(url + "/v1/messages", headers={"x-api-key": token}, json={**REQUEST_BODY, "model": model}) as response:
        assert response.status == status
        assert TEST_KEY not in await response.text()
        assert "Location" not in response.headers
    assert len(upstream[ONE_ROUND]) == ONE_ROUND


async def test_client_disconnect_cancels_upstream_stream(
    gateway: tuple[str, str], client: ClientSession, upstream_cancelled: Event,
) -> None:
    """The proxy neither buffers a live stream nor leaves inference running after its client disconnects."""
    url, token = gateway
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        async with client.post(url + "/v1/messages", headers={"x-api-key": token}, json={**REQUEST_BODY, "model": "slow", "stream": True}) as response:
            assert await response.content.readany()
        await upstream_cancelled.wait()


@pytest.mark.parametrize("variable", [AUTH_TOKEN_ENV, OAUTH_TOKEN_ENV])
async def test_bearer_and_oauth_modes_forward_only_the_upstream_credential(
    upstream: tuple[str, list[tuple[dict[str, Any], dict[str, str]]]], client: ClientSession, variable: str,
) -> None:
    """Keep the SDK authentication mode while replacing its disposable bearer token at the gateway."""
    url, calls = upstream
    async with Proxy(make_credential_pool((ProviderConfig(url, variable, TEST_KEY),)), True) as env:
        assert TEST_KEY not in env.values()
        proxy_url = env[PROVIDER_BASE_URL_ENV].replace(PROXY_DOCKER_HOST, LOOPBACK)
        async with client.post(proxy_url + "/v1/messages", headers={"Authorization": f"Bearer {env[variable]}"}, json=REQUEST_BODY) as response:
            assert response.status == HTTPStatus.OK
        _body, headers = calls.pop()
        assert headers[AUTH_ENV_HEADERS[variable]] == f"Bearer {TEST_KEY}"


@pytest.mark.parametrize(("variable", "header"), [(API_KEY_ENV, "x-api-key"), (AUTH_TOKEN_ENV, "authorization"), (OAUTH_TOKEN_ENV, "authorization")])
def test_supported_provider_authentication_stays_host_side(variable: str, header: str) -> None:
    """Anthropic keys, OAuth, and bearer-authenticated LiteLLM/Meta use the same gateway."""
    config = ProviderConfig("https://provider.example/anthropic", variable, TEST_KEY)
    assert config.url == "https://provider.example/anthropic"
    assert config.headers[header] == (TEST_KEY if variable == API_KEY_ENV else f"Bearer {TEST_KEY}")
    assert TEST_KEY not in repr(config)




@pytest.mark.parametrize(("model", "status", "spend_limited"), [
    ("failure", HTTPStatus.TOO_MANY_REQUESTS, False), ("spent", HTTPStatus.PAYMENT_REQUIRED, True),
])
async def test_credential_failure_rotates_shared_credentials_without_replaying_request(
    upstream: tuple[str, list[tuple[dict[str, Any], dict[str, str]]]], client: ClientSession,
    model: str, status: HTTPStatus, spend_limited: bool,
) -> None:
    """A run switches on its next request; another proxy sees the same cooldown and no real key enters Docker."""
    url, calls = upstream
    providers = tuple(ProviderConfig(url, API_KEY_ENV, key) for key in (TEST_KEY, SECOND_KEY))
    pool = make_credential_pool(providers)
    proxy = Proxy(pool, True)
    async with proxy as env:
        proxy_url = env[PROVIDER_BASE_URL_ENV].replace(PROXY_DOCKER_HOST, LOOPBACK)
        headers = {"x-api-key": env[API_KEY_ENV]}
        async with client.post(proxy_url + "/v1/messages", json={**REQUEST_BODY, "model": model}, headers=headers) as response:
            assert response.status == status
            assert TEST_KEY not in await response.text()
        assert len(calls) == ONE_ROUND
        async with client.post(proxy_url + "/v1/messages", json=REQUEST_BODY, headers=headers) as response:
            assert response.status == status
        assert len(calls) == ONE_ROUND
        await proxy.recover_credential(spend_limited, None)
        async with client.post(proxy_url + "/v1/messages", json=REQUEST_BODY, headers=headers) as response:
            assert response.status == HTTPStatus.OK
        async with Proxy(pool, True) as other_env:
            other_url = other_env[PROVIDER_BASE_URL_ENV].replace(PROXY_DOCKER_HOST, LOOPBACK)
            async with client.post(other_url + "/v1/messages", json=REQUEST_BODY, headers={"x-api-key": other_env[API_KEY_ENV]}) as response:
                assert response.status == HTTPStatus.OK
        assert [headers["x-api-key"] for _, headers in calls] == [TEST_KEY, SECOND_KEY, SECOND_KEY]
        assert env[API_KEY_ENV] not in (TEST_KEY, SECOND_KEY)


@pytest.mark.parametrize(("anthropic", "reset", "delay"), [
    (True, POOL_NOW + RESET_DELAY, RESET_DELAY),
    (False, POOL_NOW + RESET_DELAY, CREDENTIAL_COOLDOWN_SECONDS),
    (True, None, CREDENTIAL_COOLDOWN_SECONDS),
    (True, POOL_NOW, CREDENTIAL_COOLDOWN_SECONDS),
])
async def test_sdk_only_reset_controls_anthropic_cooldown(
    pool_clock: Mock, pool_wait: AsyncMock, anthropic: bool, reset: float | None, delay: float,
) -> None:
    """Forward worker reset information without an HTTP response; unsupported or absent resets use five minutes."""
    pool = CredentialPool((ProviderConfig(f"http://{LOOPBACK}", API_KEY_ENV, TEST_KEY),), pool_clock, pool_wait)
    proxy = Proxy(pool, anthropic)
    async with proxy:
        await proxy.recover_credential(False, reset)
    pool_wait.assert_awaited_once_with(delay)


@pytest.mark.parametrize(("model", "anthropic", "delay"), [
    ("worker-reported-limit", True, CREDENTIAL_COOLDOWN_SECONDS),
    ("limited-reset", True, RESET_DELAY), ("limited-delay", True, RESET_DELAY),
    ("limited-rejected", True, RESET_DELAY), ("limited-missing", True, CREDENTIAL_COOLDOWN_SECONDS),
    ("limited-invalid", True, CREDENTIAL_COOLDOWN_SECONDS), ("limited-past", True, CREDENTIAL_COOLDOWN_SECONDS),
    ("limited-delay", False, CREDENTIAL_COOLDOWN_SECONDS), ("limited-reset", False, CREDENTIAL_COOLDOWN_SECONDS),
    ("invalid-credential", True, None), ("unavailable", True, None),
])
async def test_provider_cooldown_policy_and_non_limit_failures(
    upstream: tuple[str, list[tuple[dict[str, Any], dict[str, str]]]], client: ClientSession,
    pool_clock: Mock, pool_wait: AsyncMock, monkeypatch: pytest.MonkeyPatch,
    model: str, anthropic: bool, delay: float | None,
) -> None:
    """Preserve provider resets, cool worker-only limits, and leave unrelated failures eligible."""
    url, calls = upstream
    monkeypatch.setattr("harness.proxy.proxy.time", pool_clock)
    pool = CredentialPool((ProviderConfig(url, API_KEY_ENV, TEST_KEY),), pool_clock, pool_wait)
    proxy = Proxy(pool, anthropic)
    async with proxy as env:
        proxy_url = env[PROVIDER_BASE_URL_ENV].replace(PROXY_DOCKER_HOST, LOOPBACK)
        headers = {"x-api-key": env[API_KEY_ENV]}
        async with client.post(proxy_url + "/v1/messages", json={**REQUEST_BODY, "model": model}, headers=headers) as response:
            assert response.status == LIMIT_RESPONSES[model][0]
        if delay is not None:
            await proxy.recover_credential(False, None)
        async with client.post(proxy_url + "/v1/messages", json=REQUEST_BODY, headers=headers) as response:
            assert response.status == HTTPStatus.OK
    if delay is None:
        pool_wait.assert_not_awaited()
    else:
        pool_wait.assert_awaited_once_with(delay)
    assert [headers["x-api-key"] for _, headers in calls] == [TEST_KEY, TEST_KEY]


async def test_cancelling_initial_cooldown_releases_proxy_lifecycle(
    pool: CredentialPool, providers: tuple[ProviderConfig, ...], pool_clock: Mock, pool_wait: AsyncMock,
) -> None:
    """Waiting for the first credential still owns startup; a second entry fails and cancellation permits reuse."""
    entered, blocked = Event(), Event()

    async def wait(seconds: float) -> None:
        """Keep initial acquisition pending until the proxy startup is cancelled."""
        entered.set()
        await blocked.wait()

    pool_wait.side_effect = wait
    for provider in providers:
        pool.mark_rate_limited(provider, None)
    proxy = Proxy(pool, True)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        starting = create_task(proxy.__aenter__())
        await entered.wait()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await proxy.__aenter__()
        finally:
            starting.cancel()
            with pytest.raises(CancelledError):
                await starting
        pool_clock.return_value += CREDENTIAL_COOLDOWN_SECONDS
        async with proxy as env:
            assert env[API_KEY_ENV] not in {provider.credential for provider in providers}
