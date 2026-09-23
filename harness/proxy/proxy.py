"""Run-scoped filtering gateway; provider credentials never enter the agent container."""

from collections.abc import Mapping
from http import HTTPStatus
from math import isfinite
from secrets import compare_digest, token_urlsafe
from time import time
from types import TracebackType
from typing import Any

from aiohttp import (
    ClientError,
    ClientResponse,
    ClientSession,
    ClientTimeout,
    DummyCookieJar,
    web,
)
from jsonschema import ValidationError

from harness.proxy.constants import (
    ANTHROPIC_LIMIT_REJECTED,
    ANTHROPIC_LIMIT_STATUS_HEADER,
    ANTHROPIC_RESET_HEADER,
    INFERENCE_PATHS,
    INFERENCE_QUERY,
    PROXY_BIND_HOST,
    PROXY_CHUNK_BYTES,
    PROXY_CONNECT_TIMEOUT_SECONDS,
    PROXY_DOCKER_HOST,
    PROXY_EPHEMERAL_PORT,
    PROXY_MAX_REQUEST_BYTES,
    PROXY_PORT_INDEX,
    PROXY_SHUTDOWN_TIMEOUT_SECONDS,
    PROXY_TOKEN_BYTES,
    RETRY_AFTER_HEADER,
)
from harness.proxy.credential_pool import CredentialPool
from harness.proxy.models import ProviderConfig
from harness.proxy.request_policy import RequestPolicy
from harness.sandbox.constants import PROVIDER_BASE_URL_ENV
from harness.session.constants import SPEND_LIMIT_MESSAGE
from harness.session.errors import is_spend_limit_error
from harness.utils.constants import INITIAL_COUNT, PROVIDER_TIMEOUT_SECONDS


class Proxy:
    """Use as an async context manager returning container-safe SDK environment settings.

    The launcher owns this server and fixed upstream. Only reviewed Messages requests are forwarded; no overrides are required.
    """

    def __init__(self, pool: CredentialPool, use_anthropic_resets: bool) -> None:
        """Share the batch pool while keeping credential selection and disposable authentication local to this run."""
        self._pool = pool
        self._use_anthropic_resets = use_anthropic_resets
        self._provider: ProviderConfig | None = None
        self._token = token_urlsafe(PROXY_TOKEN_BYTES)
        self._policy = RequestPolicy()
        self._runner: web.AppRunner | None = None
        self._client: ClientSession | None = None

    async def __aenter__(self) -> dict[str, str]:
        """Start the gateway before Docker, returning only its address and disposable authentication."""
        if self._runner is not None:
            raise RuntimeError("proxy is already running")
        app = web.Application(client_max_size=PROXY_MAX_REQUEST_BYTES)
        app.router.add_route("*", "/{path:.*}", self._forward_request)
        self._runner = web.AppRunner(
            app, access_log=None, handler_cancellation=True, auto_decompress=False,
            shutdown_timeout=PROXY_SHUTDOWN_TIMEOUT_SECONDS,
        )
        try:
            self._provider = await self._pool.acquire(None)
            assert self._provider is not None
            self._client = ClientSession(
                timeout=ClientTimeout(total=None, sock_connect=PROXY_CONNECT_TIMEOUT_SECONDS,
                                      sock_read=PROVIDER_TIMEOUT_SECONDS),
                cookie_jar=DummyCookieJar(), trust_env=False,
            )
            await self._runner.setup()
            await web.TCPSite(self._runner, PROXY_BIND_HOST, PROXY_EPHEMERAL_PORT).start()
            port = self._runner.addresses[INITIAL_COUNT][PROXY_PORT_INDEX]
            return {PROVIDER_BASE_URL_ENV: f"http://{PROXY_DOCKER_HOST}:{port}", self._provider.auth_variable: self._token}
        except BaseException as error:
            await self.__aexit__(type(error), error, error.__traceback__)
            raise

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Stop accepting requests and release active upstream connections after the run ends."""
        try:
            await self.close()
        except Exception as error:
            if exc is None:
                raise
            exc.add_note(f"Proxy cleanup also failed: {error}")

    async def recover_credential(self, spend_limited: bool, resets_at: float | None) -> None:
        """Disable exhausted keys or cool temporary limits, then await an eligible replacement."""
        assert self._provider is not None
        if spend_limited:
            self._pool.disable(self._provider)
        elif self._pool.is_available(self._provider) or (self._use_anthropic_resets and resets_at is not None):
            self._pool.mark_rate_limited(self._provider, resets_at if self._use_anthropic_resets else None)
        self._provider = await self._pool.acquire(self._provider)

    async def close(self) -> None:
        """Close partially started or active gateway resources exactly once."""
        try:
            if self._runner is not None:
                await self._runner.cleanup()
        finally:
            self._runner = None
            if self._client is not None:
                await self._client.close()
                self._client = None

    async def _forward_request(self, request: web.Request) -> web.StreamResponse:
        """Authenticate, validate, then forward; rejected requests never reach the provider."""
        self._validate_http_request(request)
        assert self._provider is not None
        provider = self._provider
        headers = {**provider.headers, "content-type": "application/json"}
        try:
            body = self._policy.parse(await request.read())
            betas = self._policy.validate_betas(",".join([
                headers.pop("anthropic-beta", ""), request.headers.get("anthropic-beta", ""),
            ]))
        except (ValueError, ValidationError, RecursionError):
            raise web.HTTPBadRequest(text="request contains invalid or unapproved inference features") from None
        if betas:
            headers["anthropic-beta"] = betas
        if self._pool.is_disabled(provider):
            raise web.HTTPPaymentRequired(text=SPEND_LIMIT_MESSAGE)
        if not self._pool.is_available(provider):
            raise web.HTTPTooManyRequests(text="provider credential cooling")
        return await self._stream_provider_response(request, body, headers, provider)

    async def _stream_provider_response(
        self, request: web.Request, body: dict[str, Any], headers: dict[str, str], provider: ProviderConfig,
    ) -> web.StreamResponse:
        """Forward approved input to the fixed provider and preserve streaming failures for recovery."""
        assert self._client is not None
        target = f"{provider.url}{request.rel_url}"
        response = web.StreamResponse()
        try:
            async with self._client.post(target, json=body, headers=headers, allow_redirects=False) as upstream:
                if upstream.status == HTTPStatus.TOO_MANY_REQUESTS or (
                    self._use_anthropic_resets
                    and upstream.headers.get(ANTHROPIC_LIMIT_STATUS_HEADER) == ANTHROPIC_LIMIT_REJECTED
                ):
                    self._pool.mark_rate_limited(provider, self._get_reset_time(upstream.headers))
                if upstream.status != HTTPStatus.OK:
                    return await self._get_error_response(upstream, provider)
                response.content_type = upstream.content_type
                await response.prepare(request)
                async for chunk in upstream.content.iter_chunked(PROXY_CHUNK_BYTES):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (ClientError, TimeoutError):
            if response.prepared:
                if request.transport is not None:
                    request.transport.close()
                return response
            raise web.HTTPBadGateway(text="provider connection failed") from None

    async def _get_error_response(self, upstream: ClientResponse, provider: ProviderConfig) -> web.Response:
        """Recognize exhausted credentials while keeping raw provider errors and possible secrets host-side."""
        if upstream.status >= HTTPStatus.BAD_REQUEST and is_spend_limit_error(await upstream.text()):
            self._pool.disable(provider)
            return web.Response(status=HTTPStatus.PAYMENT_REQUIRED, text=SPEND_LIMIT_MESSAGE)
        status = upstream.status if upstream.status >= HTTPStatus.BAD_REQUEST else HTTPStatus.BAD_GATEWAY
        return web.Response(status=status, text="provider request failed")

    def _validate_http_request(self, request: web.Request) -> None:
        """Reject authentication, route, query, and encoding tricks before reading the request body."""
        assert self._provider is not None
        header, expected = self._provider.get_auth_header(self._token)
        if not compare_digest(request.headers.get(header, "").encode(), expected.encode()):
            raise web.HTTPUnauthorized(text="invalid run token")
        if request.method != "POST" or request.raw_path.split("?")[INITIAL_COUNT] not in INFERENCE_PATHS:
            raise web.HTTPForbidden(text="only approved inference endpoints are available")
        if request.query_string not in {"", INFERENCE_QUERY}:
            raise web.HTTPForbidden(text="unapproved query parameters")
        if request.content_type != "application/json" or "Content-Encoding" in request.headers:
            raise web.HTTPUnsupportedMediaType(text="uncompressed JSON required")

    def _get_reset_time(self, headers: Mapping[str, str]) -> float | None:
        """Read Anthropic's subscription reset or API retry delay; other providers use the pool's five minutes."""
        if not self._use_anthropic_resets:
            return None
        try:
            if ANTHROPIC_RESET_HEADER in headers:
                return float(headers[ANTHROPIC_RESET_HEADER])
            if RETRY_AFTER_HEADER in headers:
                delay = float(headers[RETRY_AFTER_HEADER])
                if isfinite(delay) and delay > INITIAL_COUNT:
                    return time() + delay
        except ValueError:
            return None
        return None
