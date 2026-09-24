"""Opt-in gateway lifecycle and full native tool execution through real LiteLLM."""

import json
import os
import socket
from asyncio import create_subprocess_exec, sleep, timeout
from asyncio.subprocess import Process
from base64 import urlsafe_b64encode
from collections.abc import AsyncIterator
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
from aiohttp import ClientSession, ClientTimeout
from audit.constants import RESULT_FILENAME, STEP_RECOGNITION
from audit.models import AuditRequest
from audit.registry import AUDITS
from codex.constants import BASE_PORT, HEALTH_TIMEOUT_SECONDS, HOST, IMAGE
from codex.gateway import Gateway
from codex.models import Account
from harness.proxy.constants import AUTH_TOKEN_ENV, PROXY_DOCKER_HOST
from harness.proxy.models import ProviderConfig
from harness.proxy.proxy import Proxy
from harness.sandbox.docker import Docker
from harness.sandbox.models import SandboxConfig
from harness.sandbox.sandbox import Sandbox
from harness.self_refine.models import RefinementState
from harness.session.constants import SESSION_FILENAME, WORKSPACE_DIRECTORY
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from launcher.launcher.run_launcher import RunLauncher
from launcher.models import LaunchConfig, Problem

from tests.audit.constants import (
    CORRECTNESS_RESULT,
    REFERENCE,
    STEP_RESULT,
    STEPS,
    SUBMISSION,
)
from tests.codex.constants import (
    AUTH_TOKENS,
    CODEX_TEST_IMAGE_ENV,
    FUTURE_EXPIRY,
    GATEWAY_KEY,
    TEST_MODEL,
    TOOL_CONTENT,
    TOOL_FILENAME,
)
from tests.constants import ONE_ROUND, PROBLEM
from tests.integration import provider_server
from tests.integration.constants import DOCKER_TIMEOUT_SECONDS, OUTPUT_TOKENS
from tests.integration.docker import docker
from tests.integration.provider_server import serve_provider
from tests.integration.responses_provider import ResponsesProvider
from tests.support.helpers import make_credential_pool


@pytest.fixture
async def codex_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[Account, list[dict[str, Any]]]]:
    """Create synthetic auth and an isolated account; clean up only the resources created here."""
    if os.environ.get(CODEX_TEST_IMAGE_ENV) != IMAGE:
        pytest.skip(f"set {CODEX_TEST_IMAGE_ENV}={IMAGE} after building the gateway image")
    with socket.socket() as listener:
        listener.bind((HOST, 0))
        account = Account(listener.getsockname()[1] - BASE_PORT)
    assert not docker(["volume", "ls", "--quiet", "--filter", f"name=^{account.volume}$"])
    assert not docker(["container", "ls", "--all", "--quiet", "--filter", f"name=^/{account.container}$"])
    monkeypatch.setattr(provider_server, "LOOPBACK_ADDRESS", "0.0.0.0")
    token = urlsafe_b64encode(json.dumps({"exp": FUTURE_EXPIRY}).encode()).decode().rstrip("=")
    source = tmp_path / "synthetic-auth.json"
    source.write_text(json.dumps({"tokens": {**AUTH_TOKENS, "access_token": f"e30.{token}.fake"}}))
    with serve_provider(ResponsesProvider) as (url, requests):
        async def start_process(*arguments: str, **kwargs: Any) -> Process:
            """Point only this test gateway at the fake backend, preserving production configuration."""
            if arguments[ONE_ROUND] == "create":
                index = arguments.index(IMAGE)
                arguments = (*arguments[:index], "--add-host", f"{PROXY_DOCKER_HOST}:host-gateway",
                             "--env", f"CHATGPT_API_BASE={url.replace('0.0.0.0', PROXY_DOCKER_HOST)}", *arguments[index:])
            return await create_subprocess_exec(*arguments, **kwargs)
        async with ClientSession(timeout=ClientTimeout(total=HEALTH_TIMEOUT_SECONDS)) as client:
            gateway = Gateway(Docker(start_process), client, sleep)
            try:
                await gateway.add(account, source)
                await gateway.add(account, source)
                await gateway.start([account], GATEWAY_KEY)
                await gateway.start([account], GATEWAY_KEY)
                await gateway.stop(account)
                await gateway.start([account], GATEWAY_KEY)
                yield account, requests
            finally:
                await gateway.stop(account)
                if docker(["container", "ls", "--all", "--quiet", "--filter", f"name=^/{account.container}$"]):
                    docker(["rm", "--force", account.container])
                if docker(["volume", "ls", "--quiet", "--filter", f"name=^{account.volume}$"]):
                    docker(["volume", "rm", account.volume])


@pytest.mark.integration
async def test_native_worker_through_codex_gateway(
    docker_image: str, provider_network: str, codex_gateway: tuple[Account, list[dict[str, Any]]], tmp_path: Path,
) -> None:
    """Check streaming, system blocks, real tools, usage, and saved workspace across every production layer."""
    account, requests = codex_gateway
    config = LaunchConfig("aobench", "unaided", ONE_ROUND, TEST_MODEL, [ONE_ROUND], ONE_ROUND, None, None)
    proxy = partial(Proxy, make_credential_pool((ProviderConfig(account.url, AUTH_TOKEN_ENV, GATEWAY_KEY),)), False)
    sandbox = partial(Sandbox, SandboxConfig(docker_image, provider_network, ("python", "-m", "launcher.worker")),
                      Docker(create_subprocess_exec), proxy)
    async with timeout(DOCKER_TIMEOUT_SECONDS):
        result = await RunLauncher(config, partial(nullcontext, sandbox), tmp_path / "results").run(
            [Problem("gateway-proof", PROBLEM, "algebra", None)], TEST_MODEL,
        )
    assert result.completed == ONE_ROUND and not result.failed
    saved, = (tmp_path / "results").rglob(SESSION_FILENAME)
    manager = SessionManager(saved.parent, RunLock)
    state = manager.prepare_attempt(RefinementState(PROBLEM, config.budget_tokens, None))
    assert state.output_tokens == len(requests) * OUTPUT_TOKENS
    with ZipFile(saved) as archive:
        assert archive.read(f"{WORKSPACE_DIRECTORY}/{TOOL_FILENAME}").decode() == TOOL_CONTENT
    assert all(request["model"] == TEST_MODEL for request in requests)
    assert all(request["reasoning"]["effort"] == "high" for request in requests)
    assert all(request.get("tools") for request in requests)
    assert any(item["type"] == "function_call_output" for request in requests for item in request["input"] if "type" in item)


@pytest.mark.integration
async def test_audit_structured_output_through_codex_gateway(
    docker_image: str, provider_network: str, codex_gateway: tuple[Account, list[dict[str, Any]]], tmp_path: Path,
) -> None:
    """Both audit contracts traverse the real LiteLLM Messages-to-Responses adapter and native CLI."""
    account, requests = codex_gateway
    proxy = partial(Proxy, make_credential_pool((ProviderConfig(account.url, AUTH_TOKEN_ENV, GATEWAY_KEY),)), False)
    sandbox = Sandbox(SandboxConfig(docker_image, provider_network, ("python", "-m", "audit.worker")), Docker(create_subprocess_exec), proxy)
    async with timeout(DOCKER_TIMEOUT_SECONDS):
        for name, expected in zip(AUDITS, (CORRECTNESS_RESULT, STEP_RESULT), strict=True):
            request = AuditRequest(name, TEST_MODEL, PROBLEM, REFERENCE, SUBMISSION, STEPS if name == STEP_RECOGNITION else None)
            directory = tmp_path / name
            await sandbox.run(directory, request.to_json(), RunLock)
            assert json.loads((directory / RESULT_FILENAME).read_text()) == expected
    assert all(request["model"] == TEST_MODEL for request in requests)
