"""Real CLI persistence and provider-only networking against local, disposable endpoints."""

import json
import os
from asyncio import TaskGroup, create_subprocess_exec, sleep, timeout
from asyncio.subprocess import Process
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any
from urllib.parse import urlsplit

import pytest
from harness.proxy.constants import (
    API_KEY_ENV,
    AUTH_TOKEN_ENV,
    OAUTH_TOKEN_ENV,
    PROXY_DOCKER_HOST,
)
from harness.proxy.models import ProviderConfig
from harness.proxy.proxy import Proxy
from harness.sandbox.constants import SUCCESS_EXIT_CODE
from harness.sandbox.docker import Docker
from harness.sandbox.models import SandboxConfig
from harness.sandbox.sandbox import Sandbox
from harness.session.constants import SESSION_FILENAME
from harness.session.run_lock import RunLock
from harness.utils.constants import MUSE_PREFIX
from launcher.constants import (
    LITELLM_KEY_ENV,
    LITELLM_URL_ENV,
    META_KEY_ENV,
    META_URL_ENV,
    WORKER_MODULE,
)
from launcher.launcher.run_launcher import RunLauncher
from launcher.models import LaunchConfig, Problem
from launcher.provider import resolve_provider

from tests.constants import FIRST_SOLUTION_FILENAME, ONE_ROUND, PROBLEM, TWO_ROUNDS
from tests.integration.constants import (
    BLOCKED_PORT,
    DENIED_TOOL_IDS,
    DOCKER_TIMEOUT_SECONDS,
    ENDPOINT_POLL_SECONDS,
    LAUNCHER_MODELS,
    LIMITED_KEYS,
    LOCAL_DATASET,
    NATIVE_SESSION_MODULE,
    NETWORK_MODULE,
    NETWORK_REPORT_FILENAME,
    PROVIDER_MODEL,
    PROVIDER_PORT,
    PROVIDER_RESPONSE,
    PUBLIC_HTTPS_ENDPOINT,
    SIBLING_SOLUTION,
    SPEND_LIMIT_MODEL,
    STREAM_RATE_LIMIT_MODEL,
    STREAM_SPEND_LIMIT_MODEL,
    TEST_UPSTREAM_KEY,
    TOOL_ERROR_ID,
    TOOL_IMAGE_ID,
    TOOL_STEPS,
)
from tests.integration.credential_failure_provider import CredentialFailureProvider
from tests.integration.docker import checkout_mounts, docker
from tests.integration.messages import iter_message_blocks
from tests.integration.provider_server import serve_provider
from tests.proxy.constants import SECOND_KEY
from tests.support.helpers import make_credential_pool

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("model", LAUNCHER_MODELS)
async def test_launcher_worker_skips_overlapping_and_completed_attempts(
    docker_image: str, provider_network: str, local_provider: tuple[str, list[dict[str, Any]]], tmp_path: Path,
    model: str,
) -> None:
    """Run real worker entrypoints, reject a competing owner, and rerun without any additional inference."""
    upstream_url, requests = local_provider
    config = LaunchConfig("aobench", "unaided", ONE_ROUND, model, [ONE_ROUND, TWO_ROUNDS], TWO_ROUNDS, None, None)
    route = resolve_provider(model, {
        "ANTHROPIC_BASE_URL": upstream_url, API_KEY_ENV: TEST_UPSTREAM_KEY,
        LITELLM_URL_ENV: upstream_url, LITELLM_KEY_ENV: TEST_UPSTREAM_KEY,
        META_URL_ENV: upstream_url, META_KEY_ENV: TEST_UPSTREAM_KEY,
    })
    sandbox_config = SandboxConfig(
        docker_image, provider_network,
        ("python", "-m", WORKER_MODULE),
    )
    proxy_factory = partial(Proxy, make_credential_pool(route.providers), route.use_anthropic_resets)
    factory = partial(Sandbox, sandbox_config, Docker(create_subprocess_exec), proxy_factory)
    environment = partial(nullcontext, factory)
    problems = [Problem("local-proof", PROBLEM, "algebra", None)]
    root = tmp_path / "results"
    first_directory = root / config.dataset / config.model_directory / config.experiment_directory / problems[0].problem_id / "seed_1"
    async with timeout(DOCKER_TIMEOUT_SECONDS), TaskGroup() as group:
        first = group.create_task(RunLauncher(config, environment, root).run(problems, route.model))
        while not (first_directory / SESSION_FILENAME).exists():
            if first.done():
                pytest.fail(f"worker exited before writing its initial checkpoint: {first.result()}")
            await sleep(ENDPOINT_POLL_SECONDS)
        overlapping = await RunLauncher(replace(config, seeds=[ONE_ROUND]), environment, root).run(problems, route.model)
        assert overlapping.skipped == ONE_ROUND and not overlapping.failed
        result = await first
        assert result.completed == TWO_ROUNDS and not result.failed
        request_count = len(requests)
        rerun = await RunLauncher(config, environment, root).run(problems, route.model)
        assert rerun.skipped == TWO_ROUNDS and not rerun.failed
        assert len(requests) == request_count
        assert all(request["model"] == route.model for request in requests)
        for request in requests:
            budget = request.get("output_config", {}).get("task_budget")
            if model.startswith(MUSE_PREFIX):
                assert budget is None
            else:
                assert budget is not None and budget["type"] == "tokens"
                assert budget["total"] <= config.budget_tokens
    assert (first_directory / FIRST_SOLUTION_FILENAME).read_text().strip()


def test_native_cli_checkpoint_and_independent_forks(docker_image: str) -> None:
    """The real CLI must restore relocated native history, fork independently, and archive debug links."""
    container = docker([
        "create", "--network", "none", "--entrypoint", "python", *checkout_mounts(),
        docker_image, "-m", NATIVE_SESSION_MODULE,
    ])
    try:
        docker(["start", "--attach", container])
        assert int(docker(["inspect", "--format", "{{.State.ExitCode}}", container])) == SUCCESS_EXIT_CODE
    finally:
        docker(["rm", "--force", container])


@pytest.mark.parametrize("auth_variable", [API_KEY_ENV, AUTH_TOKEN_ENV, OAUTH_TOKEN_ENV])
async def test_run_file_isolation_and_filtered_provider_requests(
    docker_image: str, provider_network: str, run_directory: Path,
    local_provider: tuple[str, list[dict[str, Any]]],
    auth_variable: str,
) -> None:
    """Keep credentials and reference files outside Docker; real CLI inference must pass the filtering gateway."""
    provider_ip = endpoint_address(provider_network, "provider")
    peer_ip = endpoint_address(provider_network, "forbidden-peer")
    blocked = [[provider_ip, BLOCKED_PORT], [peer_ip, PROVIDER_PORT], list(PUBLIC_HTTPS_ENDPOINT), ["::1", BLOCKED_PORT]]
    upstream_url, requests = local_provider
    blocked.extend([[provider_ip, PROVIDER_PORT], [PROXY_DOCKER_HOST, urlsplit(upstream_url).port]])

    async def start_process(*arguments: str, **kwargs: Any) -> Process:
        """Add only current checkout code and test support to the otherwise production Docker launch."""
        if "create" in arguments:
            image_index = arguments.index(docker_image)
            arguments = (*arguments[:image_index], *checkout_mounts(), *arguments[image_index:])
        return await create_subprocess_exec(*arguments, **kwargs)

    config = SandboxConfig(
        docker_image, provider_network,
        ("python", "-m", NETWORK_MODULE),
    )
    hidden_files = [str(run_directory.parent / path) for path in (LOCAL_DATASET, SIBLING_SOLUTION)]
    request = json.dumps({"uid": os.getuid(), "gid": os.getgid(), "blocked": blocked, "hidden_files": hidden_files})
    async with timeout(DOCKER_TIMEOUT_SECONDS):
        try:
            provider = ProviderConfig(upstream_url, auth_variable, TEST_UPSTREAM_KEY)
            proxy_factory = partial(Proxy, make_credential_pool((provider,)), True)
            await Sandbox(config, Docker(start_process), proxy_factory).run(run_directory, request, RunLock)
        except CalledProcessError as error:
            pytest.fail(error.stderr.decode())
    report = json.loads((run_directory / NETWORK_REPORT_FILENAME).read_text())
    assert report["blocked"] == blocked
    assert report["file_isolation_checked"]
    assert report["cli_version"]
    assert requests
    assert all("mcp_servers" not in body for body in requests)
    require_tool_results(requests)


def require_tool_results(requests: list[dict[str, Any]]) -> None:
    """Require actual CLI results at the upstream, including an inline image and a failed command."""
    results = {}
    documents = []
    for request in requests:
        for block in iter_message_blocks(request["messages"]):
            if block["type"] == "tool_result":
                results[block["tool_use_id"]] = block
            elif block["type"] == "document":
                documents.append(block)
    for tool_id, (name, _) in TOOL_STEPS.items():
        result = results[tool_id]
        assert result.get("is_error", False) == (tool_id == TOOL_ERROR_ID or tool_id in DENIED_TOOL_IDS), (name, result)
        if tool_id in DENIED_TOOL_IDS:
            assert "denied" in json.dumps(result).lower(), result
        assert "/root/.bashrc" not in json.dumps(result), result
    image_result = results[TOOL_IMAGE_ID]["content"]
    assert any(block["type"] == "image" and block["source"]["type"] == "base64" for block in image_result)
    assert any(block["source"]["type"] == "base64" and block["source"]["media_type"] == "application/pdf" for block in documents)


def endpoint_address(network: str, alias: str) -> str:
    """Resolve the locally created endpoint from Docker metadata before entering the sandbox."""
    return docker(["inspect", "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", f"{network}-{alias}"])


@pytest.mark.parametrize(("model", "limited_keys"), [
    (PROVIDER_MODEL, LIMITED_KEYS), (STREAM_RATE_LIMIT_MODEL, LIMITED_KEYS[:ONE_ROUND]),
    (SPEND_LIMIT_MODEL, LIMITED_KEYS[:ONE_ROUND]), (STREAM_SPEND_LIMIT_MODEL, LIMITED_KEYS[:ONE_ROUND]),
])
async def test_worker_recovers_from_http_and_streamed_credential_failures(
    docker_image: str, provider_network: str, tmp_path: Path, model: str, limited_keys: tuple[str, ...],
) -> None:
    """A real worker rotates temporary/permanent failures and resumes saved solve history on the healthy key."""
    with serve_provider(CredentialFailureProvider) as (url, requests):
        route = resolve_provider(model, {
            "ANTHROPIC_BASE_URL": url,
            **{f"{API_KEY_ENV}_{index}": key for index, key in enumerate((*limited_keys, SECOND_KEY), start=ONE_ROUND)},
        })
        config = LaunchConfig("aobench", "unaided", ONE_ROUND, model, [ONE_ROUND], ONE_ROUND, None, None)
        sandbox_config = SandboxConfig(docker_image, provider_network, ("python", "-m", WORKER_MODULE))
        proxy_factory = partial(Proxy, make_credential_pool(route.providers), route.use_anthropic_resets)
        factory = partial(Sandbox, sandbox_config, Docker(create_subprocess_exec), proxy_factory)
        async with timeout(DOCKER_TIMEOUT_SECONDS):
            result = await RunLauncher(config, partial(nullcontext, factory), tmp_path).run(
                [Problem("local-proof", PROBLEM, "algebra", None)], route.model,
            )
        assert result.completed == ONE_ROUND and not result.failed
        credentials = [request["credential"] for request in requests if "credential" in request]
        first_success = credentials.index(SECOND_KEY)
        assert set(credentials[:first_success]) == set(limited_keys)
        assert set(credentials[first_success:]) == {SECOND_KEY}
        if model in (SPEND_LIMIT_MODEL, STREAM_SPEND_LIMIT_MODEL):
            resumed = requests[requests.index({"credential": SECOND_KEY}) + ONE_ROUND]
            assert "Critically review" in json.dumps(resumed["messages"][-ONE_ROUND])
            assert PROVIDER_RESPONSE in json.dumps(resumed["messages"]).replace("\\n", "\n")
        solution, = tmp_path.rglob(FIRST_SOLUTION_FILENAME)
        assert solution.read_text().strip()
