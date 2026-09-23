"""Native audit worker, structured-output proxy compatibility, and independent-stage resumption."""

import json
from asyncio import create_subprocess_exec, timeout
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import pytest
from audit.constants import AUDIT_FILENAME, AUDIT_WORKSPACE_DIRECTORY
from audit.models import AuditReference
from harness.proxy.constants import API_KEY_ENV
from harness.proxy.proxy import Proxy
from harness.sandbox.docker import Docker
from harness.sandbox.models import SandboxConfig
from harness.sandbox.sandbox import Sandbox
from harness.self_refine.models import RefinementState, RunStatus
from harness.session.constants import SESSION_FILENAME
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from launcher.constants import AUDIT_WORKER_MODULE
from launcher.launcher.audit_launcher import AuditLauncher
from launcher.models import AuditConfig, Problem
from launcher.provider import resolve_provider

from tests.audit.constants import (
    AUDIT_MODEL,
    CORRECTNESS_RESULT,
    REFERENCE,
    STEP_RESULT,
    STEPS,
    SUBMISSION,
)
from tests.constants import ONE_ROUND, PROBLEM, TOTAL_UNITS
from tests.integration.audit_provider import AuditProvider
from tests.integration.constants import DOCKER_TIMEOUT_SECONDS, TEST_UPSTREAM_KEY
from tests.integration.provider_server import serve_provider
from tests.support.helpers import make_credential_pool

pytestmark = pytest.mark.integration


async def test_real_audit_sequence_and_completed_skip(docker_image: str, provider_network: str, tmp_path: Path) -> None:
    """Score zero still receives independent step recognition; saved results skip all provider work on rerun."""
    config = AuditConfig("aobench", "unaided", TOTAL_UNITS, "claude-test", [ONE_ROUND], ONE_ROUND, None, None, f"litellm/{AUDIT_MODEL}")
    problem = Problem("local-proof", PROBLEM, "algebra", None)
    source = tmp_path / config.dataset / config.model_directory / config.experiment_directory / problem.problem_id / "seed_1"
    manager = SessionManager(source, RunLock)
    state = RefinementState(PROBLEM, config.budget_tokens, None)
    manager.open(state)
    state.solution, state.status = SUBMISSION, RunStatus.FINISHED
    state.solution_checkpoints = {key: SUBMISSION for key in range(ONE_ROUND, TOTAL_UNITS + ONE_ROUND)}
    manager.checkpoint()
    manager.close()
    archive = (source / SESSION_FILENAME).read_bytes()
    with serve_provider(AuditProvider) as (url, requests):
        route = resolve_provider("claude-audit-test", {"ANTHROPIC_BASE_URL": url, API_KEY_ENV: TEST_UPSTREAM_KEY})
        sandbox = SandboxConfig(docker_image, provider_network, ("python", "-m", AUDIT_WORKER_MODULE))
        factory = partial(Sandbox, sandbox, Docker(create_subprocess_exec), partial(Proxy, make_credential_pool(route.providers), True))
        launcher = AuditLauncher(config, partial(nullcontext, factory), tmp_path, {problem.problem_id: AuditReference(REFERENCE, STEPS)})
        async with timeout(DOCKER_TIMEOUT_SECONDS):
            result = await launcher.run([problem], route.model)
            assert result.completed == ONE_ROUND and not result.failed
            count = len(requests)
            repeated = await launcher.run([problem], route.model)
        assert repeated.skipped == ONE_ROUND and len(requests) == count
    checkpoints = json.loads((source / AUDIT_FILENAME).read_text())["checkpoints"]
    assert len(checkpoints) == TOTAL_UNITS
    assert all(verdict == checkpoints["1x"] for verdict in checkpoints.values())
    assert checkpoints["1x"] == {
        "correctness": CORRECTNESS_RESULT, "correctness_audit_model": config.audit_model,
        "step_recognition": STEP_RESULT, "step_audit_model": config.audit_model,
    }
    assert (source / SESSION_FILENAME).read_bytes() == archive
    assert not (source / AUDIT_WORKSPACE_DIRECTORY).exists()
    prompts = [json.dumps(request["messages"]) for request in requests]
    assert any(STEPS[0] in prompt for prompt in prompts)
    assert all(CORRECTNESS_RESULT["note"] not in prompt for prompt in prompts)
