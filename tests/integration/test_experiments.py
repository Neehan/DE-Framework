"""Real launcher and native-session checks for sketch interventions and shared continuation prefixes."""

import json
from asyncio import create_subprocess_exec, timeout
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
from experiments.constants import (
    CONTINUATION_TOTAL_MULTIPLIER,
    CONTINUE,
    CONTINUE_ORACLE,
    PREFIX_DIRECTORY,
    PREFIX_TOKENS,
)
from harness.proxy.constants import API_KEY_ENV
from harness.proxy.models import ProviderConfig
from harness.proxy.proxy import Proxy
from harness.sandbox.docker import Docker
from harness.sandbox.models import SandboxConfig
from harness.sandbox.sandbox import Sandbox
from harness.self_refine.models import RunStatus
from harness.session.constants import (
    CHECKPOINT_FILENAME,
    SESSION_FILENAME,
)
from harness.session.models import SessionState
from harness.utils.constants import (
    DEFAULT_MIN_NO_GAP_CRITIQUES,
    OUTPUT_TOKENS_PER_BLOCK,
    TEXT_ENCODING,
)
from launcher.constants import WORKER_MODULE
from launcher.launcher.continuation_launcher import ContinuationLauncher
from launcher.models import LaunchConfig, Problem

from tests.constants import FIRST_SOLUTION_FILENAME, ONE_ROUND, PROBLEM
from tests.integration.constants import (
    DOCKER_TIMEOUT_SECONDS,
    PREFIX_OVERRUN_RESPONSE,
    PROVIDER_MODEL,
    TEST_UPSTREAM_KEY,
)
from tests.integration.local_provider import LocalProvider
from tests.integration.prefix_overrun_provider import PrefixOverrunProvider
from tests.integration.provider_server import serve_provider
from tests.launcher.constants import REFERENCE_SKETCH
from tests.support.helpers import make_credential_pool

pytestmark = pytest.mark.integration


@pytest.fixture(params=[LocalProvider, PrefixOverrunProvider])
def late_provider(request: pytest.FixtureRequest) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Exercise natural prefix completion and actual CLI interruption using only local scripted inference."""
    with serve_provider(request.param) as provider:
        yield provider


async def test_late_arms_reuse_native_prefix_and_fork_independent_conversations(
    docker_image: str, provider_network: str, late_provider: tuple[str, list[dict[str, Any]]], tmp_path: Path,
) -> None:
    """Both real workers preserve one prefix, receive only permitted input, and skip completed reruns."""
    url, requests = late_provider
    config = LaunchConfig("aobench", CONTINUE, CONTINUATION_TOTAL_MULTIPLIER, PROVIDER_MODEL, [ONE_ROUND], ONE_ROUND, None, None)
    sandbox_config = SandboxConfig(docker_image, provider_network, ("python", "-m", WORKER_MODULE))
    providers = (ProviderConfig(url, API_KEY_ENV, TEST_UPSTREAM_KEY),)
    proxy = partial(Proxy, make_credential_pool(providers), True)
    environment = partial(nullcontext, partial(Sandbox, sandbox_config, Docker(create_subprocess_exec), proxy))
    problem = Problem("local-proof", PROBLEM, "algebra", None)
    model_root = tmp_path / config.dataset / config.model_directory
    prefix_path = model_root / PREFIX_DIRECTORY / problem.problem_id / "seed_1" / SESSION_FILENAME
    branches = []
    original: bytes | None = None
    async with timeout(DOCKER_TIMEOUT_SECONDS):
        for experiment in (CONTINUE, CONTINUE_ORACLE):
            selected = replace(config, experiment=experiment)
            selected_problem = replace(problem, sketch=REFERENCE_SKETCH if experiment == CONTINUE_ORACLE else None)
            launcher = ContinuationLauncher(selected, environment, tmp_path)
            result = await launcher.run([selected_problem], PROVIDER_MODEL)
            assert result.completed == ONE_ROUND and not result.failed
            branches.append(read_session(model_root / selected.experiment_directory / problem.problem_id / "seed_1" / SESSION_FILENAME))
            if experiment == CONTINUE:
                original = prefix_path.read_bytes()
                assert REFERENCE_SKETCH not in json.dumps(requests)
            else:
                assert prefix_path.read_bytes() == original
                assert REFERENCE_SKETCH in json.dumps(requests[-ONE_ROUND])
            count = len(requests)
            rerun = await launcher.run([selected_problem], PROVIDER_MODEL)
            assert rerun.skipped == ONE_ROUND and not rerun.failed
            assert len(requests) == count
    assert str(config.budget_tokens) in json.dumps(requests[0]["messages"])
    assert PREFIX_OVERRUN_RESPONSE not in json.dumps(requests)
    assert all(path.read_text().strip() for path in model_root.rglob(FIRST_SOLUTION_FILENAME))
    _check_retained_prefix(read_session(prefix_path), branches)


def _check_retained_prefix(prefix: SessionState, branches: list[SessionState]) -> None:
    """Require native fork independence, unchanged 1x–3x proofs, and one additional block after retained prefix usage."""
    assert prefix.refinement.output_tokens <= PREFIX_TOKENS
    assert len({prefix.session_id, *(branch.session_id for branch in branches)}) == len(branches) + ONE_ROUND
    assert len(prefix.refinement.solution_checkpoints) == prefix.refinement.checkpoint_count == CONTINUATION_TOTAL_MULTIPLIER - ONE_ROUND
    for branch in branches:
        assert branch.refinement.budget_tokens == prefix.refinement.output_tokens + OUTPUT_TOKENS_PER_BLOCK
        assert branch.refinement.status == RunStatus.FINISHED and not branch.fork_session
        assert branch.refinement.rounds == prefix.refinement.rounds + DEFAULT_MIN_NO_GAP_CRITIQUES
        assert branch.refinement.checkpoint_count == CONTINUATION_TOTAL_MULTIPLIER
        assert len(branch.refinement.solution_checkpoints) == CONTINUATION_TOTAL_MULTIPLIER
        for multiplier, solution in prefix.refinement.solution_checkpoints.items():
            assert branch.refinement.solution_checkpoints[multiplier] == solution


def read_session(path: Path) -> SessionState:
    """Read durable session metadata without restoring or modifying either native conversation."""
    with ZipFile(path) as archive:
        return SessionState.from_json(archive.read(CHECKPOINT_FILENAME).decode(TEXT_ENCODING))
