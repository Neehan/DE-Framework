"""Bounded scheduling, independent failures, cancellation, and completed-run skipping."""

import json
from asyncio import CancelledError, Event, create_task, timeout
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, create_autospec, patch

import pytest
from harness.sandbox.sandbox import Sandbox
from harness.self_refine.models import RefinementState, RunStatus
from harness.session.constants import (
    RUNTIME_DIRECTORY,
    SESSION_FILENAME,
)
from harness.session.errors import AttemptAlreadyRunning
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import COUNT_INCREMENT, INITIAL_COUNT
from launcher.launcher.run_launcher import RunLauncher
from launcher.models import LaunchConfig, Problem

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    INITIAL_SOLUTION,
    ONE_ROUND,
    TOTAL_UNITS,
)
from tests.launcher.constants import REFERENCE_SENTINEL, TEST_SEEDS, TWO_WORKERS


async def test_worker_limit_and_one_failure_does_not_cancel_others(
    launch_config: LaunchConfig, problems: list[Problem], tmp_path: Path,
) -> None:
    """Admit exactly the allowed workers, then continue the remaining seed after a failure."""
    active = INITIAL_COUNT
    started = Event()
    release = Event()
    config = replace(launch_config, seeds=TEST_SEEDS, max_concurrency=TWO_WORKERS)
    seen = []

    async def run(directory: Path, request: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Hold initial workers so a scheduler that starts too many cannot pass."""
        nonlocal active
        seen.append(directory)
        active += COUNT_INCREMENT
        assert active <= TWO_WORKERS
        if active == TWO_WORKERS:
            started.set()
        try:
            assert REFERENCE_SENTINEL not in request
            assert set(json.loads(request)) == {"experiment", "model", "problem", "budget_tokens", "sketch", "pause_at_tokens"}
            await release.wait()
            if directory.name == "seed_1":
                raise RuntimeError("one seed failed")
        finally:
            active -= COUNT_INCREMENT

    sandbox = create_autospec(Sandbox, instance=True)
    sandbox.run.side_effect = run
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(RunLauncher(config, lambda: nullcontext(lambda: sandbox), tmp_path).run(problems[:ONE_ROUND], "provider-alias"))
        await started.wait()
        assert len(seen) == TWO_WORKERS
        release.set()
        result = await task
    assert result.failed == ONE_ROUND and result.completed == TWO_WORKERS
    assert all(path.parent.parent.name == "unaided-1x" for path in seen)


@pytest.mark.parametrize("solution", [INITIAL_SOLUTION, None])
async def test_completed_checkpoint_repairs_export_without_preparing_docker(
    launch_config: LaunchConfig, problems: list[Problem], sandbox_environment: Mock, tmp_path: Path,
    solution: str | None,
) -> None:
    """An interruption after archive commit must not leave a stale export, including exhausted empty results."""
    problem = problems[INITIAL_COUNT]
    config = replace(launch_config, seeds=[ONE_ROUND], compute_multiplier_k=TOTAL_UNITS)
    directory = tmp_path / config.dataset / config.model / config.experiment_directory / problem.problem_id / "seed_1"
    state = RefinementState(problem.statement, config.budget_tokens, None)
    manager = SessionManager(directory, RunLock)
    manager.open(state)
    state.status = RunStatus.FINISHED if solution is not None else RunStatus.EXHAUSTED
    state.solution = solution
    state.solution_checkpoints = {key: solution for key in range(ONE_ROUND, TOTAL_UNITS + ONE_ROUND)}
    with patch.object(manager, "_write_solutions", side_effect=OSError("export interrupted")):
        with pytest.raises(OSError, match="export interrupted"):
            manager.checkpoint()
    manager.close()
    original = (directory / SESSION_FILENAME).read_bytes()
    result = await RunLauncher(config, sandbox_environment, tmp_path).run([problem], config.model)
    assert result.skipped == ONE_ROUND
    sandbox_environment.assert_not_called()
    assert (directory / SESSION_FILENAME).read_bytes() == original
    exports = sorted(directory.glob("solution_*x.md"))
    assert len(exports) == TOTAL_UNITS
    assert all(path.read_text() == (solution or "") for path in exports)
    assert not (directory / RUNTIME_DIRECTORY).exists()


async def test_running_attempt_lock_is_a_skip(
    launch_config: LaunchConfig, problems: list[Problem], tmp_path: Path,
) -> None:
    """An exclusive lock rejection is distinct from an inference or container failure."""
    sandbox = create_autospec(Sandbox, instance=True)
    sandbox.run.side_effect = AttemptAlreadyRunning("attempt already running")
    config = replace(launch_config, seeds=[ONE_ROUND])
    result = await RunLauncher(config, lambda: nullcontext(lambda: sandbox), tmp_path).run(problems[:ONE_ROUND], config.model)
    assert result.skipped == ONE_ROUND and result.failed == INITIAL_COUNT


async def test_cancellation_reaches_active_attempts(
    launch_config: LaunchConfig, problems: list[Problem], tmp_path: Path,
) -> None:
    """Cancellation must not be reported as an ordinary failed seed and continue scheduling."""
    started, stopped = Event(), Event()

    async def run(directory: Path, request: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Expose cleanup of the currently admitted attempt."""
        started.set()
        try:
            await Event().wait()
        finally:
            stopped.set()

    sandbox = create_autospec(Sandbox, instance=True)
    sandbox.run.side_effect = run
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(RunLauncher(launch_config, lambda: nullcontext(lambda: sandbox), tmp_path).run(problems, launch_config.model))
        await started.wait()
        task.cancel()
        with pytest.raises(CancelledError):
            await task
    assert stopped.is_set()
    sandbox.run.assert_awaited_once()


@pytest.mark.parametrize("status", [RunStatus.RUNNING, RunStatus.FINISHED])
@pytest.mark.parametrize(("first_model", "second_model"), [
    ("litellm/team/model", "litellm/team-model"), ("litellm/Team/model", "litellm/team/model"),
])
async def test_distinct_models_never_resume_or_skip_each_others_checkpoints(
    launch_config: LaunchConfig, problems: list[Problem], tmp_path: Path,
    status: RunStatus, first_model: str, second_model: str,
) -> None:
    """Preserve model identity in directory names without introducing checkpoint hashes."""
    first = replace(launch_config, model=first_model, seeds=[ONE_ROUND])
    second = replace(first, model=second_model)
    assert first.model_directory.casefold() != second.model_directory.casefold()
    problem = problems[INITIAL_COUNT]
    directory = tmp_path / first.dataset / first.model_directory / first.experiment_directory / problem.problem_id / "seed_1"
    manager = SessionManager(directory, RunLock)
    state = RefinementState(problem.statement, first.budget_tokens, None)
    state.status, state.solution = status, INITIAL_SOLUTION
    manager.open(state)
    manager.close()
    original = (directory / SESSION_FILENAME).read_bytes()

    async def run(destination: Path, request: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Use real storage to prove the second model starts independently."""
        other = SessionManager(destination, RunLock)
        try:
            restored = other.open(RefinementState(problem.statement, second.budget_tokens, None))
            assert restored.solution is None and restored.output_tokens == INITIAL_COUNT
        finally:
            other.close()

    sandbox = create_autospec(Sandbox, instance=True)
    sandbox.run.side_effect = run
    result = await RunLauncher(second, lambda: nullcontext(lambda: sandbox), tmp_path).run([problem], second_model)
    assert result.completed == ONE_ROUND and not result.skipped and not result.failed
    assert (directory / SESSION_FILENAME).read_bytes() == original
