"""Shared-prefix ownership, durable branch identity, and continuation scheduling."""

import json
from asyncio import CancelledError, Event, create_task, timeout
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch
from zipfile import ZipFile

import pytest
from experiments.constants import (
    CONTINUATION_TOTAL_MULTIPLIER,
    CONTINUE,
    CONTINUE_ORACLE,
    PREFIX_DIRECTORY,
    PREFIX_TOKENS,
    UNAIDED,
)
from harness.sandbox.sandbox import Sandbox
from harness.self_refine.models import Phase, RunStatus
from harness.session.constants import (
    CHECKPOINT_FILENAME,
    SDK_DIRECTORY,
    SESSION_FILENAME,
    TRANSCRIPT_EXTENSION,
    WORKSPACE_DIRECTORY,
)
from harness.session.errors import AttemptAlreadyRunning
from harness.session.models import SessionState
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import INITIAL_COUNT, OUTPUT_TOKENS_PER_BLOCK
from launcher.launcher.continuation_launcher import ContinuationLauncher
from launcher.models import LaunchConfig, Problem, RunRequest

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    INITIAL_SOLUTION,
    NATIVE_SESSION_ID,
    ONE_ROUND,
    PHASE_TOKENS,
    PREFIX_UNITS,
    REVISED_SOLUTION,
    SDK_MESSAGE_ID,
    SDK_PROJECT_DIRECTORY,
    TOTAL_UNITS,
    WORKSPACE_FILENAME,
)
from tests.launcher.constants import REFERENCE_SKETCH, TEST_SEEDS, TWO_WORKERS


@pytest.fixture
def continuation_config(launch_config: LaunchConfig) -> LaunchConfig:
    """Select one branch and one seed using the existing launcher configuration."""
    return replace(launch_config, experiment=CONTINUE, compute_multiplier_k=CONTINUATION_TOTAL_MULTIPLIER, seeds=[ONE_ROUND])


def _directory(root: Path, config: LaunchConfig, problem: Problem) -> Path:
    """Locate the public results directory without depending on private launcher helpers."""
    return root / config.dataset / config.model_directory / config.experiment_directory / problem.problem_id / "seed_1"


def _prefix_directory(root: Path, config: LaunchConfig, problem: Problem) -> Path:
    """Locate the shared source independently of either continuation variant."""
    return root / config.dataset / config.model_directory / PREFIX_DIRECTORY / problem.problem_id / "seed_1"


def _read_saved(directory: Path) -> SessionState:
    """Inspect durable state after the worker has released its runtime."""
    with ZipFile(directory / SESSION_FILENAME) as archive:
        return SessionState.from_json(archive.read(CHECKPOINT_FILENAME).decode())


def _save_prefix(manager: SessionManager, request: RunRequest, status: RunStatus) -> None:
    """Persist a stopped prefix with native history and workspace content for genuine fork restoration."""
    assert request.experiment == UNAIDED and request.sketch is None
    assert request.budget_tokens == CONTINUATION_TOTAL_MULTIPLIER * OUTPUT_TOKENS_PER_BLOCK
    assert request.pause_at_tokens == PREFIX_TOKENS
    state = manager.session_state.refinement
    state.status, state.phase = status, Phase.CRITIQUE
    state.output_tokens = PREFIX_TOKENS if status == RunStatus.PAUSED else PHASE_TOKENS
    state.rounds, state.solution = ONE_ROUND, INITIAL_SOLUTION
    state.solution_checkpoints = {key: INITIAL_SOLUTION for key in range(ONE_ROUND, PREFIX_UNITS + ONE_ROUND)}
    manager.session_state.session_id = NATIVE_SESSION_ID
    manager.session_state.completed_message_ids.add(SDK_MESSAGE_ID)
    transcript = manager.runtime_directory / SDK_DIRECTORY / SDK_PROJECT_DIRECTORY / f"{NATIVE_SESSION_ID}{TRANSCRIPT_EXTENSION}"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(INITIAL_SOLUTION)
    (manager.runtime_directory / WORKSPACE_DIRECTORY / WORKSPACE_FILENAME).write_text(INITIAL_SOLUTION)


def _make_sandbox(prefix_status: RunStatus) -> MagicMock:
    """Inject deterministic execution while exercising real ownership, archives, and native-file restoration."""
    async def run(directory: Path, encoded: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Write a stopped prefix or advance an independently restored branch."""
        request = RunRequest(**json.loads(encoded))
        with ownership(directory):
            manager = SessionManager(directory, nullcontext)
            try:
                state = manager.open(request.initial_state)
                if request.experiment == UNAIDED:
                    _save_prefix(manager, request, prefix_status)
                else:
                    assert state.phase in (Phase.CONTINUE, Phase.REVISE)
                    assert state.rounds == ONE_ROUND
                    assert manager.session_state.fork_session
                    assert manager.session_state.session_id == NATIVE_SESSION_ID
                    assert manager.session_state.completed_message_ids == {SDK_MESSAGE_ID}
                    assert manager.resume_target is not None
                    assert Path(manager.resume_target).read_text() == INITIAL_SOLUTION
                    assert (manager.runtime_directory / WORKSPACE_DIRECTORY / WORKSPACE_FILENAME).read_text() == INITIAL_SOLUTION
                    assert state.checkpoint_count == TOTAL_UNITS
                    state.solution_checkpoints[TOTAL_UNITS] = REVISED_SOLUTION
                    state.output_tokens += PHASE_TOKENS
                    state.status, state.phase, state.solution = RunStatus.FINISHED, Phase.CRITIQUE, REVISED_SOLUTION
                manager.checkpoint()
            finally:
                manager.close()

    sandbox = create_autospec(Sandbox, instance=True)
    sandbox.run.side_effect = run
    return sandbox


@pytest.mark.parametrize("status", (RunStatus.PAUSED, RunStatus.FINISHED))
async def test_both_branches_share_prefix_and_keep_independent_archives(
    continuation_config: LaunchConfig, problems: list[Problem], tmp_path: Path, status: RunStatus,
) -> None:
    """Early convergence and a full prefix each grant exactly one additional block to both variants."""
    problem = problems[INITIAL_COUNT]
    sandbox = _make_sandbox(status)
    environment = lambda: nullcontext(lambda: sandbox)
    ordinary = ContinuationLauncher(continuation_config, environment, tmp_path)
    first = await ordinary.run([problem], continuation_config.model)
    prefix = _prefix_directory(tmp_path, continuation_config, problem)
    original = (prefix / SESSION_FILENAME).read_bytes()
    oracle_config = replace(continuation_config, experiment=CONTINUE_ORACLE)
    oracle = ContinuationLauncher(oracle_config, environment, tmp_path)
    second = await oracle.run([replace(problem, sketch=REFERENCE_SKETCH)], oracle_config.model)
    assert first.completed == second.completed == ONE_ROUND
    assert first.failed == second.failed == INITIAL_COUNT
    requests = [RunRequest(**json.loads(call.args[ONE_ROUND])) for call in sandbox.run.await_args_list]
    assert [request.experiment for request in requests] == [UNAIDED, CONTINUE, CONTINUE_ORACLE]
    spent = PREFIX_TOKENS if status == RunStatus.PAUSED else PHASE_TOKENS
    assert all(request.budget_tokens == spent + OUTPUT_TOKENS_PER_BLOCK for request in requests[ONE_ROUND:])
    assert requests[-ONE_ROUND].sketch == REFERENCE_SKETCH
    assert (prefix / SESSION_FILENAME).read_bytes() == original
    for config in (continuation_config, oracle_config):
        saved = _read_saved(_directory(tmp_path, config, problem))
        assert saved.refinement.budget_tokens == spent + OUTPUT_TOKENS_PER_BLOCK
        assert saved.refinement.solution == REVISED_SOLUTION
        assert saved.refinement.solution_checkpoints == {
            **{key: INITIAL_SOLUTION for key in range(ONE_ROUND, PREFIX_UNITS + ONE_ROUND)}, TOTAL_UNITS: REVISED_SOLUTION,
        }
        assert saved.refinement.rounds == ONE_ROUND
        assert saved.fork_session and saved.session_id == NATIVE_SESSION_ID


async def test_concurrent_variants_wait_but_duplicate_branch_skips(
    continuation_config: LaunchConfig, problems: list[Problem], tmp_path: Path,
) -> None:
    """The sibling variant waits on its dependency while a duplicate invocation cannot own the first branch."""
    problem = problems[INITIAL_COUNT]
    sandbox = _make_sandbox(RunStatus.PAUSED)
    execute = sandbox.run.side_effect
    started, release, waiting = Event(), Event(), Event()

    async def run(directory: Path, request: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Hold the first prefix worker while competing launchers try to acquire their locks."""
        if json.loads(request)["experiment"] == UNAIDED:
            started.set()
            await release.wait()
        await execute(directory, request, ownership)

    async def wait(delay: float) -> None:
        """Confirm actual prefix-lock contention before allowing its owner to finish."""
        waiting.set()
        await release.wait()

    sandbox.run.side_effect = run
    environment = lambda: nullcontext(lambda: sandbox)
    ordinary = ContinuationLauncher(continuation_config, environment, tmp_path)
    oracle_config = replace(continuation_config, experiment=CONTINUE_ORACLE)
    oracle = ContinuationLauncher(oracle_config, environment, tmp_path)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        first = create_task(ordinary.run([problem], continuation_config.model))
        await started.wait()
        duplicate = await ordinary.run([problem], continuation_config.model)
        assert duplicate.skipped == ONE_ROUND and duplicate.failed == INITIAL_COUNT
        with patch("launcher.launcher.continuation_launcher.sleep", side_effect=wait):
            second = create_task(oracle.run([replace(problem, sketch=REFERENCE_SKETCH)], oracle_config.model))
            await waiting.wait()
            assert not second.done()
            release.set()
            first_result, second_result = await first, await second
    assert first_result.completed == second_result.completed == ONE_ROUND
    assert first_result.failed == second_result.failed == INITIAL_COUNT
    experiments = [json.loads(call.args[ONE_ROUND])["experiment"] for call in sandbox.run.await_args_list]
    assert experiments.count(UNAIDED) == ONE_ROUND


async def test_cancelling_prefix_wait_releases_branch_ownership(
    continuation_config: LaunchConfig, problems: list[Problem], tmp_path: Path,
) -> None:
    """Cancelling a dependency wait releases the waiting branch and does not release someone else's prefix."""
    problem = problems[INITIAL_COUNT]
    waiting = Event()
    sandbox = _make_sandbox(RunStatus.PAUSED)
    launcher = ContinuationLauncher(continuation_config, lambda: nullcontext(lambda: sandbox), tmp_path)
    branch = _directory(tmp_path, continuation_config, problem)
    prefix = _prefix_directory(tmp_path, continuation_config, problem)

    async def wait(delay: float) -> None:
        """Hold an actual lock retry until task cancellation."""
        waiting.set()
        await Event().wait()

    with RunLock(prefix), patch("launcher.launcher.continuation_launcher.sleep", side_effect=wait):
        async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
            task = create_task(launcher.run([problem], continuation_config.model))
            await waiting.wait()
            with pytest.raises(AttemptAlreadyRunning), RunLock(branch):
                pass
            task.cancel()
            with pytest.raises(CancelledError):
                await task
        with RunLock(branch):
            pass
        with pytest.raises(AttemptAlreadyRunning), RunLock(prefix):
            pass
    sandbox.run.assert_not_awaited()


async def test_busy_prefix_does_not_block_an_independent_seed(
    continuation_config: LaunchConfig, problems: list[Problem], tmp_path: Path,
) -> None:
    """A shared-prefix wait occupies one worker while another seed can finish its own branch."""
    problem = problems[INITIAL_COUNT]
    config = replace(continuation_config, seeds=TEST_SEEDS[:TWO_WORKERS], max_concurrency=TWO_WORKERS)
    sandbox = _make_sandbox(RunStatus.PAUSED)
    execute = sandbox.run.side_effect
    waiting, release, independent_finished = Event(), Event(), Event()

    async def run(directory: Path, request: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Expose completion of the unrelated seed through the normal injected execution path."""
        await execute(directory, request, ownership)
        if directory.name == "seed_2" and json.loads(request)["experiment"] == CONTINUE:
            independent_finished.set()

    async def wait(delay: float) -> None:
        """Keep the first worker's lock retry asleep until the test releases its prefix."""
        waiting.set()
        await release.wait()

    sandbox.run.side_effect = run
    launcher = ContinuationLauncher(config, lambda: nullcontext(lambda: sandbox), tmp_path)
    with patch("launcher.launcher.continuation_launcher.sleep", side_effect=wait):
        async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
            with RunLock(_prefix_directory(tmp_path, config, problem)):
                task = create_task(launcher.run([problem], config.model))
                await waiting.wait()
                await independent_finished.wait()
                assert not task.done()
            release.set()
            result = await task
    assert result.completed == TWO_WORKERS
    assert result.failed == result.skipped == INITIAL_COUNT


@pytest.mark.parametrize("prefix_present", (True, False))
async def test_existing_branch_resumes_without_refork_or_budget_extension(
    continuation_config: LaunchConfig, problems: list[Problem], tmp_path: Path, prefix_present: bool,
) -> None:
    """Saved branches resume and skip independently of a busy or unavailable prefix, retaining their allowance."""
    problem = problems[INITIAL_COUNT]
    sandbox = _make_sandbox(RunStatus.FINISHED)
    launcher = ContinuationLauncher(continuation_config, lambda: nullcontext(lambda: sandbox), tmp_path)
    assert (await launcher.run([problem], continuation_config.model)).completed == ONE_ROUND
    prefix = _prefix_directory(tmp_path, continuation_config, problem)
    if not prefix_present:
        prefix.rename(prefix.with_suffix(".archived"))
    directory = _directory(tmp_path, continuation_config, problem)
    saved = _read_saved(directory)
    manager = SessionManager(directory, RunLock)
    try:
        manager.open(saved.refinement)
        state = manager.continue_run(saved.refinement.budget_tokens)
        state.checkpoint_count = TOTAL_UNITS
        del state.solution_checkpoints[TOTAL_UNITS]
        state.phase = Phase.REVISE
        manager.checkpoint()
    finally:
        manager.close()
    before = _read_saved(directory).refinement
    sandbox.run.reset_mock()
    with RunLock(prefix), patch.object(SessionManager, "fork", side_effect=AssertionError("existing branch must not be forked again")):
        async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
            result = await launcher.run([problem], continuation_config.model)
    assert result.completed == ONE_ROUND and result.failed == INITIAL_COUNT
    sandbox.run.assert_awaited_once()
    after = _read_saved(directory).refinement
    assert after.budget_tokens == before.budget_tokens
    assert after.output_tokens == before.output_tokens + PHASE_TOKENS
    sandbox.run.reset_mock()
    with RunLock(prefix):
        async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
            stopped = await launcher.run([problem], continuation_config.model)
    assert stopped.skipped == ONE_ROUND
    sandbox.run.assert_not_awaited()
