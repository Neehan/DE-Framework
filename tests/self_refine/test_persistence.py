"""Phase checkpoint, rollback, locking, and completion tests without provider calls."""

from asyncio import CancelledError, timeout
from pathlib import Path
from unittest.mock import patch

import pytest
from harness.self_refine.constants import NO_GAPS_VERDICT
from harness.self_refine.models import Phase, RefinementState, RunStatus
from harness.session.constants import (
    LOCK_FILENAME,
    RUNTIME_DIRECTORY,
    SESSION_FILENAME,
)
from harness.utils.constants import ZERO_TOKENS

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    BUDGET_TOKENS,
    FAILED_PHASE_FILENAME,
    FIRST_SOLUTION_FILENAME,
    INITIAL_SOLUTION,
    ONE_ROUND,
    PHASE_TOKENS,
    PROBLEM,
    REVISED_SOLUTION,
    SDK_MESSAGE_ID,
    TWO_PHASES,
    UNIT_TOKENS,
)
from tests.support.persistence_harness import PersistenceHarness


async def test_self_refine_saves_before_starting_next_phase(persistence: PersistenceHarness) -> None:
    """The next SDK connection sees committed progress, and completion retains only outputs."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = persistence.start(persistence.manager, persistence.state)
        await persistence.wait_for_prompt(ONE_ROUND)
        assert persistence.sdk.clients[-ONE_ROUND].options.task_budget == {"total": BUDGET_TOKENS}
        persistence.send_result(persistence.sdk.transports[-ONE_ROUND], INITIAL_SOLUTION)
        await persistence.wait_for_prompt(TWO_PHASES)
        assert persistence.sdk.clients[-ONE_ROUND].options.task_budget == {"total": BUDGET_TOKENS - PHASE_TOKENS}
        assert persistence.checkpoint().refinement.solution == INITIAL_SOLUTION
        persistence.send_result(persistence.sdk.transports[-ONE_ROUND], NO_GAPS_VERDICT)
        result = await task
    assert result.status == RunStatus.FINISHED
    assert result.output_tokens == TWO_PHASES * PHASE_TOKENS
    assert persistence.checkpoint().refinement.status == RunStatus.FINISHED
    assert {path.name for path in persistence.directory.iterdir()} == {
        LOCK_FILENAME,
        SESSION_FILENAME,
        FIRST_SOLUTION_FILENAME,
    }


async def test_disconnect_restores_only_completed_phases(persistence: PersistenceHarness) -> None:
    """A disconnected critique is rerun from the saved solve, without retaining its files."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = persistence.start(persistence.manager, persistence.state)
        await persistence.wait_for_prompt(ONE_ROUND)
        persistence.send_result(persistence.sdk.transports[-ONE_ROUND], INITIAL_SOLUTION)
        await persistence.wait_for_prompt(TWO_PHASES)
        (persistence.workspace() / FAILED_PHASE_FILENAME).write_text(REVISED_SOLUTION)
        persistence.sdk.transports[-ONE_ROUND].incoming.put_nowait(None)
        with pytest.raises(ConnectionError):
            await task
    restored = persistence.new_manager().open(RefinementState(PROBLEM, BUDGET_TOKENS, None))
    assert restored.phase == Phase.CRITIQUE
    assert restored.output_tokens == PHASE_TOKENS
    assert restored.solution == INITIAL_SOLUTION
    assert not (persistence.workspace() / FAILED_PHASE_FILENAME).exists()


async def test_cancellation_keeps_checkpoint_and_releases_lock(persistence: PersistenceHarness) -> None:
    """Cancelling a pending phase never commits it or prevents a later launch."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = persistence.start(persistence.manager, persistence.state)
        await persistence.wait_for_prompt(ONE_ROUND)
        persistence.send_result(persistence.sdk.transports[-ONE_ROUND], INITIAL_SOLUTION)
        await persistence.wait_for_prompt(TWO_PHASES)
        original = (persistence.directory / SESSION_FILENAME).read_bytes()
        task.cancel()
        with pytest.raises(CancelledError):
            await task
    assert (persistence.directory / SESSION_FILENAME).read_bytes() == original
    restored = persistence.new_manager().open(RefinementState(PROBLEM, BUDGET_TOKENS, None))
    assert restored.phase == Phase.CRITIQUE
    assert restored.output_tokens == PHASE_TOKENS


async def test_budget_stop_is_saved_without_opening_another_connection(
    persistence: PersistenceHarness,
) -> None:
    """An exact-budget completed solution is committed as exhausted and remains gradeable."""
    state = RefinementState(PROBLEM, PHASE_TOKENS, None)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = persistence.start(persistence.manager, state)
        await persistence.wait_for_prompt(ONE_ROUND)
        persistence.send_result(persistence.sdk.transports[-ONE_ROUND], INITIAL_SOLUTION)
        result = await task
    assert result.status == RunStatus.EXHAUSTED
    assert persistence.checkpoint().refinement.solution == INITIAL_SOLUTION
    assert len(persistence.sdk.transports) == ONE_ROUND


async def test_live_recovery_keeps_lock_and_rolls_back_failed_phase(persistence: PersistenceHarness, critique_prompt: str) -> None:
    """A disconnect resumes automatically from the saved solve with exclusive ownership."""

    async def wait(delay: float) -> None:
        """Prove the seed is still exclusively locked during backoff."""
        with pytest.raises(RuntimeError, match="already running"):
            persistence.new_manager().open(persistence.state)

    state = RefinementState(PROBLEM, TWO_PHASES * UNIT_TOKENS, None)
    with (
        patch("harness.self_refine.recovery.RECOVERY_MAX_RETRIES", ONE_ROUND),
        patch.object(persistence.wait, "side_effect", wait),
    ):
        async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
            task = persistence.start(persistence.manager, state)
            await persistence.wait_for_prompt(ONE_ROUND)
            persistence.send_result(persistence.sdk.transports[-ONE_ROUND], INITIAL_SOLUTION)
            await persistence.wait_for_prompt(TWO_PHASES)
            source_id = persistence.checkpoint().session_id
            (persistence.workspace() / FAILED_PHASE_FILENAME).write_text(REVISED_SOLUTION)
            failed = persistence.sdk.transports[-ONE_ROUND]
            failed.start(SDK_MESSAGE_ID)
            failed.delta(UNIT_TOKENS)
            failed.incoming.put_nowait(None)
            prompt = await persistence.wait_for_prompt(TWO_PHASES + ONE_ROUND)
            recovered = persistence.sdk.transports[-ONE_ROUND]
            assert state.solution_checkpoints == {ONE_ROUND: INITIAL_SOLUTION}
            assert persistence.checkpoint().refinement.solution_checkpoints == {}
            assert prompt["message"]["content"] == critique_prompt
            assert recovered.options.resume is not None and Path(recovered.options.resume).stem == source_id
            assert not (persistence.workspace() / FAILED_PHASE_FILENAME).exists()
            persistence.send_result(recovered, NO_GAPS_VERDICT)
            result = await task
    assert result.status == RunStatus.FINISHED
    assert result.output_tokens == TWO_PHASES * PHASE_TOKENS
    assert result.solution == INITIAL_SOLUTION
    assert result.solution_checkpoints == {ONE_ROUND: INITIAL_SOLUTION, TWO_PHASES: INITIAL_SOLUTION}


@pytest.mark.parametrize("status", (RunStatus.PAUSED, RunStatus.FINISHED, RunStatus.EXHAUSTED))
async def test_stopped_checkpoints_skip_provider(persistence: PersistenceHarness, status: RunStatus) -> None:
    """An ordinary rerun returns the saved outcome without starting another connection."""
    state = persistence.state
    state.status = status
    state.solution = INITIAL_SOLUTION
    persistence.manager.open(state)
    persistence.manager.close()
    result = await persistence.start(persistence.new_manager(), RefinementState(PROBLEM, BUDGET_TOKENS, None))
    assert result.status == status
    assert result.solution == INITIAL_SOLUTION
    assert persistence.sdk.transports == []
    assert not (persistence.directory / RUNTIME_DIRECTORY).exists()


async def test_first_solve_failure_restores_initial_checkpoint(persistence: PersistenceHarness) -> None:
    """A failed first solve restores an empty conversation, workspace, and token count."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = persistence.start(persistence.manager, persistence.state)
        await persistence.wait_for_prompt(ONE_ROUND)
        (persistence.workspace() / FAILED_PHASE_FILENAME).write_text(REVISED_SOLUTION)
        persistence.sdk.transports[-ONE_ROUND].incoming.put_nowait(None)
        with pytest.raises(ConnectionError):
            await task
    restored = persistence.new_manager().open(RefinementState(PROBLEM, BUDGET_TOKENS, None))
    assert restored.phase == Phase.SOLVE
    assert restored.output_tokens == ZERO_TOKENS
    assert restored.solution is None
    assert persistence.checkpoint().session_id is None
    assert not (persistence.workspace() / FAILED_PHASE_FILENAME).exists()


async def test_duplicate_runner_preserves_active_connection(persistence: PersistenceHarness) -> None:
    """Rejecting a second runner leaves the first owner's SDK and process lock active."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        owner = persistence.start(persistence.manager, persistence.state)
        await persistence.wait_for_prompt(ONE_ROUND)
        with pytest.raises(RuntimeError, match="already open"):
            await persistence.start(persistence.manager, persistence.state)
        with pytest.raises(RuntimeError, match="already running"):
            persistence.new_manager().open(persistence.state)
        assert persistence.sdk.transports[-ONE_ROUND].ready
        owner.cancel()
        with pytest.raises(CancelledError):
            await owner


async def test_replay_ids_survive_archive_and_recovery(persistence: PersistenceHarness) -> None:
    """Restoring the saved solve does not charge replayed output against the next critique."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        first = persistence.start(persistence.manager, persistence.state)
        await persistence.wait_for_prompt(ONE_ROUND)
        persistence.send_result(persistence.sdk.transports[-ONE_ROUND], INITIAL_SOLUTION)
        await persistence.wait_for_prompt(TWO_PHASES)
        old_id = next(iter(persistence.checkpoint().completed_message_ids))
        first.cancel()
        with pytest.raises(CancelledError):
            await first
        resumed = persistence.start(persistence.new_manager(), RefinementState(PROBLEM, BUDGET_TOKENS, None))
        await persistence.wait_for_prompt(TWO_PHASES + ONE_ROUND)
        transport = persistence.sdk.transports[-ONE_ROUND]
        transport.assistant(old_id, INITIAL_SOLUTION, PHASE_TOKENS)
        persistence.send_result(transport, NO_GAPS_VERDICT)
        result = await resumed
    assert result.status == RunStatus.FINISHED
    assert result.output_tokens == TWO_PHASES * PHASE_TOKENS
