"""Prefix forks retain matching within-budget conversation, files, and solution."""

from asyncio import timeout
from zipfile import ZipFile

import pytest
from experiments.constants import CONTINUE, CONTINUE_ORACLE
from harness.self_refine.constants import GAPS_FOUND_VERDICT
from harness.self_refine.models import Phase, RefinementState, RunStatus
from harness.session.constants import CHECKPOINT_FILENAME, SESSION_FILENAME
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import INITIAL_COUNT

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    FAILED_PHASE_FILENAME,
    INITIAL_SOLUTION,
    ONE_ROUND,
    PHASE_TOKENS,
    PREFIX_UNITS,
    PROBLEM,
    REVISED_SOLUTION,
    SDK_INITIAL_TOKENS,
    SDK_MESSAGE_ID,
    TOTAL_UNITS,
    TWO_PHASES,
    UNIT_TOKENS,
    WORKSPACE_FILENAME,
)
from tests.support.persistence_harness import PersistenceHarness


@pytest.mark.parametrize("streamed, overshoot", [
    (False, INITIAL_COUNT), (False, PHASE_TOKENS),
    (True, INITIAL_COUNT), (True, UNIT_TOKENS + PHASE_TOKENS),
])
async def test_pause_and_both_forks_keep_only_eligible_phase_files(
    persistence: PersistenceHarness, streamed: bool, overshoot: int,
) -> None:
    """Real persistence rolls back interrupted/over-budget phases but retains an exact-boundary completed revision."""
    boundary = PREFIX_UNITS * UNIT_TOKENS
    state = RefinementState(PROBLEM, TOTAL_UNITS * UNIT_TOKENS, boundary)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = persistence.start(persistence.manager, state)
        await persistence.wait_for_prompt(ONE_ROUND)
        (persistence.workspace() / WORKSPACE_FILENAME).write_text(INITIAL_SOLUTION)
        persistence.send_result(persistence.sdk.transports[-ONE_ROUND], INITIAL_SOLUTION)
        await persistence.wait_for_prompt(TWO_PHASES)
        persistence.send_result(persistence.sdk.transports[-ONE_ROUND], GAPS_FOUND_VERDICT)
        await persistence.wait_for_prompt(PREFIX_UNITS)
        with ZipFile(persistence.directory / SESSION_FILENAME) as archive:
            before = {name: archive.read(name) for name in archive.namelist() if name != CHECKPOINT_FILENAME}
        (persistence.workspace() / WORKSPACE_FILENAME).write_text(REVISED_SOLUTION)
        (persistence.workspace() / FAILED_PHASE_FILENAME).write_text(REVISED_SOLUTION)
        transport = persistence.sdk.transports[-ONE_ROUND]
        tokens = boundary - TWO_PHASES * PHASE_TOKENS + overshoot
        transport.assistant(SDK_MESSAGE_ID, REVISED_SOLUTION, tokens if streamed else SDK_INITIAL_TOKENS)
        transport.result(REVISED_SOLUTION, tokens, ONE_ROUND, False)
        result = await task
    rolled_back = streamed or overshoot > INITIAL_COUNT
    assert result.status == RunStatus.PAUSED
    assert result.output_tokens == (TWO_PHASES * PHASE_TOKENS if rolled_back else boundary)
    assert result.solution == (INITIAL_SOLUTION if rolled_back else REVISED_SOLUTION)
    assert result.phase == (Phase.REVISE if rolled_back else Phase.CRITIQUE)
    assert result.solution_checkpoints == {
        **{key: INITIAL_SOLUTION for key in range(ONE_ROUND, PREFIX_UNITS)}, PREFIX_UNITS: result.solution,
    }
    with ZipFile(persistence.directory / SESSION_FILENAME) as archive:
        retained = {name: archive.read(name) for name in archive.namelist() if name != CHECKPOINT_FILENAME}
    assert (retained == before) == rolled_back
    _check_both_forks(persistence, result, retained)


def _check_both_forks(persistence: PersistenceHarness, prefix: RefinementState, retained_files: dict[str, bytes]) -> None:
    """Compare both archived native forks to the exact retained prefix and verify fresh one-block allowances."""
    source = persistence.new_manager()
    source.open(prefix)
    original = (persistence.directory / SESSION_FILENAME).read_bytes()
    for experiment in (CONTINUE, CONTINUE_ORACLE):
        destination = persistence.directory.parent / experiment
        target = SessionManager(destination, RunLock)
        source.fork(target, prefix.output_tokens + UNIT_TOKENS)
        forked = target.read_checkpoint()
        assert forked.refinement.budget_tokens == prefix.output_tokens + UNIT_TOKENS
        assert forked.refinement.output_tokens == prefix.output_tokens
        assert forked.refinement.solution_checkpoints == prefix.solution_checkpoints
        assert forked.refinement.phase == Phase.CONTINUE
        assert forked.fork_session == (prefix.output_tokens > INITIAL_COUNT)
        assert forked.refinement.checkpoint_count == TOTAL_UNITS
        with ZipFile(destination / SESSION_FILENAME) as archive:
            assert {name: archive.read(name) for name in archive.namelist() if name != CHECKPOINT_FILENAME} == retained_files
    assert (persistence.directory / SESSION_FILENAME).read_bytes() == original


async def test_first_solve_overrun_forks_an_empty_retained_session(persistence: PersistenceHarness) -> None:
    """Discard all first-phase output and files without inventing a completed proof or inherited conversation."""
    boundary = PREFIX_UNITS * UNIT_TOKENS
    state = RefinementState(PROBLEM, TOTAL_UNITS * UNIT_TOKENS, boundary)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = persistence.start(persistence.manager, state)
        await persistence.wait_for_prompt(ONE_ROUND)
        (persistence.workspace() / FAILED_PHASE_FILENAME).write_text(REVISED_SOLUTION)
        transport = persistence.sdk.transports[-ONE_ROUND]
        transport.assistant(SDK_MESSAGE_ID, REVISED_SOLUTION, boundary + PHASE_TOKENS)
        transport.result(REVISED_SOLUTION, boundary + PHASE_TOKENS, ONE_ROUND, False)
        result = await task
    assert result.status == RunStatus.PAUSED and result.output_tokens == INITIAL_COUNT
    assert result.solution is None and persistence.checkpoint().session_id is None
    assert result.solution_checkpoints == {key: None for key in range(ONE_ROUND, PREFIX_UNITS + ONE_ROUND)}
    with ZipFile(persistence.directory / SESSION_FILENAME) as archive:
        retained = {name: archive.read(name) for name in archive.namelist() if name != CHECKPOINT_FILENAME}
    assert all(not content for content in retained.values())
    _check_both_forks(persistence, result, retained)
