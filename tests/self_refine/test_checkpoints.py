"""Budget snapshots preserve eligible solutions and allocated continuation labels."""

from pathlib import Path

import pytest
from harness.self_refine.constants import GAPS_FOUND_VERDICT, NO_GAPS_VERDICT
from harness.self_refine.models import (
    Phase,
    RefinementConfig,
    RefinementState,
    RunStatus,
)
from harness.self_refine.self_refine import SelfRefine
from harness.session.constants import SESSION_FILENAME, SOLUTION_FILENAME_TEMPLATE
from harness.session.models import EventKind, SessionEvent
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import INITIAL_COUNT

from tests.constants import (
    INITIAL_SOLUTION,
    ONE_ROUND,
    PHASE_TOKENS,
    PREFIX_UNITS,
    PROBLEM,
    REVISED_SOLUTION,
    TOTAL_UNITS,
    TWO_PHASES,
    UNIT_TOKENS,
)
from tests.support.fake_agent_session import FakeAgentSession
from tests.support.fake_recovery import FakeRecovery


@pytest.mark.parametrize("overshoot", (INITIAL_COUNT, PHASE_TOKENS))
async def test_revision_at_checkpoint_boundary_and_early_finish(overshoot: int) -> None:
    """A within-boundary revision qualifies; a later revision only reaches subsequent carried checkpoints."""
    revision_tokens = UNIT_TOKENS - TWO_PHASES * PHASE_TOKENS
    session = FakeAgentSession([
        [SessionEvent(EventKind.COMPLETED, PHASE_TOKENS, INITIAL_SOLUTION)],
        [SessionEvent(EventKind.COMPLETED, PHASE_TOKENS, GAPS_FOUND_VERDICT)],
        [SessionEvent(EventKind.USAGE, revision_tokens, ""),
         SessionEvent(EventKind.COMPLETED, revision_tokens + overshoot, REVISED_SOLUTION)],
        [SessionEvent(EventKind.COMPLETED, PHASE_TOKENS, NO_GAPS_VERDICT)],
    ])
    state = RefinementState(PROBLEM, TOTAL_UNITS * UNIT_TOKENS, None)
    await SelfRefine(FakeRecovery(session), RefinementConfig(ONE_ROUND, ONE_ROUND)).run(state)
    assert state.solution_checkpoints == {
        ONE_ROUND: INITIAL_SOLUTION if overshoot else REVISED_SOLUTION,
        **{key: REVISED_SOLUTION for key in range(TWO_PHASES, TOTAL_UNITS + ONE_ROUND)},
    }
    assert state.status == RunStatus.FINISHED
    assert state.output_tokens == UNIT_TOKENS + overshoot + PHASE_TOKENS
    assert session.interruptions == INITIAL_COUNT and not session.queued_messages


async def test_one_usage_jump_can_cross_multiple_empty_checkpoints() -> None:
    """A late first solution cannot be assigned retroactively to any of the earlier budgets."""
    session = FakeAgentSession([
        [SessionEvent(EventKind.COMPLETED, PREFIX_UNITS * UNIT_TOKENS + PHASE_TOKENS, INITIAL_SOLUTION)],
        [SessionEvent(EventKind.COMPLETED, PHASE_TOKENS, NO_GAPS_VERDICT)],
    ])
    state = RefinementState(PROBLEM, TOTAL_UNITS * UNIT_TOKENS, None)
    await SelfRefine(FakeRecovery(session), RefinementConfig(ONE_ROUND, ONE_ROUND)).run(state)
    assert state.solution_checkpoints == {
        **{key: None for key in range(ONE_ROUND, PREFIX_UNITS + ONE_ROUND)}, TOTAL_UNITS: INITIAL_SOLUTION,
    }


@pytest.mark.parametrize("prefix_tokens", (TWO_PHASES * PHASE_TOKENS, PREFIX_UNITS * UNIT_TOKENS, PREFIX_UNITS * UNIT_TOKENS + PHASE_TOKENS))
async def test_fork_keeps_prefix_snapshots_and_labels_full_extra_block_as_four(tmp_path: Path, prefix_tokens: int) -> None:
    """Early finish and a completed 3x phase both grant one additional block with allocated 3x/4x labels."""
    prefix = RefinementState(PROBLEM, TOTAL_UNITS * UNIT_TOKENS, PREFIX_UNITS * UNIT_TOKENS)
    prefix.phase, prefix.solution = Phase.REVISE, INITIAL_SOLUTION
    session = FakeAgentSession([
        [SessionEvent(EventKind.COMPLETED, prefix_tokens - PHASE_TOKENS, INITIAL_SOLUTION)],
        [SessionEvent(EventKind.COMPLETED, PHASE_TOKENS, NO_GAPS_VERDICT)],
    ])
    config = RefinementConfig(ONE_ROUND, ONE_ROUND)
    prefix = await SelfRefine(FakeRecovery(session), config).run(prefix)
    source = SessionManager(tmp_path / "prefix", RunLock)
    source.open(prefix)
    original = (tmp_path / "prefix" / SESSION_FILENAME).read_bytes()
    target = SessionManager(tmp_path / "branch", RunLock)
    try:
        forked = source.fork(target, prefix.output_tokens + UNIT_TOKENS)
        state = target.open(forked)
        session = FakeAgentSession([[SessionEvent(EventKind.COMPLETED, UNIT_TOKENS, REVISED_SOLUTION)]])
        await SelfRefine(FakeRecovery(session), config).run(state)
        target.checkpoint()
        assert state.status == RunStatus.EXHAUSTED
        assert state.output_tokens == prefix.output_tokens + UNIT_TOKENS
        assert state.solution_checkpoints == {**prefix.solution_checkpoints, TOTAL_UNITS: REVISED_SOLUTION}
        assert len(state.solution_checkpoints) == TOTAL_UNITS
        for multiplier, solution in state.solution_checkpoints.items():
            assert (tmp_path / "branch" / SOLUTION_FILENAME_TEMPLATE.format(multiplier=multiplier)).read_text() == solution
        assert (tmp_path / "prefix" / SESSION_FILENAME).read_bytes() == original
    finally:
        target.close()
        source.close()
