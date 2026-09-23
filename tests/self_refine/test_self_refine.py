"""Behavioral checks for refinement, budget warnings, and resumable pauses."""

from copy import deepcopy

import pytest
from harness.self_refine.constants import (
    GAPS_FOUND_VERDICT,
    NO_GAPS_VERDICT,
)
from harness.self_refine.models import (
    Phase,
    RefinementConfig,
    RefinementState,
    RunStatus,
)
from harness.self_refine.self_refine import SelfRefine
from harness.session.models import EventKind, SessionEvent
from harness.utils.constants import DEFAULT_MIN_ROUNDS, INITIAL_COUNT, WARNING_TOKENS

from tests.constants import (
    BUDGET_TOKENS,
    EXPECTED_WARNING_REMAINDER,
    FIVE_ROUNDS,
    INITIAL_SOLUTION,
    LATER_USAGE_TOKENS,
    ONE_ROUND,
    PARTIAL_SOLUTION,
    PHASE_TOKENS,
    PREFIX_UNITS,
    PROBLEM,
    REVISED_SOLUTION,
    THREE_CLEAN_CRITIQUES,
    TOTAL_UNITS,
    TWO_PHASES,
    TWO_ROUNDS,
    UNIT_TOKENS,
    WARNING_CROSSING_TOKENS,
)
from tests.support.fake_agent_session import FakeAgentSession
from tests.support.fake_recovery import FakeRecovery


def _reply(text: str) -> list[SessionEvent]:
    """Build a completed response with known usage."""
    return [SessionEvent(EventKind.COMPLETED, PHASE_TOKENS, text)]


def _conversation(verdicts: list[str]) -> FakeAgentSession:
    """Script a solve followed by critiques and standalone revisions."""
    replies = [_reply(INITIAL_SOLUTION)]
    for verdict in verdicts:
        replies.append(_reply(verdict))
        replies.append(_reply(REVISED_SOLUTION))
    return FakeAgentSession(replies)


async def test_clean_critiques_do_not_bypass_minimum_rounds(
    refinement_config: RefinementConfig, state: RefinementState, critique_prompt: str
) -> None:
    """The default ten-round floor holds even when every critique is clean."""
    session = _conversation([NO_GAPS_VERDICT] * DEFAULT_MIN_ROUNDS)
    result = await SelfRefine(FakeRecovery(session), refinement_config).run(state)
    assert result.status == RunStatus.FINISHED
    assert result.rounds == DEFAULT_MIN_ROUNDS
    assert result.solution == REVISED_SOLUTION
    assert session.prompts.count(critique_prompt) == DEFAULT_MIN_ROUNDS


async def test_gap_resets_consecutive_clean_critiques(state: RefinementState) -> None:
    """Nonconsecutive clean critiques cannot satisfy the stopping rule."""
    verdicts = [NO_GAPS_VERDICT, GAPS_FOUND_VERDICT]
    verdicts.extend([NO_GAPS_VERDICT] * THREE_CLEAN_CRITIQUES)
    session = _conversation(verdicts)
    config = RefinementConfig(TWO_ROUNDS, THREE_CLEAN_CRITIQUES)
    result = await SelfRefine(FakeRecovery(session), config).run(state)
    assert result.rounds == FIVE_ROUNDS
    assert result.no_gap_critiques == THREE_CLEAN_CRITIQUES


async def test_phase_order_and_retained_solution(
    state: RefinementState, critique_prompt: str, revise_prompt: str,
) -> None:
    """A final critique never replaces the preceding standalone revision."""
    session = _conversation([GAPS_FOUND_VERDICT, NO_GAPS_VERDICT])
    config = RefinementConfig(TWO_ROUNDS, ONE_ROUND)
    result = await SelfRefine(FakeRecovery(session), config).run(state)
    assert session.prompts[INITIAL_COUNT].startswith("Solve the following olympiad problem.")
    assert session.prompts[INITIAL_COUNT].endswith(PROBLEM)
    assert str(BUDGET_TOKENS) in session.prompts[INITIAL_COUNT]
    assert session.prompts[ONE_ROUND:] == [critique_prompt, revise_prompt, critique_prompt]
    assert result.solution == REVISED_SOLUTION


@pytest.mark.parametrize(
    "crossing, remaining",
    [(BUDGET_TOKENS - WARNING_TOKENS, WARNING_TOKENS), (WARNING_CROSSING_TOKENS, EXPECTED_WARNING_REMAINDER)],
)
async def test_warning_reports_first_observed_remaining_tokens_once(
    state: RefinementState, crossing: int, remaining: int
) -> None:
    """Repeated cumulative snapshots neither repeat warnings nor double-charge."""
    session = FakeAgentSession(
        [
            [
                SessionEvent(EventKind.USAGE, crossing, ""),
                SessionEvent(EventKind.USAGE, LATER_USAGE_TOKENS, ""),
                SessionEvent(EventKind.COMPLETED, LATER_USAGE_TOKENS, INITIAL_SOLUTION),
            ],
            _reply(NO_GAPS_VERDICT),
        ]
    )
    config = RefinementConfig(ONE_ROUND, ONE_ROUND)
    result = await SelfRefine(FakeRecovery(session), config).run(state)
    assert len(session.queued_messages) == ONE_ROUND
    assert session.queued_messages[INITIAL_COUNT].startswith(f"You have {remaining} output tokens left. Wrap up")
    assert result.output_tokens == LATER_USAGE_TOKENS + PHASE_TOKENS
    assert result.warning_sent


async def test_terminal_usage_can_trigger_warning(state: RefinementState) -> None:
    """A terminal event supplies usage even when no earlier update arrived."""
    session = FakeAgentSession(
        [
            [SessionEvent(EventKind.COMPLETED, WARNING_CROSSING_TOKENS, INITIAL_SOLUTION)],
            _reply(NO_GAPS_VERDICT),
        ]
    )
    config = RefinementConfig(ONE_ROUND, ONE_ROUND)
    await SelfRefine(FakeRecovery(session), config).run(state)
    assert len(session.queued_messages) == ONE_ROUND
    assert session.queued_messages[INITIAL_COUNT].startswith(f"You have {EXPECTED_WARNING_REMAINDER} output tokens left.")


async def test_resumed_low_budget_warns_before_spending_more_tokens(state: RefinementState) -> None:
    """Existing low-budget progress supplies the first observed remainder."""
    state.output_tokens = WARNING_CROSSING_TOKENS
    state.phase = Phase.CRITIQUE
    state.solution = INITIAL_SOLUTION
    session = FakeAgentSession([_reply(NO_GAPS_VERDICT)])
    config = RefinementConfig(ONE_ROUND, ONE_ROUND)
    await SelfRefine(FakeRecovery(session), config).run(state)
    assert len(session.queued_messages) == ONE_ROUND
    assert session.queued_messages[INITIAL_COUNT].startswith(f"You have {EXPECTED_WARNING_REMAINDER} output tokens left.")


async def test_exhaustion_preserves_solution_before_interrupted_revision(
    refinement_config: RefinementConfig, state: RefinementState
) -> None:
    """An interrupted revision cannot overwrite an earlier complete solution."""
    revision_tokens = BUDGET_TOKENS - TWO_PHASES * PHASE_TOKENS
    session = FakeAgentSession(
        [
            _reply(INITIAL_SOLUTION),
            _reply(GAPS_FOUND_VERDICT),
            [
                SessionEvent(EventKind.USAGE, revision_tokens, ""),
                SessionEvent(EventKind.INTERRUPTED, revision_tokens, PARTIAL_SOLUTION),
            ],
        ]
    )
    result = await SelfRefine(FakeRecovery(session), refinement_config).run(state)
    assert result.status == RunStatus.EXHAUSTED
    assert result.solution == INITIAL_SOLUTION
    assert result.phase == Phase.REVISE
    assert session.interruptions == ONE_ROUND
    assert result.output_tokens == BUDGET_TOKENS


async def test_exhausted_initial_solve_has_no_complete_solution(
    refinement_config: RefinementConfig, state: RefinementState
) -> None:
    """Do not invent a final solution when the first solve is interrupted."""
    session = FakeAgentSession(
        [
            [
                SessionEvent(EventKind.USAGE, BUDGET_TOKENS, ""),
                SessionEvent(EventKind.INTERRUPTED, BUDGET_TOKENS, PARTIAL_SOLUTION),
            ]
        ]
    )
    result = await SelfRefine(FakeRecovery(session), refinement_config).run(state)
    assert result.status == RunStatus.EXHAUSTED
    assert result.solution is None


async def test_pause_preserves_pending_revision_and_original_budget_prompt(
    critique_prompt: str, revise_prompt: str,
) -> None:
    """Resume an interrupted revision without repeating completed phases."""
    budget = TOTAL_UNITS * UNIT_TOKENS
    boundary = PREFIX_UNITS * UNIT_TOKENS
    revision_tokens = boundary - TWO_PHASES * PHASE_TOKENS
    session = FakeAgentSession(
        [
            _reply(INITIAL_SOLUTION),
            _reply(GAPS_FOUND_VERDICT),
            [
                SessionEvent(EventKind.USAGE, revision_tokens, ""),
                SessionEvent(EventKind.INTERRUPTED, revision_tokens, PARTIAL_SOLUTION),
            ],
        ]
    )
    config = RefinementConfig(TWO_ROUNDS, ONE_ROUND)
    state = RefinementState(PROBLEM, budget, boundary)
    state = await SelfRefine(FakeRecovery(session), config).run(state)
    assert state.status == RunStatus.PAUSED
    assert state.phase == Phase.REVISE
    assert state.rounds == ONE_ROUND
    assert state.solution == INITIAL_SOLUTION
    assert str(budget) in session.prompts[INITIAL_COUNT]
    assert session.queued_messages == []
    resumed_state = deepcopy(state)
    resumed_state.pause_at_tokens = None
    resumed_state.status = RunStatus.RUNNING
    resumed_session = FakeAgentSession([_reply(REVISED_SOLUTION), _reply(NO_GAPS_VERDICT)])
    await SelfRefine(FakeRecovery(resumed_session), config).run(resumed_state)
    assert resumed_session.prompts == [revise_prompt, critique_prompt]
    assert resumed_state.status == RunStatus.FINISHED
    assert resumed_state.rounds == TWO_ROUNDS
    assert resumed_state.output_tokens == TWO_PHASES * TWO_PHASES * PHASE_TOKENS
    assert state.status == RunStatus.PAUSED


async def test_warning_in_discarded_phase_is_delivered_again_when_needed() -> None:
    """Rollback discards the warning with its phase so the retained conversation can receive a fresh one."""
    session = FakeAgentSession(
        [
            [
                SessionEvent(EventKind.USAGE, WARNING_CROSSING_TOKENS, ""),
                SessionEvent(EventKind.USAGE, LATER_USAGE_TOKENS, ""),
                SessionEvent(EventKind.INTERRUPTED, LATER_USAGE_TOKENS, PARTIAL_SOLUTION),
            ]
        ]
    )
    state = RefinementState(PROBLEM, BUDGET_TOKENS, LATER_USAGE_TOKENS)
    config = RefinementConfig(ONE_ROUND, ONE_ROUND)
    state = await SelfRefine(FakeRecovery(session), config).run(state)
    assert not state.warning_sent
    state.pause_at_tokens = None
    state.status = RunStatus.RUNNING
    resumed_session = FakeAgentSession([
        [SessionEvent(EventKind.COMPLETED, WARNING_CROSSING_TOKENS, INITIAL_SOLUTION)],
        _reply(NO_GAPS_VERDICT),
    ])
    await SelfRefine(FakeRecovery(resumed_session), config).run(state)
    assert len(resumed_session.queued_messages) == ONE_ROUND
    assert state.status == RunStatus.FINISHED


async def test_response_completed_beyond_budget_cannot_replace_solution(
    refinement_config: RefinementConfig, state: RefinementState
) -> None:
    """Keep the earlier solution when usage is reported after an overrun."""
    session = FakeAgentSession(
        [
            _reply(INITIAL_SOLUTION),
            _reply(GAPS_FOUND_VERDICT),
            [SessionEvent(EventKind.COMPLETED, BUDGET_TOKENS, REVISED_SOLUTION)],
        ]
    )
    result = await SelfRefine(FakeRecovery(session), refinement_config).run(state)
    assert result.status == RunStatus.EXHAUSTED
    assert result.solution == INITIAL_SOLUTION
    assert result.output_tokens == BUDGET_TOKENS + TWO_PHASES * PHASE_TOKENS


async def test_solution_completed_exactly_at_budget_is_retained(
    refinement_config: RefinementConfig, state: RefinementState
) -> None:
    """A complete response at the boundary is still within its allowance."""
    session = FakeAgentSession([[SessionEvent(EventKind.COMPLETED, BUDGET_TOKENS, INITIAL_SOLUTION)]])
    result = await SelfRefine(FakeRecovery(session), refinement_config).run(state)
    assert result.status == RunStatus.EXHAUSTED
    assert result.solution == INITIAL_SOLUTION


@pytest.mark.parametrize("overshoot", (INITIAL_COUNT, PHASE_TOKENS, UNIT_TOKENS + PHASE_TOKENS))
async def test_terminal_usage_respects_pause_boundary(overshoot: int) -> None:
    """Final usage can reach the pause boundary, but an overrun cannot replace the prior solution."""
    boundary = PREFIX_UNITS * UNIT_TOKENS
    state = RefinementState(PROBLEM, TOTAL_UNITS * UNIT_TOKENS, boundary)
    state.phase = Phase.REVISE
    state.solution = INITIAL_SOLUTION
    session = FakeAgentSession([[SessionEvent(EventKind.COMPLETED, boundary + overshoot, REVISED_SOLUTION)]])
    result = await SelfRefine(FakeRecovery(session), RefinementConfig(ONE_ROUND, ONE_ROUND)).run(state)
    assert result.status == RunStatus.PAUSED
    assert result.output_tokens == (INITIAL_COUNT if overshoot else boundary)
    assert result.solution == (INITIAL_SOLUTION if overshoot else REVISED_SOLUTION)


async def test_quoted_clean_verdict_does_not_hide_a_gap(state: RefinementState) -> None:
    """Only the final verdict controls the clean-critique counter."""
    critique = f"Earlier you wrote {NO_GAPS_VERDICT}.\n{GAPS_FOUND_VERDICT}"
    session = _conversation([critique, NO_GAPS_VERDICT])
    config = RefinementConfig(ONE_ROUND, ONE_ROUND)
    result = await SelfRefine(FakeRecovery(session), config).run(state)
    assert result.rounds == TWO_ROUNDS


async def test_malformed_critique_resets_streak_and_proceeds_to_revision(state: RefinementState, critique_prompt: str) -> None:
    """A formatting mistake consumes its round without granting convergence or another critique."""
    critiques = [NO_GAPS_VERDICT, PARTIAL_SOLUTION, NO_GAPS_VERDICT, NO_GAPS_VERDICT]
    session = _conversation(critiques)
    recovery = FakeRecovery(session)
    config = RefinementConfig(ONE_ROUND, TWO_ROUNDS)
    result = await SelfRefine(recovery, config).run(state)
    assert result.status == RunStatus.FINISHED
    assert result.rounds == len(critiques)
    malformed = next((saved for saved in recovery.saved_states if saved.rounds == TWO_ROUNDS))
    assert malformed.no_gap_critiques == INITIAL_COUNT
    assert malformed.phase == Phase.REVISE
    assert session.prompts.count(critique_prompt) == len(critiques)
    assert result.solution == REVISED_SOLUTION


@pytest.mark.parametrize(
    "replies, error, message",
    [
        (
            [
                [
                    SessionEvent(EventKind.USAGE, TWO_PHASES * PHASE_TOKENS, ""),
                    SessionEvent(EventKind.COMPLETED, PHASE_TOKENS, INITIAL_SOLUTION),
                ]
            ],
            ValueError,
            "must not decrease",
        ),
        ([[SessionEvent(EventKind.USAGE, PHASE_TOKENS, "")]], RuntimeError, "without a terminal event"),
        (
            [[SessionEvent(EventKind.INTERRUPTED, PHASE_TOKENS, PARTIAL_SOLUTION)]],
            RuntimeError,
            "without a request",
        ),
    ],
)
async def test_invalid_event_sequence_fails(
    replies: list[list[SessionEvent]],
    error: type[Exception],
    message: str,
    refinement_config: RefinementConfig,
    state: RefinementState,
) -> None:
    """Reject decreasing usage, missing completion, and unrequested interruption."""
    session = FakeAgentSession(replies)
    with pytest.raises(error, match=message):
        await SelfRefine(FakeRecovery(session), refinement_config).run(state)
