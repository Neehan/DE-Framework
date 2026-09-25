"""Fault-injection tests for cancellation, disconnects, and concurrent execution."""

from asyncio import CancelledError, Event, create_task, wait_for
from collections.abc import Callable
from unittest.mock import AsyncMock, patch

import pytest
from harness.self_refine.constants import (
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

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    BUDGET_TOKENS,
    DISCONNECT_MESSAGE,
    INITIAL_SOLUTION,
    ONE_ROUND,
    PARTIAL_SOLUTION,
    PHASE_TOKENS,
    PROBLEM,
    WARNING_CROSSING_TOKENS,
)
from tests.support.async_test_agent_session import AsyncTestAgentSession
from tests.support.fake_recovery import FakeRecovery


async def test_cancellation_during_warning_closes_stream(
    blocked_warning: AsyncMock,
    make_async_agent: Callable[[list[SessionEvent]], AsyncTestAgentSession],
    refinement_config: RefinementConfig,
    state: RefinementState,
    warning_started: Event,
) -> None:
    """Cancellation must propagate and synchronously close a suspended stream."""
    session = make_async_agent([SessionEvent(EventKind.USAGE, WARNING_CROSSING_TOKENS, "")])
    with patch.object(session, "queue", AsyncMock(side_effect=blocked_warning)):
        task = create_task(SelfRefine(FakeRecovery(session), refinement_config).run(state))
        try:
            await wait_for(warning_started.wait(), ASYNC_TEST_TIMEOUT_SECONDS)
        finally:
            task.cancel()
            with pytest.raises(CancelledError):
                await task
    assert session.closed_streams == ONE_ROUND
    assert state.output_tokens == WARNING_CROSSING_TOKENS
    assert state.phase == Phase.SOLVE
    assert not state.warning_sent


async def test_warning_failure_closes_stream_without_claiming_delivery(
    make_async_agent: Callable[[list[SessionEvent]], AsyncTestAgentSession],
    refinement_config: RefinementConfig,
    state: RefinementState,
) -> None:
    """An unsuccessful warning cannot set the sent flag or leave its reader open."""
    session = make_async_agent([SessionEvent(EventKind.USAGE, WARNING_CROSSING_TOKENS, "")])
    failure = ConnectionError(DISCONNECT_MESSAGE)
    with patch.object(session, "queue", AsyncMock(side_effect=failure)):
        with pytest.raises(ConnectionError):
            await SelfRefine(FakeRecovery(session), refinement_config).run(state)
    assert session.closed_streams == ONE_ROUND
    assert not state.warning_sent
    assert state.solution is None


async def test_interrupt_failure_closes_stream_and_preserves_solution(
    make_async_agent: Callable[[list[SessionEvent]], AsyncTestAgentSession],
    refinement_config: RefinementConfig,
    state: RefinementState,
) -> None:
    """An unacknowledged interruption must not claim a successful budget stop."""
    state.phase = Phase.REVISE
    state.solution = INITIAL_SOLUTION
    session = make_async_agent([SessionEvent(EventKind.USAGE, BUDGET_TOKENS, "")])
    failure = ConnectionError(DISCONNECT_MESSAGE)
    with patch.object(session, "interrupt", AsyncMock(side_effect=failure)):
        with pytest.raises(ConnectionError):
            await SelfRefine(FakeRecovery(session), refinement_config).run(state)
    assert session.closed_streams == ONE_ROUND
    assert state.solution == INITIAL_SOLUTION
    assert state.status == RunStatus.RUNNING


async def test_cancellation_after_terminal_response_closes_stream(
    make_async_agent: Callable[[list[SessionEvent]], AsyncTestAgentSession], state: RefinementState
) -> None:
    """Cancellation while draining a terminal response propagates and closes the stream."""
    state.phase = Phase.CRITIQUE
    state.solution = INITIAL_SOLUTION
    session = make_async_agent([SessionEvent(EventKind.COMPLETED, PHASE_TOKENS, NO_GAPS_VERDICT)])
    session.pause_after_terminal = True
    config = RefinementConfig(ONE_ROUND, ONE_ROUND)
    task = create_task(SelfRefine(FakeRecovery(session), config).run(state))
    try:
        await wait_for(session.terminal_reached.wait(), ASYNC_TEST_TIMEOUT_SECONDS)
    finally:
        task.cancel()
        with pytest.raises(CancelledError):
            await task
    assert session.closed_streams == ONE_ROUND


async def test_overlapping_runs_fail_before_starting_another_prompt(
    blocked_warning: AsyncMock,
    make_async_agent: Callable[[list[SessionEvent]], AsyncTestAgentSession],
    refinement_config: RefinementConfig,
    state: RefinementState,
    warning_started: Event,
) -> None:
    """One controller must never interleave two attempts on its conversation."""
    session = make_async_agent([SessionEvent(EventKind.USAGE, WARNING_CROSSING_TOKENS, "")])
    runner = SelfRefine(FakeRecovery(session), refinement_config)
    other_state = RefinementState(PROBLEM, BUDGET_TOKENS, None)
    with patch.object(session, "queue", AsyncMock(side_effect=blocked_warning)):
        task = create_task(runner.run(state))
        try:
            await wait_for(warning_started.wait(), ASYNC_TEST_TIMEOUT_SECONDS)
            with pytest.raises(RuntimeError, match="already running"):
                await runner.run(other_state)
            assert len(session.prompts) == ONE_ROUND
        finally:
            task.cancel()
            with pytest.raises(CancelledError):
                await task


async def test_cutoff_waits_for_terminal_acknowledgement(
    make_async_agent: Callable[[list[SessionEvent]], AsyncTestAgentSession],
    refinement_config: RefinementConfig,
    state: RefinementState,
) -> None:
    """Consume the interruption result and charge any reported output after cutoff."""
    overrun_tokens = BUDGET_TOKENS + PHASE_TOKENS
    session = make_async_agent(
        [
            SessionEvent(EventKind.USAGE, BUDGET_TOKENS, ""),
            SessionEvent(EventKind.USAGE, overrun_tokens, ""),
            SessionEvent(EventKind.INTERRUPTED, overrun_tokens, PARTIAL_SOLUTION),
        ]
    )
    result = await wait_for(
        SelfRefine(FakeRecovery(session), refinement_config).run(state), ASYNC_TEST_TIMEOUT_SECONDS
    )
    assert session.interruptions == ONE_ROUND
    assert result.status == RunStatus.EXHAUSTED
    assert result.output_tokens == overrun_tokens
    assert session.closed_streams == ONE_ROUND
