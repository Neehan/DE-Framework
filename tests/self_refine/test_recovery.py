"""Retry scope, checkpoint ordering, and cancellation of phase-boundary recovery."""

from asyncio import CancelledError, Event, create_task, timeout
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from harness.self_refine.models import RefinementState, RunStatus
from harness.self_refine.recovery import Recovery
from harness.session.connection_manager import ConnectionManager
from harness.session.rate_limit_error import RateLimitError
from harness.session.spend_limit_error import SpendLimitError
from harness.utils.constants import COUNT_INCREMENT, INITIAL_COUNT, ZERO_TOKENS

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    DISCONNECT_MESSAGE,
    EXPECTED_RECOVERY_DELAYS,
    ONE_ROUND,
    PHASE_TOKENS,
    SDK_SESSION_ID,
    TWO_PHASES,
)


async def test_disconnect_retries_with_restored_state(
    recovery: Recovery,
    mock_connections: MagicMock,
    restored_state: RefinementState,
    retry_wait: AsyncMock,
    session_store: MagicMock,
    state: RefinementState,
) -> None:
    """A retried phase receives saved progress rather than the failed mutable state."""
    restored_state.output_tokens = PHASE_TOKENS
    execute = AsyncMock(side_effect=[ConnectionError(DISCONNECT_MESSAGE), restored_state])
    result = await recovery.run(state, execute)
    assert result is restored_state
    assert execute.await_args_list[-ONE_ROUND].args[ZERO_TOKENS] is restored_state
    session_store.open.assert_called_once()
    session_store.restore.assert_called_once()
    session_store.close.assert_called_once()
    retry_wait.assert_awaited_once_with(EXPECTED_RECOVERY_DELAYS[ZERO_TOKENS])
    assert mock_connections.connect.await_args_list[-ONE_ROUND].args[-ONE_ROUND] == restored_state.budget_tokens - PHASE_TOKENS


async def _run_with_shutdown_failure(
    execute: AsyncMock, retry_wait: AsyncMock, session_store: MagicMock, state: RefinementState
) -> RefinementState:
    """Exercise real cleanup policy with a failing first disconnect and injected I/O."""
    options = ClaudeAgentOptions()
    client = create_autospec(ClaudeSDKClient, instance=True)
    connections = ConnectionManager(options, lambda options: client)
    with (
        patch.object(connections, "connect", AsyncMock(return_value=SDK_SESSION_ID)),
        patch.object(connections, "close", AsyncMock(side_effect=[ConnectionError("shutdown failed"), None])),
    ):
        return await Recovery(session_store, connections, retry_wait).run(state, execute)


async def test_shutdown_failure_does_not_abort_recovery(
    restored_state: RefinementState, retry_wait: AsyncMock, session_store: MagicMock, state: RefinementState
) -> None:
    """A broken disconnect does not prevent restoring and retrying a failed phase."""
    execute = AsyncMock(side_effect=[ConnectionError(DISCONNECT_MESSAGE), restored_state])
    assert (
        await _run_with_shutdown_failure(
            execute, retry_wait=retry_wait, session_store=session_store, state=state
        )
        is restored_state
    )
    session_store.restore.assert_called_once()
    session_store.close.assert_called_once()
    retry_wait.assert_awaited_once()


@pytest.mark.parametrize("error", (CancelledError(), ValueError("invalid state"), RateLimitError("rate limited", None), SpendLimitError("spend limit")))
async def test_shutdown_failure_preserves_cancellation_and_permanent_errors(
    retry_wait: AsyncMock, session_store: MagicMock, state: RefinementState, error: BaseException
) -> None:
    """Secondary shutdown errors cannot change the original cancellation or failure."""
    with pytest.raises(type(error)) as raised:
        await _run_with_shutdown_failure(
            AsyncMock(side_effect=error), retry_wait=retry_wait, session_store=session_store, state=state
        )
    assert raised.value is error
    assert "shutdown failed" in "\n".join(error.__notes__)
    retry_wait.assert_not_awaited()
    session_store.restore.assert_not_called()
    assert session_store.close.call_count == ONE_ROUND


async def test_successful_execution_does_not_hide_shutdown_failure(
    retry_wait: AsyncMock, session_store: MagicMock, state: RefinementState
) -> None:
    """Without an earlier failure, shutdown errors must reach the caller."""
    with pytest.raises(ConnectionError, match="shutdown failed"):
        await _run_with_shutdown_failure(
            AsyncMock(return_value=state), retry_wait=retry_wait, session_store=session_store, state=state
        )
    session_store.close.assert_called_once()
    retry_wait.assert_not_awaited()


async def test_checkpoint_shutdown_failure_prevents_saving(
    mock_connections: MagicMock, recovery: Recovery, session_store: MagicMock
) -> None:
    """Never archive native files after an unsuccessful checkpoint shutdown."""
    mock_connections.close.side_effect = ConnectionError("shutdown failed")
    with pytest.raises(ConnectionError, match="shutdown failed"):
        await recovery.checkpoint()
    session_store.checkpoint.assert_not_called()
    mock_connections.connect.assert_not_awaited()


async def test_failed_connection_startup_restores_without_releasing_seed(
    mock_connections: MagicMock,
    recovery: Recovery,
    restored_state: RefinementState,
    retry_wait: AsyncMock,
    session_store: MagicMock,
    state: RefinementState,
) -> None:
    """Startup retries retain the acquired seed lock and use its initial checkpoint."""
    mock_connections.connect.side_effect = [ConnectionError(DISCONNECT_MESSAGE), SDK_SESSION_ID]
    result = await recovery.run(state, AsyncMock(return_value=restored_state))
    assert result is restored_state
    session_store.open.assert_called_once()
    session_store.restore.assert_called_once()
    session_store.close.assert_called_once()
    retry_wait.assert_awaited_once()


async def test_repeated_reconnect_failures_stop_at_retry_limit(
    mock_connections: MagicMock,
    recovery: Recovery,
    retry_wait: AsyncMock,
    session_store: MagicMock,
    state: RefinementState,
) -> None:
    """Repeated connection failures consume the bounded current-phase retry allowance."""
    mock_connections.connect.side_effect = ConnectionError(DISCONNECT_MESSAGE)
    execute = AsyncMock()
    with pytest.raises(ConnectionError):
        await recovery.run(state, execute)
    assert tuple((call.args[ZERO_TOKENS] for call in retry_wait.await_args_list)) == EXPECTED_RECOVERY_DELAYS
    execute.assert_not_awaited()
    session_store.close.assert_called_once()


async def test_successful_checkpoint_resets_retry_allowance(
    recovery: Recovery,
    restored_state: RefinementState,
    retry_wait: AsyncMock,
    session_store: MagicMock,
    state: RefinementState,
) -> None:
    """Two different phases can each recover once when the per-phase allowance is one."""
    calls = INITIAL_COUNT

    async def run_phases(state: RefinementState) -> RefinementState:
        """Fail, commit progress, then fail again in the next phase."""
        nonlocal calls
        calls += COUNT_INCREMENT
        if calls == ONE_ROUND:
            raise ConnectionError(DISCONNECT_MESSAGE)
        if calls == TWO_PHASES:
            await recovery.checkpoint()
            raise ConnectionError(DISCONNECT_MESSAGE)
        return state

    with patch("harness.self_refine.recovery.RECOVERY_MAX_RETRIES", ONE_ROUND):
        result = await recovery.run(state, run_phases)
    assert result is restored_state
    session_store.checkpoint.assert_called_once()
    assert retry_wait.await_count == TWO_PHASES
    assert all((call.args == (EXPECTED_RECOVERY_DELAYS[ZERO_TOKENS],) for call in retry_wait.await_args_list))


async def test_checkpoint_orders_close_save_and_connect(
    mock_connections: MagicMock, recovery: Recovery, session_store: MagicMock
) -> None:
    """Native files are saved after shutdown and before the next connection starts."""
    order: list[str] = []
    mock_connections.close.side_effect = lambda: order.append("close")
    session_store.checkpoint.side_effect = lambda: order.append("checkpoint")
    mock_connections.connect.side_effect = lambda runtime, session_id, fork_session, remaining_tokens: (
        order.append("connect") or SDK_SESSION_ID
    )
    await recovery.checkpoint()
    assert order == ["close", "checkpoint", "connect"]


@pytest.mark.parametrize("error", (ValueError("invalid state"), OSError("disk full")))
async def test_permanent_errors_are_not_retried(
    recovery: Recovery,
    retry_wait: AsyncMock,
    session_store: MagicMock,
    state: RefinementState,
    error: Exception,
) -> None:
    """Configuration and disk errors remain visible instead of consuming transport retries."""
    with pytest.raises(type(error)):
        await recovery.run(state, AsyncMock(side_effect=error))
    retry_wait.assert_not_awaited()
    session_store.restore.assert_not_called()


async def test_corrupt_restore_is_not_retried(
    recovery: Recovery, retry_wait: AsyncMock, session_store: MagicMock, state: RefinementState
) -> None:
    """A transport failure cannot turn checkpoint corruption into a retry loop."""
    session_store.restore.side_effect = ValueError("corrupt checkpoint")
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        await recovery.run(state, AsyncMock(side_effect=ConnectionError(DISCONNECT_MESSAGE)))
    retry_wait.assert_awaited_once()
    session_store.close.assert_called_once()


async def test_duplicate_open_does_not_close_another_owner(
    mock_connections: MagicMock,
    recovery: Recovery,
    retry_wait: AsyncMock,
    session_store: MagicMock,
    state: RefinementState,
) -> None:
    """A runner that failed to acquire storage ownership cannot close its owner's connection."""
    session_store.open.side_effect = RuntimeError("already open")
    with pytest.raises(RuntimeError, match="already open"):
        await recovery.run(state, AsyncMock())
    session_store.close.assert_not_called()
    mock_connections.close.assert_not_awaited()
    retry_wait.assert_not_awaited()


async def test_cancellation_during_backoff_closes_without_retry(
    mock_connections: MagicMock, session_store: MagicMock, state: RefinementState
) -> None:
    """Cancellation during the retry wait closes transport and releases seed ownership."""
    waiting = Event()
    release = Event()

    async def wait(delay: float) -> None:
        """Expose the retry wait without depending on elapsed time."""
        waiting.set()
        await release.wait()

    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        recovery = Recovery(session_store, mock_connections, wait)
        task = create_task(recovery.run(state, AsyncMock(side_effect=ConnectionError(DISCONNECT_MESSAGE))))
        await waiting.wait()
        task.cancel()
        with pytest.raises(CancelledError):
            await task
    session_store.restore.assert_not_called()
    session_store.close.assert_called_once()


async def test_execution_cancellation_is_not_a_retry(
    recovery: Recovery, retry_wait: AsyncMock, session_store: MagicMock, state: RefinementState
) -> None:
    """A caller's cancellation never restarts the phase."""
    with pytest.raises(CancelledError):
        await recovery.run(state, AsyncMock(side_effect=CancelledError()))
    retry_wait.assert_not_awaited()
    session_store.close.assert_called_once()


@pytest.mark.parametrize("status", (RunStatus.FINISHED, RunStatus.EXHAUSTED, RunStatus.PAUSED))
async def test_stopped_states_do_not_open_connections(
    mock_connections: MagicMock, recovery: Recovery, state: RefinementState, status: RunStatus
) -> None:
    """Finished, exhausted, and paused progress can be returned without provider work."""
    state.status = status
    await recovery.run(state, AsyncMock(return_value=state))
    mock_connections.connect.assert_not_awaited()


async def test_invalid_checkpoint_keeps_connection_open(
    recovery: Recovery, session_store: MagicMock, mock_connections: MagicMock, state: RefinementState,
) -> None:
    """Reject invalid progress before closing a usable connection or changing durable state."""
    state.output_tokens = -ONE_ROUND
    with pytest.raises(ValueError, match="nonnegative"):
        await recovery.checkpoint()
    mock_connections.close.assert_not_awaited()
    session_store.checkpoint.assert_not_called()
