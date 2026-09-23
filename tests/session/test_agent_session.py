"""SDK protocol tests for streaming, injected warnings, accounting, and cleanup."""

from asyncio import CancelledError, create_task, timeout
from contextlib import aclosing
from typing import Any
from uuid import uuid4

import pytest
from harness.self_refine.constants import NO_GAPS_VERDICT
from harness.self_refine.models import (
    Phase,
    RefinementConfig,
    RefinementState,
    RunStatus,
)
from harness.self_refine.self_refine import SelfRefine
from harness.session.agent_session import AgentSession
from harness.session.models import EventKind, SessionEvent
from harness.session.rate_limit_error import RateLimitError
from harness.session.spend_limit_error import SpendLimitError
from harness.utils.constants import ZERO_TOKENS

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    BUDGET_TOKENS,
    DISCONNECT_MESSAGE,
    INITIAL_SOLUTION,
    ONE_ROUND,
    PROBLEM,
    REVISED_SOLUTION,
    SDK_BUFFER_OVERFLOW_MESSAGES,
    SDK_FINAL_TOKENS,
    SDK_FIRST_TOKENS,
    SDK_INITIAL_TOKENS,
    SDK_MESSAGE_ID,
    SDK_MODEL,
    SDK_SECOND_TOKENS,
    SDK_SESSION_ID,
    SDK_WARNING,
    SYNTHETIC_SUCCESS,
    TRANSIENT_PROVIDER_ERROR,
    WARNING_CROSSING_TOKENS,
)
from tests.support.fake_recovery import FakeRecovery
from tests.support.helpers import collect_events
from tests.support.sdk_harness import SdkHarness
from tests.support.sdk_transport import SdkTransport


async def test_shared_connection_and_per_run_accounting(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Separate phases reuse one connection but start their usage from zero."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        for text in (INITIAL_SOLUTION, REVISED_SOLUTION):
            task = create_task(collect_events(session, PROBLEM))
            await transport.prompts.get()
            transport.start(text)
            transport.delta(SDK_FIRST_TOKENS)
            transport.result(text, SDK_FINAL_TOKENS, ONE_ROUND, False)
            events = await task
            assert events[-ONE_ROUND] == SessionEvent(EventKind.COMPLETED, SDK_FINAL_TOKENS, text)
        assert transport.controls == ["initialize"]


async def test_duplicate_usage_and_compaction_do_not_reset_spend(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Assistant snapshots and stream deltas count once across tool/model calls."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(collect_events(session, PROBLEM))
        await transport.prompts.get()
        transport.start(SDK_MESSAGE_ID)
        transport.delta(SDK_FIRST_TOKENS)
        transport.delta(SDK_FIRST_TOKENS)
        transport.incoming.put_nowait(
            {
                "type": "assistant",
                "message": {
                    "id": SDK_MESSAGE_ID,
                    "model": SDK_MODEL,
                    "content": [],
                    "usage": {"output_tokens": SDK_INITIAL_TOKENS},
                },
            }
        )
        transport.incoming.put_nowait({"type": "system", "subtype": "compact_boundary"})
        transport.start(SDK_MODEL)
        transport.delta(SDK_SECOND_TOKENS)
        transport.result(INITIAL_SOLUTION, ZERO_TOKENS, ONE_ROUND, False)
        events = await task
    totals = [event.output_tokens for event in events]
    assert totals == sorted(totals)
    assert events[-ONE_ROUND].output_tokens == SDK_FIRST_TOKENS + SDK_SECOND_TOKENS
    updates = [event.output_tokens for event in events if event.kind == EventKind.USAGE]
    assert updates == [
        SDK_INITIAL_TOKENS,
        SDK_FIRST_TOKENS,
        SDK_FIRST_TOKENS + SDK_INITIAL_TOKENS,
        SDK_FIRST_TOKENS + SDK_SECOND_TOKENS,
    ]


async def test_warning_injected_without_waiting_for_response(
    session: AgentSession, transport: SdkTransport
) -> None:
    """A warning can join the current turn without requiring a second result."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(collect_events(session, PROBLEM))
        await transport.prompts.get()
        await session.queue(SDK_WARNING)
        warning = await transport.prompts.get()
        assert warning["message"]["content"] == SDK_WARNING
        transport.start(SDK_MESSAGE_ID)
        transport.result(INITIAL_SOLUTION, SDK_FINAL_TOKENS, ONE_ROUND, False)
        events = await task
    assert events[-ONE_ROUND].kind == EventKind.COMPLETED


async def test_zero_turn_success_is_recoverable_not_a_solution(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Synthetic acknowledgements cannot replace a solve or revise response."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.result(SYNTHETIC_SUCCESS, ZERO_TOKENS, ZERO_TOKENS, False)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        with pytest.raises(ConnectionError, match="no fresh completed response"):
            await task


async def test_old_message_and_fresh_output_are_not_double_charged(
    session: AgentSession, transport: SdkTransport
) -> None:
    """A later phase charges fresh output even when the CLI replays the previous phase."""
    first = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.start(SDK_MESSAGE_ID)
    transport.delta(SDK_FIRST_TOKENS)
    transport.result(INITIAL_SOLUTION, SDK_FIRST_TOKENS, ONE_ROUND, False)
    await first
    second = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.start(SDK_MESSAGE_ID)
    transport.delta(SDK_FINAL_TOKENS)
    transport.start(SDK_MODEL)
    transport.delta(SDK_SECOND_TOKENS)
    transport.result(REVISED_SOLUTION, SDK_SECOND_TOKENS, ONE_ROUND, False)
    events = await second
    assert events[-ONE_ROUND].output_tokens == SDK_SECOND_TOKENS


async def test_replay_only_result_cannot_complete_next_phase(
    session: AgentSession, transport: SdkTransport
) -> None:
    """A nonzero replay is still not evidence of a newly completed model response."""
    first = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.start(SDK_MESSAGE_ID)
    transport.delta(SDK_FIRST_TOKENS)
    transport.result(INITIAL_SOLUTION, SDK_FIRST_TOKENS, ONE_ROUND, False)
    await first
    second = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.start(SDK_MESSAGE_ID)
    transport.delta(SDK_FIRST_TOKENS)
    transport.result(INITIAL_SOLUTION, SDK_FIRST_TOKENS, ONE_ROUND, False)
    with pytest.raises(ConnectionError, match="no fresh completed response"):
        await second


@pytest.mark.parametrize(("diagnostic", "exception"), [
    (TRANSIENT_PROVIDER_ERROR, ConnectionError), ("API Error: 429 rate_limit_error", RateLimitError),
    ("Provider spend limit reached", SpendLimitError),
])
async def test_transient_provider_result_is_recoverable(
    session: AgentSession, transport: SdkTransport, diagnostic: str, exception: type[Exception],
) -> None:
    """Transport failures retry within the worker; rate limits transfer recovery to the host."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.result(diagnostic, ZERO_TOKENS, ONE_ROUND, True)
    with pytest.raises(exception):
        await task


@pytest.mark.parametrize(
    "error, diagnostic, exception",
    (("unknown", "API Error: connection reset", ConnectionError),
     (None, "API Error: connection reset", ConnectionError),
     ("authentication_failed", "Invalid API key", RuntimeError),
     ("rate_limit", "API Error: 429", RateLimitError),
     (None, "API Error: rate_limit_error", RateLimitError),
     (None, "API Error: usage limit reached", SpendLimitError),
     ("unknown", "Credit balance is too low", SpendLimitError)),
)
async def test_assistant_failure_keeps_diagnostic_and_retry_classification(
    session: AgentSession, transport: SdkTransport, error: str | None,
    diagnostic: str, exception: type[Exception],
) -> None:
    """Synthetic transient errors retry even without a specific SDK code; authentication still fails."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.send({
        "type": "assistant", "error": error,
        "message": {"id": SDK_MESSAGE_ID, "model": SDK_MODEL,
                    "content": [{"type": "text", "text": diagnostic}]},
    })
    with pytest.raises(exception, match=diagnostic):
        await task


@pytest.mark.parametrize(
    "fresh_message",
    [
        {
            "type": "stream_event",
            "uuid": str(uuid4()),
            "session_id": SDK_SESSION_ID,
            "event": {
                "type": "message_start",
                "message": {"id": str(uuid4()), "usage": {"output_tokens": ZERO_TOKENS}},
            },
        },
        {
            "type": "assistant",
            "message": {
                "id": str(uuid4()),
                "model": SDK_MODEL,
                "content": [{"type": "text", "text": SYNTHETIC_SUCCESS}],
                "usage": {"output_tokens": ZERO_TOKENS},
            },
        },
    ],
)
async def test_zero_token_message_cannot_certify_replayed_result(
    fresh_message: dict[str, Any], session: AgentSession, transport: SdkTransport
) -> None:
    """Complete a real turn, then reject its replay despite a new zero-token envelope."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        first = create_task(collect_events(session, PROBLEM))
        await transport.prompts.get()
        transport.start(SDK_MESSAGE_ID)
        transport.delta(SDK_FIRST_TOKENS)
        transport.result(INITIAL_SOLUTION, SDK_FIRST_TOKENS, ONE_ROUND, False)
        await first
        second = create_task(collect_events(session, PROBLEM))
        await transport.prompts.get()
        transport.incoming.put_nowait(fresh_message)
        transport.start(SDK_MESSAGE_ID)
        transport.delta(SDK_FINAL_TOKENS)
        transport.result(INITIAL_SOLUTION, SDK_FINAL_TOKENS, ONE_ROUND, False)
        with pytest.raises(ConnectionError, match="no fresh completed response"):
            await second


@pytest.mark.parametrize("terminal_text", (None, "", " "))
async def test_missing_terminal_fields_use_complete_assistant_response(
    session: AgentSession, transport: SdkTransport, terminal_text: str | None
) -> None:
    """A successful result can omit both text and usage already received in full."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.assistant(str(uuid4()), INITIAL_SOLUTION, SDK_FIRST_TOKENS)
    transport.result(terminal_text, None, ONE_ROUND, False)
    events = await task
    assert events[-ONE_ROUND] == SessionEvent(EventKind.COMPLETED, SDK_FIRST_TOKENS, INITIAL_SOLUTION)


async def test_terminal_text_and_usage_take_precedence(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Use terminal text and reconcile its larger usage without duplicating assistant output."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_FIRST_TOKENS)
    transport.result(REVISED_SOLUTION, SDK_FINAL_TOKENS, ONE_ROUND, False)
    events = await task
    assert events[-ONE_ROUND] == SessionEvent(EventKind.COMPLETED, SDK_FINAL_TOKENS, REVISED_SOLUTION)


@pytest.mark.parametrize("terminal_text", ("", REVISED_SOLUTION))
async def test_split_text_envelopes_require_terminal_text(
    session: AgentSession, transport: SdkTransport, terminal_text: str
) -> None:
    """Multiple text envelopes for one API message cannot certify a complete fallback."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    message_id = str(uuid4())
    transport.assistant(message_id, INITIAL_SOLUTION, SDK_FIRST_TOKENS)
    transport.assistant(message_id, REVISED_SOLUTION, SDK_SECOND_TOKENS)
    transport.result(terminal_text, SDK_FINAL_TOKENS, ONE_ROUND, False)
    if terminal_text:
        events = await task
        assert events[-ONE_ROUND] == SessionEvent(EventKind.COMPLETED, SDK_FINAL_TOKENS, terminal_text)
    else:
        with pytest.raises(ConnectionError, match="no fresh completed response"):
            await task


async def test_thinking_envelope_does_not_make_text_fallback_ambiguous(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Thinking and text can share an API message ID without splitting the answer."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.incoming.put_nowait(
        {
            "type": "assistant",
            "message": {
                "id": SDK_MESSAGE_ID,
                "model": SDK_MODEL,
                "content": [{"type": "thinking", "thinking": PROBLEM, "signature": ""}],
                "usage": {"output_tokens": SDK_FIRST_TOKENS},
            },
        }
    )
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_SECOND_TOKENS)
    transport.result(None, None, ONE_ROUND, False)
    events = await task
    assert events[-ONE_ROUND] == SessionEvent(EventKind.COMPLETED, SDK_SECOND_TOKENS, INITIAL_SOLUTION)


async def test_complete_text_can_use_usage_from_stream(
    session: AgentSession, transport: SdkTransport
) -> None:
    """An assistant without usage can use counts streamed for that same message."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.start(SDK_MESSAGE_ID)
    transport.delta(SDK_FIRST_TOKENS)
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, None)
    transport.result(None, None, ONE_ROUND, False)
    events = await task
    assert events[-ONE_ROUND] == SessionEvent(EventKind.COMPLETED, SDK_FIRST_TOKENS, INITIAL_SOLUTION)


async def test_replayed_assistant_cannot_supply_missing_text(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Fresh usage in a new phase cannot make an old assistant answer eligible."""
    first = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_FIRST_TOKENS)
    transport.result(None, None, ONE_ROUND, False)
    await first
    second = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_FINAL_TOKENS)
    transport.start(SDK_MODEL)
    transport.delta(SDK_SECOND_TOKENS)
    transport.result(None, None, ONE_ROUND, False)
    with pytest.raises(ConnectionError, match="no fresh completed response"):
        await second


async def test_later_fragment_only_response_does_not_reuse_earlier_text(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Starting a new response invalidates earlier text; fragments alone require recovery."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_FIRST_TOKENS)
    transport.start(SDK_MODEL)
    transport.stream(
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": REVISED_SOLUTION}}
    )
    transport.delta(SDK_SECOND_TOKENS)
    transport.result(None, None, ONE_ROUND, False)
    with pytest.raises(ConnectionError, match="no fresh completed response"):
        await task


async def test_latest_complete_assistant_replaces_earlier_text(
    session: AgentSession, transport: SdkTransport
) -> None:
    """A fresh API response replaces earlier text even when the earlier response was split."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_FIRST_TOKENS)
    transport.assistant(SDK_MESSAGE_ID, REVISED_SOLUTION, SDK_FIRST_TOKENS)
    transport.assistant(SDK_MODEL, REVISED_SOLUTION, SDK_SECOND_TOKENS)
    transport.result(None, None, ONE_ROUND, False)
    events = await task
    assert events[-ONE_ROUND] == SessionEvent(
        EventKind.COMPLETED, SDK_FIRST_TOKENS + SDK_SECOND_TOKENS, REVISED_SOLUTION
    )


async def test_child_assistant_text_cannot_be_the_final_response(
    session: AgentSession, transport: SdkTransport
) -> None:
    """A nested tool agent's response cannot fill in for the parent agent's answer."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.start(SDK_MESSAGE_ID)
    transport.incoming.put_nowait(
        {
            "type": "assistant",
            "parent_tool_use_id": SDK_MESSAGE_ID,
            "message": {
                "id": SDK_MODEL,
                "model": SDK_MODEL,
                "content": [{"type": "text", "text": INITIAL_SOLUTION}],
                "usage": {"output_tokens": SDK_FIRST_TOKENS},
            },
        }
    )
    transport.result(None, None, ONE_ROUND, False)
    with pytest.raises(ConnectionError, match="no fresh completed response"):
        await task


async def test_replayed_text_cannot_replace_a_fresh_complete_answer(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Late replay after a fresh assistant answer changes neither its text nor usage."""
    first = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_FIRST_TOKENS)
    transport.result(None, None, ONE_ROUND, False)
    await first
    second = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.assistant(SDK_MODEL, REVISED_SOLUTION, SDK_SECOND_TOKENS)
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_FINAL_TOKENS)
    transport.result(None, None, ONE_ROUND, False)
    events = await second
    assert events[-ONE_ROUND] == SessionEvent(EventKind.COMPLETED, SDK_SECOND_TOKENS, REVISED_SOLUTION)


async def test_unmetered_assistant_cannot_use_another_messages_usage(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Positive output elsewhere in the turn cannot authenticate a zero-token answer."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.start(SDK_MESSAGE_ID)
    transport.delta(SDK_FIRST_TOKENS)
    transport.assistant(SDK_MODEL, SYNTHETIC_SUCCESS, ZERO_TOKENS)
    transport.result(None, SDK_FIRST_TOKENS, ONE_ROUND, False)
    with pytest.raises(ConnectionError, match="no fresh completed response"):
        await task


async def test_complete_assistant_without_terminal_result_is_incomplete(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Full assistant text does not turn a disconnected stream into a completed phase."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_FIRST_TOKENS)
    transport.incoming.put_nowait(None)
    with pytest.raises(ConnectionError, match="before the phase completed"):
        await task


async def test_interruption_discards_assistant_text_with_missing_terminal_fields(
    session: AgentSession, transport: SdkTransport
) -> None:
    """An interrupted run retains observed usage but cannot publish its assistant text."""
    async with aclosing(session.run(PROBLEM)) as events:
        transport.assistant(SDK_MESSAGE_ID, INITIAL_SOLUTION, SDK_FIRST_TOKENS)
        await anext(events)
        await session.interrupt()
        transport.result(None, None, ZERO_TOKENS, False)
        remaining = [event async for event in events]
        assert remaining[-ONE_ROUND] == SessionEvent(EventKind.INTERRUPTED, SDK_FIRST_TOKENS, "")


async def test_buffered_result_before_warning_ack_is_not_terminal(
    session: AgentSession, transport: SdkTransport
) -> None:
    """A separately processed warning stays in the same phase and budget."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(collect_events(session, PROBLEM))
        await transport.prompts.get()
        transport.echo_prompts = False
        await session.queue(SDK_WARNING)
        warning = await transport.prompts.get()
        transport.start(SDK_MESSAGE_ID)
        transport.result(INITIAL_SOLUTION, SDK_FIRST_TOKENS, ONE_ROUND, False)
        transport.incoming.put_nowait(warning)
        transport.start(SDK_MODEL)
        transport.result(REVISED_SOLUTION, SDK_SECOND_TOKENS, ONE_ROUND, False)
        events = await task
    terminal = [event for event in events if event.kind != EventKind.USAGE]
    assert terminal == [
        SessionEvent(EventKind.COMPLETED, SDK_FIRST_TOKENS + SDK_SECOND_TOKENS, REVISED_SOLUTION)
    ]


async def test_warning_before_run_is_charged_to_that_run(
    session: AgentSession, transport: SdkTransport
) -> None:
    """A low-budget resumed phase can warn before submitting its instruction."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        await session.queue(SDK_WARNING)
        await transport.prompts.get()
        transport.start(SDK_MESSAGE_ID)
        transport.result(INITIAL_SOLUTION, SDK_FIRST_TOKENS, ONE_ROUND, False)
        task = create_task(collect_events(session, PROBLEM))
        await transport.prompts.get()
        transport.start(SDK_MODEL)
        transport.result(REVISED_SOLUTION, SDK_SECOND_TOKENS, ONE_ROUND, False)
        events = await task
    assert events[-ONE_ROUND].output_tokens == SDK_FIRST_TOKENS + SDK_SECOND_TOKENS
    assert events[-ONE_ROUND].text == REVISED_SOLUTION


async def test_interrupt_does_not_deadlock_behind_sdk_buffer(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Control replies remain reachable with more output than the SDK can buffer."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        async with aclosing(session.run(PROBLEM)) as events:
            transport.start(SDK_MESSAGE_ID)
            await anext(events)
            for _ in range(SDK_BUFFER_OVERFLOW_MESSAGES):
                transport.delta(SDK_FIRST_TOKENS)
            await session.interrupt()
            await session.interrupt()
            transport.result(INITIAL_SOLUTION, ZERO_TOKENS, ONE_ROUND, True)
            remaining = [event async for event in events]
    assert transport.controls == ["initialize", "interrupt"]
    assert remaining[-ONE_ROUND] == SessionEvent(EventKind.INTERRUPTED, SDK_FIRST_TOKENS, "")


async def test_owner_closes_connection_after_partial_run(
    sdk: SdkHarness[SdkTransport], session: AgentSession, transport: SdkTransport
) -> None:
    """The owner closes the transport after a partially consumed run."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        async with aclosing(session.run(PROBLEM)) as events:
            transport.start(SDK_MESSAGE_ID)
            await anext(events)
        await sdk.connection.close()
        assert not transport.ready
        with pytest.raises(RuntimeError, match="not connected"):
            await collect_events(session, PROBLEM)


async def test_cancellation_closes_reader_and_propagates(
    sdk: SdkHarness[SdkTransport], session: AgentSession, transport: SdkTransport
) -> None:
    """Cancelling a waiting run releases the SDK transport synchronously."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(collect_events(session, PROBLEM))
        await transport.prompts.get()
        task.cancel()
        with pytest.raises(CancelledError):
            await task
        await sdk.connection.close()
        assert not transport.ready


async def test_disconnect_propagates_without_completed_event(
    sdk: SdkHarness[SdkTransport], session: AgentSession, transport: SdkTransport
) -> None:
    """A dead transport is an incomplete phase, never a successful solution."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(collect_events(session, PROBLEM))
        await transport.prompts.get()
        transport.incoming.put_nowait(None)
        with pytest.raises(ConnectionError, match="before the phase completed"):
            await task
        await sdk.connection.close()
        assert not transport.ready


async def test_failed_result_preserves_usage_and_raises(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Expose reported spend before propagating an SDK failure."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        async with aclosing(session.run(PROBLEM)) as events:
            transport.start(SDK_MESSAGE_ID)
            await anext(events)
            transport.result(INITIAL_SOLUTION, SDK_FINAL_TOKENS, ONE_ROUND, True)
            usage = await anext(events)
            assert usage.output_tokens == SDK_FINAL_TOKENS
            with pytest.raises(RuntimeError, match="SDK turn failed"):
                await anext(events)


async def test_overlapping_run_is_rejected_without_closing_first(
    session: AgentSession, transport: SdkTransport
) -> None:
    """A second consumer cannot steal messages from an active phase."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        first = create_task(collect_events(session, PROBLEM))
        await transport.prompts.get()
        with pytest.raises(RuntimeError, match="already running"):
            await collect_events(session, PROBLEM)
        transport.start(SDK_MESSAGE_ID)
        transport.result(INITIAL_SOLUTION, SDK_FINAL_TOKENS, ONE_ROUND, False)
        assert (await first)[-ONE_ROUND].text == INITIAL_SOLUTION


async def test_warning_write_failure_closes_active_run(
    sdk: SdkHarness[SdkTransport], session: AgentSession, transport: SdkTransport
) -> None:
    """A failed injection propagates through the caller's generator cleanup."""
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        with pytest.raises(ConnectionError, match=DISCONNECT_MESSAGE):
            async with aclosing(session.run(PROBLEM)) as events:
                transport.start(SDK_MESSAGE_ID)
                await anext(events)
                transport.write_failure = ConnectionError(DISCONNECT_MESSAGE)
                await session.queue(SDK_WARNING)
        await sdk.connection.close()
        assert not transport.ready


async def test_self_refine_warns_then_critiques_in_same_session(
    session: AgentSession, transport: SdkTransport
) -> None:
    """Run a complete solve and critique through the concrete session adapter."""
    state = RefinementState(PROBLEM, BUDGET_TOKENS, None)
    refine = SelfRefine(FakeRecovery(session), RefinementConfig(ONE_ROUND, ONE_ROUND))
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(refine.run(state))
        await transport.prompts.get()
        transport.start(SDK_MESSAGE_ID)
        transport.delta(WARNING_CROSSING_TOKENS)
        warning = await transport.prompts.get()
        assert str(BUDGET_TOKENS - WARNING_CROSSING_TOKENS) in warning["message"]["content"]
        transport.result(INITIAL_SOLUTION, WARNING_CROSSING_TOKENS, ONE_ROUND, False)
        critique = await transport.prompts.get()
        assert "Critically review" in critique["message"]["content"]
        transport.start(SDK_MODEL)
        transport.result(NO_GAPS_VERDICT, SDK_FINAL_TOKENS, ONE_ROUND, False)
        result = await task
    assert result.status == RunStatus.FINISHED
    assert result.solution == INITIAL_SOLUTION
    assert result.output_tokens == WARNING_CROSSING_TOKENS + SDK_FINAL_TOKENS
    assert result.warning_sent


async def test_exhausted_revision_keeps_previous_solution(
    session: AgentSession, transport: SdkTransport
) -> None:
    """A streamed cutoff interrupts once and never replaces a finished solution."""
    state = RefinementState(PROBLEM, BUDGET_TOKENS, None)
    state.phase = Phase.REVISE
    state.solution = INITIAL_SOLUTION
    refine = SelfRefine(FakeRecovery(session), RefinementConfig(ONE_ROUND, ONE_ROUND))
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(refine.run(state))
        await transport.prompts.get()
        transport.start(SDK_MESSAGE_ID)
        transport.delta(BUDGET_TOKENS)
        transport.result(REVISED_SOLUTION, ZERO_TOKENS, ONE_ROUND, True)
        result = await task
    assert result.status == RunStatus.EXHAUSTED
    assert result.solution == INITIAL_SOLUTION
    assert result.output_tokens == BUDGET_TOKENS
    assert transport.controls == ["initialize", "interrupt"]


async def test_rate_limit_event_transfers_recovery_to_host(session: AgentSession, transport: SdkTransport) -> None:
    """A native SDK rejection must not enter the bounded connection retry loop."""
    task = create_task(collect_events(session, PROBLEM))
    await transport.prompts.get()
    transport.send({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"},
                    "uuid": SDK_MESSAGE_ID, "session_id": SDK_SESSION_ID})
    with pytest.raises(RateLimitError):
        await task
