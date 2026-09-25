"""Provider-independent tests for cumulative output-token accounting."""

import pytest
from harness.session.token_usage import TokenUsage
from harness.utils.constants import ZERO_TOKENS

from tests.constants import (
    SDK_FINAL_TOKENS,
    SDK_FIRST_TOKENS,
    SDK_INITIAL_TOKENS,
    SDK_MESSAGE_ID,
    SDK_MODEL,
    SDK_SECOND_TOKENS,
)


def test_snapshots_charge_only_new_output(usage: TokenUsage) -> None:
    """Duplicate and stale snapshots cannot increase or reduce known spend."""
    for tokens in (SDK_INITIAL_TOKENS, SDK_FIRST_TOKENS, SDK_FIRST_TOKENS, SDK_INITIAL_TOKENS):
        usage.update_message_tokens(SDK_MESSAGE_ID, tokens)
    assert usage.output_tokens == SDK_FIRST_TOKENS


def test_distinct_messages_accumulate(usage: TokenUsage) -> None:
    """Separate model responses contribute independently to the turn total."""
    usage.update_message_tokens(SDK_MESSAGE_ID, SDK_FIRST_TOKENS)
    usage.update_message_tokens(SDK_MODEL, SDK_SECOND_TOKENS)
    assert usage.output_tokens == SDK_FIRST_TOKENS + SDK_SECOND_TOKENS


def test_final_total_reconciles_unstreamed_output(usage: TokenUsage) -> None:
    """A larger final total includes output that was not reported incrementally."""
    usage.update_message_tokens(SDK_MESSAGE_ID, SDK_FIRST_TOKENS)
    usage.finalize_turn_usage(SDK_FINAL_TOKENS)
    usage.update_message_tokens(SDK_MODEL, SDK_SECOND_TOKENS)
    usage.finalize_turn_usage(SDK_SECOND_TOKENS)
    assert usage.output_tokens == SDK_FINAL_TOKENS + SDK_SECOND_TOKENS


def test_lower_final_total_preserves_observed_output(usage: TokenUsage) -> None:
    """An interruption result cannot erase tokens already charged to the turn."""
    usage.update_message_tokens(SDK_MESSAGE_ID, SDK_FIRST_TOKENS)
    usage.finalize_turn_usage(ZERO_TOKENS)
    assert usage.output_tokens == SDK_FIRST_TOKENS


def test_completed_message_snapshot_is_not_charged_again(usage: TokenUsage) -> None:
    """A repeated snapshot remains deduplicated after its turn is reconciled."""
    usage.update_message_tokens(SDK_MESSAGE_ID, SDK_FIRST_TOKENS)
    usage.finalize_turn_usage(SDK_FINAL_TOKENS)
    usage.update_message_tokens(SDK_MESSAGE_ID, SDK_FIRST_TOKENS)
    usage.finalize_turn_usage(ZERO_TOKENS)
    assert usage.output_tokens == SDK_FINAL_TOKENS


def test_negative_counts_fail_without_changing_usage(usage: TokenUsage) -> None:
    """Reject invalid snapshots and final totals before mutating the ledger."""
    usage.update_message_tokens(SDK_MESSAGE_ID, SDK_FIRST_TOKENS)
    with pytest.raises(ValueError, match="nonnegative"):
        usage.update_message_tokens(SDK_MESSAGE_ID, -SDK_INITIAL_TOKENS)
    with pytest.raises(ValueError, match="nonnegative"):
        usage.finalize_turn_usage(-SDK_INITIAL_TOKENS)
    assert usage.output_tokens == SDK_FIRST_TOKENS


def test_completed_message_true_up_is_sealed_across_new_ledgers() -> None:
    """A larger replay after a result true-up cannot be charged again on reconnect."""
    completed: set[str] = set()
    first = TokenUsage(completed)
    first.update_message_tokens(SDK_MESSAGE_ID, SDK_FIRST_TOKENS)
    first.finalize_turn_usage(SDK_FINAL_TOKENS)
    resumed = TokenUsage(completed)
    resumed.update_message_tokens(SDK_MESSAGE_ID, SDK_FINAL_TOKENS)
    assert resumed.turn_output_tokens == ZERO_TOKENS
    assert resumed.output_tokens == ZERO_TOKENS
    resumed.update_message_tokens(SDK_MODEL, SDK_SECOND_TOKENS)
    resumed.finalize_turn_usage(SDK_SECOND_TOKENS)
    assert resumed.output_tokens == SDK_SECOND_TOKENS
