"""Output-token accounting independent of SDK message formats."""

from harness.utils.constants import ZERO_TOKENS


class TokenUsage:
    """Update token counts by message ID and reconcile completed turn totals.

    output_tokens covers this ledger's lifetime. No overrides are required.
    """

    def __init__(self, completed_message_ids: set[str]) -> None:
        """Share completed message identities across runs and checkpointed connections."""
        self._completed_message_ids = completed_message_ids
        self._message_tokens: dict[str, int] = {}
        self._completed_tokens = ZERO_TOKENS

    @property
    def output_tokens(self) -> int:
        """Return completed turn totals plus output observed in the current turn."""
        return self._completed_tokens + self.turn_output_tokens

    @property
    def turn_output_tokens(self) -> int:
        """Return fresh output observed since the previous result."""
        return sum(self._message_tokens.values())

    def message_output_tokens(self, message_id: str) -> int:
        """Return observed output for one fresh message in the current turn."""
        return self._message_tokens.get(message_id, ZERO_TOKENS)

    def update_message_tokens(self, message_id: str, output_tokens: int) -> None:
        """Update the message token count and add only its increase to the turn total."""
        self._validate_token_count(output_tokens)
        if message_id in self._completed_message_ids:
            return
        previous = self._message_tokens.get(message_id, ZERO_TOKENS)
        self._message_tokens[message_id] = max(previous, output_tokens)

    def finalize_turn_usage(self, output_tokens: int) -> None:
        """Add the reconciled turn total to completed usage and reset the current turn count."""
        self._validate_token_count(output_tokens)
        self._completed_tokens += max(self.turn_output_tokens, output_tokens)
        self._completed_message_ids.update(self._message_tokens)
        self._message_tokens.clear()

    def _validate_token_count(self, output_tokens: int) -> None:
        """Reject invalid counts before changing the ledger."""
        if output_tokens < ZERO_TOKENS:
            raise ValueError("output_tokens must be nonnegative")
