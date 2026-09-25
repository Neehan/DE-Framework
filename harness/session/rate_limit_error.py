"""Distinguish credential availability from retryable connection failures."""


class RateLimitError(RuntimeError):
    """Stop this worker so the host can await credentials outside inference deadlines."""

    def __init__(self, message: str, resets_at: float | None) -> None:
        """Carry an optional provider reset timestamp to the host credential pool."""
        super().__init__(message)
        self.resets_at = resets_at
