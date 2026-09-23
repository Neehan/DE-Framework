"""Classify SDK provider failures and run ownership conflicts."""

from claude_agent_sdk import CLIConnectionError, CLINotFoundError, ProcessError

from harness.session.constants import (
    RATE_LIMIT_DIAGNOSTIC,
    SPEND_LIMIT_MARKERS,
    TRANSIENT_ERROR_MARKERS,
)
from harness.session.rate_limit_error import RateLimitError
from harness.session.spend_limit_error import SpendLimitError
from harness.utils.constants import INITIAL_COUNT


def is_transient_provider_error(diagnostic: str) -> bool:
    """Recognize temporary provider failures from their diagnostic text."""
    return is_rate_limit_error(diagnostic) or any(marker in diagnostic.lower() for marker in TRANSIENT_ERROR_MARKERS)


def normalize_connection_error(error: Exception) -> Exception:
    """Expose recoverable SDK failures as ConnectionError; retain permanent failures."""
    match error:
        case CLINotFoundError():
            return error
        case CLIConnectionError():
            return ConnectionError(str(error))
        case ProcessError(exit_code=code):
            classified = normalize_provider_error(error)
            if classified is not error:
                return classified
            if code is not None and code < INITIAL_COUNT:
                return ConnectionError(str(error))
    return error


class AttemptAlreadyRunning(RuntimeError):
    """Report an owned seed without treating it as a failed experiment."""


def is_rate_limit_error(diagnostic: str) -> bool:
    """Identify provider rate limits separately from broken connections."""
    return RATE_LIMIT_DIAGNOSTIC.search(diagnostic) is not None


def is_spend_limit_error(diagnostic: str) -> bool:
    """Recognize explicit exhausted-budget diagnostics without treating generic authentication failures as exhaustion."""
    return any(marker in diagnostic.lower() for marker in SPEND_LIMIT_MARKERS)


def normalize_provider_error(error: Exception) -> Exception:
    """Classify provider diagnostics consistently across SDK messages, results, and process failures."""
    diagnostic = str(error)
    if is_spend_limit_error(diagnostic):
        return SpendLimitError(diagnostic)
    if is_rate_limit_error(diagnostic):
        return RateLimitError(diagnostic, None)
    if is_transient_provider_error(diagnostic):
        return ConnectionError(diagnostic)
    return error
