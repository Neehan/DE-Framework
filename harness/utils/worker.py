"""Common worker termination and host credential-recovery exit codes."""

import sys
from collections.abc import Coroutine
from typing import Any

from harness.session.rate_limit_error import RateLimitError
from harness.session.spend_limit_error import SpendLimitError
from harness.utils.asyncio import run_process
from harness.utils.constants import (
    RATE_LIMIT_EXIT_CODE,
    RATE_LIMIT_RESET_PREFIX,
    SPEND_LIMIT_EXIT_CODE,
)


def run_worker(operation: Coroutine[Any, Any, int]) -> None:
    """Run one container task and report credential failures to the owning host sandbox."""
    try:
        code = run_process(operation)
    except RateLimitError as error:
        reset = "" if error.resets_at is None else str(error.resets_at)
        print(f"{RATE_LIMIT_RESET_PREFIX}{reset}", file=sys.stderr, flush=True)
        code = RATE_LIMIT_EXIT_CODE
    except SpendLimitError:
        code = SPEND_LIMIT_EXIT_CODE
    raise SystemExit(code)
