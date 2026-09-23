"""SDK credential failures retain reset information across the worker process boundary."""

from contextlib import aclosing
from unittest.mock import create_autospec

import pytest
from claude_agent_sdk import RateLimitEvent, RateLimitInfo
from harness.sandbox.constants import SUCCESS_EXIT_CODE
from harness.session.agent_session import AgentSession
from harness.session.connection_manager import ConnectionManager
from harness.utils.constants import RATE_LIMIT_EXIT_CODE, RATE_LIMIT_RESET_PREFIX
from harness.utils.worker import run_worker

from tests.constants import PROBLEM, SDK_MESSAGE_ID, SDK_SESSION_ID
from tests.proxy.constants import POOL_NOW, RESET_DELAY


@pytest.mark.parametrize("resets_at", [None, int(POOL_NOW + RESET_DELAY)])
def test_sdk_reset_is_reported_to_host(resets_at: int | None, capsys: pytest.CaptureFixture[str]) -> None:
    """A rejected SDK event becomes exit 75 plus a reset report, including the explicit unknown-reset case."""
    connection = create_autospec(ConnectionManager, instance=True)
    connection.receive.return_value = RateLimitEvent(
        rate_limit_info=RateLimitInfo(status="rejected", resets_at=resets_at),
        uuid=SDK_MESSAGE_ID, session_id=SDK_SESSION_ID,
    )

    async def operation() -> int:
        """Run the real agent interpreter without starting a CLI or contacting a provider."""
        async with aclosing(AgentSession(connection, set()).run(PROBLEM)) as events:
            async for _ in events:
                pass
        return SUCCESS_EXIT_CODE

    with pytest.raises(SystemExit) as exited:
        run_worker(operation())
    assert exited.value.code == RATE_LIMIT_EXIT_CODE
    reset = "" if resets_at is None else str(resets_at)
    assert capsys.readouterr().err == f"{RATE_LIMIT_RESET_PREFIX}{reset}\n"
