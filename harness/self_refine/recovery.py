"""Coordinate connection replacement and recovery from durable phase boundaries."""

from collections.abc import Awaitable, Callable

from harness.self_refine.models import RefinementState, RunStatus
from harness.session.agent_session import AgentSession
from harness.session.connection_manager import ConnectionManager
from harness.session.session_manager import SessionManager
from harness.utils.constants import COUNT_INCREMENT, INITIAL_COUNT, RECOVERY_MAX_RETRIES
from harness.utils.retry import wait_before_retry


class Recovery:
    """Own run lifetime and checkpoint coordination with a retry allowance per saved phase.

    run opens the seed and cleans up; session exposes execution; checkpoint closes, saves, and reconnects; restore rolls back a phase before a prefix fork. Inject storage, connections, and timing; no overrides are required.
    """

    def __init__(
        self, sessions: SessionManager, connections: ConnectionManager,
        wait: Callable[[float], Awaitable[None]],
    ) -> None:
        """Accept persistence, transport, and timing without duplicating their state."""
        self._sessions = sessions
        self._connections = connections
        self._wait = wait
        self._session: AgentSession | None = None
        self._retries = INITIAL_COUNT

    @property
    def session(self) -> AgentSession:
        """Return the interpreter for the current connection and checkpoint identities."""
        if self._session is None:
            raise RuntimeError("recovery has no connected session")
        return self._session

    async def run(
        self, initial_state: RefinementState,
        execute: Callable[[RefinementState], Awaitable[RefinementState]],
    ) -> RefinementState:
        """Retry transport failures; propagate rate limits, cancellation, and storage errors."""
        state = self._sessions.open(initial_state)
        self._retries = INITIAL_COUNT
        try:
            while True:
                try:
                    await self._connect_session()
                    result = await execute(state)
                    break
                except (ConnectionError, TimeoutError) as error:
                    await self._close_connection(error)
                    if self._retries >= RECOVERY_MAX_RETRIES:
                        raise
                    await wait_before_retry(error, self._retries, self._wait)
                    self._retries += COUNT_INCREMENT
                    state = self._sessions.restore()
        except BaseException as error:
            await self._close_connection(error)
            raise
        else:
            await self._close_connection(None)
            return result
        finally:
            self._sessions.close()

    async def checkpoint(self) -> None:
        """Close before saving, reset the retry allowance, and reconnect continuing work."""
        self._sessions.session_state.refinement.validate()
        await self._close_connection(None)
        self._sessions.checkpoint()
        self._retries = INITIAL_COUNT
        await self._connect_session()

    async def restore(self) -> RefinementState:
        """Close the active connection before restoring progress, conversation, and workspace together."""
        await self._close_connection(None)
        return self._sessions.restore()

    async def _connect_session(self) -> None:
        """Connect only running work and bind the interpreter to the current saved identities."""
        session_state = self._sessions.session_state
        if session_state.refinement.status != RunStatus.RUNNING:
            return
        session_state.session_id = await self._connections.connect(
            self._sessions.runtime_directory, self._sessions.resume_target, session_state.fork_session,
            session_state.refinement.budget_tokens - session_state.refinement.output_tokens,
        )
        session_state.fork_session = False
        self._session = AgentSession(self._connections, session_state.completed_message_ids)

    async def _close_connection(self, error: BaseException | None) -> None:
        """Discard the interpreter and close, preserving any failure already in progress."""
        self._session = None
        if error is None:
            await self._connections.close()
        else:
            await self._connections.close_after_failure(error)
