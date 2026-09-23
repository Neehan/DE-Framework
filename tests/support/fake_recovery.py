"""In-memory phase execution for focused refinement-loop tests."""

from collections.abc import Awaitable, Callable
from copy import deepcopy

from harness.self_refine.models import RefinementState
from harness.self_refine.recovery import Recovery
from harness.session.agent_session import AgentSession
from harness.utils.constants import COUNT_INCREMENT


class FakeRecovery(Recovery):
    """Expose an injected executor and snapshot progress without disk or connection work."""

    def __init__(self, session: AgentSession) -> None:
        """Accept the executor whose behavior the refinement test controls."""
        self._executor = session
        self._state: RefinementState | None = None
        self.saved_states: list[RefinementState] = []

    @property
    def session(self) -> AgentSession:
        """Return the injected executor."""
        return self._executor

    async def run(
        self, initial_state: RefinementState,
        execute: Callable[[RefinementState], Awaitable[RefinementState]],
    ) -> RefinementState:
        """Execute refinement directly; retry behavior has separate integration tests."""
        self._state = initial_state
        await self.checkpoint()
        return await execute(initial_state)

    async def checkpoint(self) -> None:
        """Keep independent copies so later phases cannot change earlier checkpoints."""
        assert self._state is not None
        self.saved_states.append(deepcopy(self._state))

    async def restore(self) -> RefinementState:
        """Replace live progress with an independent copy of the last saved phase."""
        self._state = deepcopy(self.saved_states[-COUNT_INCREMENT])
        return self._state
