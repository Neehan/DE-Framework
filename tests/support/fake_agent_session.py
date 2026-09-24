"""Scripted session double; no provider calls or filesystem state."""

from collections.abc import AsyncGenerator

from harness.session.agent_session import AgentSession
from harness.session.models import SessionEvent
from harness.utils.constants import COUNT_INCREMENT, INITIAL_COUNT


class FakeAgentSession(AgentSession):
    """Replay one event sequence per run and record queued input and interrupts."""

    def __init__(self, replies: list[list[SessionEvent]]) -> None:
        """Accept the exact replies expected during a test."""
        self.replies = replies
        self.prompts: list[str] = []
        self.queued_messages: list[str] = []
        self.interruptions = INITIAL_COUNT
        self._reply_index = INITIAL_COUNT

    async def run(self, prompt: str) -> AsyncGenerator[SessionEvent, None]:
        """Yield the next scripted reply without interpreting the prompt."""
        if self._reply_index >= len(self.replies):
            raise AssertionError("unexpected session run")
        self.prompts.append(prompt)
        reply = self.replies[self._reply_index]
        self._reply_index += COUNT_INCREMENT
        for event in reply:
            yield event

    async def queue(self, message: str) -> None:
        """Record injected input independently of phase instructions."""
        self.queued_messages.append(message)

    async def interrupt(self) -> None:
        """Record the request; the scripted run supplies its terminal event."""
        self.interruptions += COUNT_INCREMENT
