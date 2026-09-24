"""Session double with real scheduling points and injectable stream failures."""

from asyncio import Event, sleep
from collections.abc import AsyncGenerator
from contextlib import aclosing

from harness.session.models import EventKind, SessionEvent
from harness.utils.constants import COUNT_INCREMENT, INITIAL_COUNT

from tests.constants import EVENT_LOOP_YIELD_SECONDS
from tests.support.fake_agent_session import FakeAgentSession


class AsyncTestAgentSession(FakeAgentSession):
    """Retain streams for cleanup assertions; optionally block or fail at termination."""

    def __init__(self, replies: list[list[SessionEvent]]) -> None:
        """Add deterministic scheduling controls to the scripted session."""
        super().__init__(replies)
        self.streams: list[AsyncGenerator[SessionEvent, None]] = []
        self.closed_streams = INITIAL_COUNT
        self.failure: Exception | None = None
        self.pause_after_terminal = False
        self.terminal_reached = Event()
        self.release_terminal = Event()

    def run(self, prompt: str) -> AsyncGenerator[SessionEvent, None]:
        """Keep a strong reference so garbage collection cannot hide missing cleanup."""
        stream = self._stream(prompt)
        self.streams.append(stream)
        return stream

    async def _stream(self, prompt: str) -> AsyncGenerator[SessionEvent, None]:
        """Yield scripted events with actual task switches and observable cleanup."""
        try:
            async with aclosing(super().run(prompt)) as events:
                async for event in events:
                    await sleep(EVENT_LOOP_YIELD_SECONDS)
                    yield event
                    if event.kind != EventKind.USAGE:
                        self.terminal_reached.set()
                        if self.pause_after_terminal:
                            await self.release_terminal.wait()
            if self.failure is not None:
                raise self.failure
        finally:
            self.closed_streams += COUNT_INCREMENT

    async def close_streams(self) -> None:
        """Release retained generators during test cleanup, including failing tests."""
        for stream in self.streams:
            await stream.aclose()
