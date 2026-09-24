"""In-memory CLI protocol transport for exercising the real installed SDK."""

import json
from asyncio import Queue
from collections.abc import AsyncIterator
from typing import Any

from claude_agent_sdk import Transport
from harness.session.constants import SUCCESS_SUBTYPE
from harness.utils.constants import ZERO_TOKENS

from tests.constants import (
    SDK_INITIAL_TOKENS,
    SDK_MESSAGE_ID,
    SDK_MODEL,
    SDK_SESSION_ID,
)


class SdkTransport(Transport):
    """Accept test messages and expose SDK writes without starting a CLI or model."""

    def __init__(self) -> None:
        """Create independent protocol queues and connection state."""
        self.incoming: Queue[dict[str, Any] | None] = Queue()
        self.prompts: Queue[dict[str, Any]] = Queue()
        self.controls: list[str] = []
        self.ready = False
        self.echo_prompts = True
        self.write_failure: Exception | None = None

    async def connect(self) -> None:
        """Open the in-memory connection."""
        self.ready = True

    async def write(self, data: str) -> None:
        """Acknowledge control requests and capture user messages."""
        message = json.loads(data)
        match message["type"]:
            case "control_request":
                self.controls.append(message["request"]["subtype"])
                self.incoming.put_nowait(
                    {
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": message["request_id"],
                            "response": {},
                        },
                    }
                )
            case "user":
                if self.write_failure is not None:
                    raise self.write_failure
                self.prompts.put_nowait(message)
                if self.echo_prompts:
                    self.incoming.put_nowait(message)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield CLI protocol packets until the test closes the stream."""
        while True:
            message = await self.incoming.get()
            if message is None:
                return
            yield message

    async def close(self) -> None:
        """Release the connection and unblock any waiting reader."""
        self.ready = False
        self.incoming.put_nowait(None)

    def is_ready(self) -> bool:
        """Report whether the test connection is open."""
        return self.ready

    async def end_input(self) -> None:
        """Reject unexpected closure of interactive input."""
        raise AssertionError("interactive sessions must keep input open")

    def stream(self, event: dict[str, Any]) -> None:
        """Enqueue a raw Anthropic streaming event inside its SDK envelope."""
        self.send(
            {"type": "stream_event", "uuid": SDK_MESSAGE_ID, "session_id": SDK_SESSION_ID, "event": event}
        )

    def start(self, message_id: str) -> None:
        """Start a model response with the small initial usage snapshot."""
        self.stream(
            {
                "type": "message_start",
                "message": {"id": message_id, "usage": {"output_tokens": SDK_INITIAL_TOKENS}},
            }
        )

    def delta(self, tokens: int) -> None:
        """Report cumulative output for the current model response."""
        self.stream({"type": "message_delta", "usage": {"output_tokens": tokens}})

    def assistant(self, message_id: str, text: str, tokens: int | None) -> None:
        """Deliver a complete assistant text envelope, optionally without usage."""
        self.send(
            {
                "type": "assistant",
                "message": {
                    "id": message_id,
                    "model": SDK_MODEL,
                    "content": [{"type": "text", "text": text}],
                    "usage": None if tokens is None else {"output_tokens": tokens},
                },
            }
        )

    def send(self, message: dict[str, Any]) -> None:
        """Deliver one CLI protocol packet to the SDK reader."""
        self.incoming.put_nowait(message)

    def result(self, text: str | None, tokens: int | None, turns: int, failed: bool) -> None:
        """Deliver a terminal packet, including intentionally missing fields for protocol tests."""
        self.send(
            {
                "type": "result",
                "subtype": "error_during_execution" if failed else SUCCESS_SUBTYPE,
                "duration_ms": ZERO_TOKENS,
                "duration_api_ms": ZERO_TOKENS,
                "is_error": failed,
                "num_turns": turns,
                "session_id": SDK_SESSION_ID,
                "usage": None if tokens is None else {"output_tokens": tokens},
                "result": text,
            }
        )
