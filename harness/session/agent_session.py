"""SDK-backed agent execution for one shared refinement conversation."""

import json
from collections.abc import AsyncGenerator, Iterator
from typing import Any, Never
from uuid import uuid4

from claude_agent_sdk import (
    AssistantMessage,
    Message,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    TextBlock,
    UserMessage,
)

from harness.session.connection_manager import ConnectionManager
from harness.session.constants import (
    API_ERROR_PREFIX,
    DEFAULT_SESSION_ID,
    RATE_LIMIT_REJECTED,
    SUCCESS_SUBTYPE,
)
from harness.session.errors import (
    is_spend_limit_error,
    is_transient_provider_error,
    normalize_provider_error,
)
from harness.session.models import EventKind, SessionEvent
from harness.session.rate_limit_error import RateLimitError
from harness.session.token_usage import TokenUsage
from harness.utils.constants import ZERO_TOKENS


class AgentSession:
    """Interpret one conversation through an externally owned ConnectionManager.

    queue submits conversation input; run streams output; interrupt requests a stop. Connection lifetime belongs to the caller; no overrides are required.
    """

    def __init__(self, connection: ConnectionManager, completed_message_ids: set[str]) -> None:
        """Accept transport ownership and checkpointed message identities."""
        self._connection = connection
        self._completed_message_ids = completed_message_ids
        self._stream_ids: dict[str | None, str] = {}
        self._running = False
        self._interrupted = False
        self._pending_message_id: str | None = None
        self._last_assistant_message: AssistantMessage | None = None
        self._has_multiple_text_envelopes = False

    async def run(self, prompt: str) -> AsyncGenerator[SessionEvent, None]:
        """Execute an instruction, yielding cumulative output usage and one terminal event last.

        Queued input, tool turns, and compaction share this run's accounting; the owner must close the connection when a run fails or is cancelled.
        """
        if self._running:
            raise RuntimeError("agent session is already running")
        self._running = True
        self._interrupted = False
        self._stream_ids.clear()
        self._last_assistant_message = None
        self._has_multiple_text_envelopes = False
        usage = TokenUsage(self._completed_message_ids)
        try:
            await self.queue(prompt)
            while True:
                message = await self._connection.receive()
                for event in self._dispatch_sdk_message(message, usage):
                    yield event
                    if event.kind != EventKind.USAGE:
                        return
        finally:
            self._running = False

    async def queue(self, message: str) -> None:
        """Submit input to the SDK before or during a run without waiting for a model response."""
        if not message.strip():
            raise ValueError("message must not be empty")
        self._pending_message_id = str(uuid4())
        await self._connection.query(self._encode_input_message(message))

    async def interrupt(self) -> None:
        """Request a cutoff while the independent reader keeps draining SDK events."""
        if not self._running:
            raise RuntimeError("agent session has no active run to interrupt")
        if not self._interrupted:
            await self._connection.interrupt()
            self._interrupted = True

    def _dispatch_sdk_message(
        self, message: Message, usage: TokenUsage,
    ) -> Iterator[SessionEvent]:
        """Route each SDK message to its acknowledgement, usage, or result handler."""
        match message:
            case UserMessage():
                self._clear_pending_message(message)
            case RateLimitEvent(rate_limit_info=info) if info.status == RATE_LIMIT_REJECTED:
                raise RateLimitError("SDK provider rate_limit rejected the request", info.resets_at)
            case StreamEvent() | AssistantMessage():
                event = self._update_response(message, usage)
                if event is not None:
                    yield event
            case ResultMessage():
                yield from self._resolve_turn_result(message, usage)

    def _clear_pending_message(self, message: UserMessage) -> None:
        """Clear the pending message ID when the SDK echoes that same message back."""
        if message.uuid == self._pending_message_id:
            self._pending_message_id = None

    def _update_response(
        self, message: StreamEvent | AssistantMessage, usage: TokenUsage,
    ) -> SessionEvent | None:
        """Retain the latest fresh assistant message and report newly observed usage."""
        previous_tokens = usage.output_tokens
        match message:
            case StreamEvent():
                self._update_stream_usage(message, usage)
            case AssistantMessage(message_id=message_id, usage=message_usage):
                self._raise_assistant_error_if_needed(message)
                if (message_id is not None and message_id not in self._completed_message_ids
                        and message.parent_tool_use_id is None):
                    self._retain_response_text(message)
                if message_usage is not None:
                    if message_id is None:
                        raise ValueError("SDK assistant usage requires a message ID")
                    usage.update_message_tokens(message_id, message_usage["output_tokens"])
        return self._create_usage_event_if_changed(usage, previous_tokens)

    def _raise_assistant_error_if_needed(self, message: AssistantMessage) -> None:
        """Preserve synthetic API diagnostics and classify known transport failures for retry."""
        text = self._get_assistant_text(message).strip()
        diagnostic = f"SDK provider failure ({message.error}): {text}"
        if message.error is not None or (
            text.lower().startswith(API_ERROR_PREFIX) and (is_transient_provider_error(text) or is_spend_limit_error(text))
        ):
            raise normalize_provider_error(RuntimeError(diagnostic))

    def _retain_response_text(self, message: AssistantMessage) -> None:
        """Keep one text envelope per API response; flag split text as ambiguous."""
        previous = self._last_assistant_message
        if previous is None or previous.message_id != message.message_id:
            self._has_multiple_text_envelopes = False
            self._last_assistant_message = message
        elif self._get_assistant_text(message).strip():
            if self._get_assistant_text(previous).strip():
                self._has_multiple_text_envelopes = True
            self._last_assistant_message = message

    def _resolve_turn_result(
        self, message: ResultMessage, usage: TokenUsage,
    ) -> Iterator[SessionEvent]:
        """Reconcile turn usage, propagate failures, and decide whether this run is complete."""
        text = self._get_response_text(message, usage)
        failed = message.is_error or message.subtype != SUCCESS_SUBTYPE
        if not failed and not self._interrupted and not self._has_fresh_result(message, usage, text):
            if self._pending_message_id is not None:
                return
            raise ConnectionError("SDK returned no fresh completed response")
        previous_tokens = usage.output_tokens
        reported_tokens = usage.turn_output_tokens if message.usage is None else message.usage["output_tokens"]
        usage.finalize_turn_usage(reported_tokens)
        self._last_assistant_message = None
        self._has_multiple_text_envelopes = False
        if failed or self._pending_message_id is not None:
            event = self._create_usage_event_if_changed(usage, previous_tokens)
            if event is not None:
                yield event
        if failed and not self._interrupted:
            self._raise_failed_result(message)
        if self._pending_message_id is None or self._interrupted:
            yield self._create_terminal_event(text, usage.output_tokens)

    def _has_fresh_result(self, message: ResultMessage, usage: TokenUsage, text: str) -> bool:
        """Require positive output from fresh message IDs before accepting a result."""
        return (
            message.num_turns > ZERO_TOKENS and usage.turn_output_tokens > ZERO_TOKENS
            and bool(text.strip())
        )

    def _get_response_text(self, result: ResultMessage, usage: TokenUsage) -> str:
        """Prefer terminal text; otherwise require one fresh, metered assistant text envelope."""
        if result.structured_output is not None:
            return json.dumps(result.structured_output)
        if result.result and result.result.strip():
            return result.result
        assistant = self._last_assistant_message
        if assistant is None or assistant.message_id is None or self._has_multiple_text_envelopes:
            return ""
        if usage.message_output_tokens(assistant.message_id) <= ZERO_TOKENS:
            return ""
        return self._get_assistant_text(assistant)

    def _get_assistant_text(self, assistant: AssistantMessage) -> str:
        """Extract text blocks from one envelope without assembling streamed fragments."""
        parts: list[str] = []
        for block in assistant.content:
            match block:
                case TextBlock(text=text):
                    parts.append(text)
        return "\n".join(parts)

    def _raise_failed_result(self, message: ResultMessage) -> Never:
        """Expose recognized temporary provider failures as recoverable connection errors."""
        diagnostic = f"SDK turn failed ({message.subtype}): {message.errors} {message.result}"
        raise normalize_provider_error(RuntimeError(diagnostic))

    def _create_usage_event_if_changed(self, usage: TokenUsage, previous_tokens: int) -> SessionEvent | None:
        """Create a usage event when the token total changes; otherwise return None."""
        if usage.output_tokens == previous_tokens:
            return None
        return SessionEvent(EventKind.USAGE, usage.output_tokens, "")

    def _update_stream_usage(self, message: StreamEvent, usage: TokenUsage) -> None:
        """Track model response IDs and update their token counts from stream events."""
        event = message.event
        parent = message.parent_tool_use_id
        if event["type"] == "message_start":
            payload = event["message"]
            self._stream_ids[parent] = payload["id"]
            if parent is None and payload["id"] not in self._completed_message_ids:
                self._last_assistant_message = None
                self._has_multiple_text_envelopes = False
            usage.update_message_tokens(payload["id"], payload["usage"]["output_tokens"])
        elif event["type"] == "message_delta":
            usage.update_message_tokens(self._stream_ids[parent], event["usage"]["output_tokens"])

    async def _encode_input_message(self, message: str) -> AsyncGenerator[dict[str, Any], None]:
        """Encode one user message using the SDK's streaming-input interface."""
        yield {
            "type": "user", "uuid": self._pending_message_id,
            "session_id": DEFAULT_SESSION_ID, "parent_tool_use_id": None,
            "message": {"role": "user", "content": message},
        }

    def _create_terminal_event(self, text: str, output_tokens: int) -> SessionEvent:
        """Create an interrupted event or a completed event containing the final response."""
        if self._interrupted:
            return SessionEvent(EventKind.INTERRUPTED, output_tokens, "")
        return SessionEvent(EventKind.COMPLETED, output_tokens, text)
