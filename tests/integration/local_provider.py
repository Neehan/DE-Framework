"""Deterministic Anthropic Messages endpoint for real CLI tests without paid provider calls."""

import json
from http.server import BaseHTTPRequestHandler
from typing import Any
from uuid import uuid4

from harness.self_refine.constants import NO_GAPS_VERDICT
from harness.utils.constants import TEXT_ENCODING, ZERO_TOKENS

from tests.constants import ONE_ROUND
from tests.integration.constants import (
    BLOCK_INDEX,
    HTTP_OK,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    PROVIDER_RESPONSE,
)


class LocalProvider(BaseHTTPRequestHandler):
    """Serve streaming or ordinary Messages responses; inject the request log through the handler factory."""

    def __init__(self, *args: Any, requests: list[dict[str, Any]], **kwargs: Any) -> None:
        """Bind the test-owned request log before the standard handler processes input."""
        self.requests = requests
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress HTTP access logs so the child process prints only its test report."""

    def do_POST(self) -> None:
        """Return local token counts or a deterministic solve/critique response."""
        self._respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

    def _respond(self, body: dict[str, Any]) -> None:
        """Serve an already decoded request so failure-injection subclasses do not reread its stream."""
        if "count_tokens" in self.path:
            self._write_response(json.dumps({"input_tokens": INPUT_TOKENS}), "application/json")
            return
        self.requests.append(body)
        message = self._make_message(body)
        if body.get("stream"):
            events = self._stream_events(message)
            payload = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
            self._write_response(payload, "text/event-stream")
        else:
            self._write_response(json.dumps(message), "application/json")

    def _make_message(self, body: dict[str, Any]) -> dict[str, Any]:
        """Choose the phase response from the last user message."""
        prompt = json.dumps(body["messages"][-ONE_ROUND]["content"])
        text = NO_GAPS_VERDICT if "Critically review" in prompt else PROVIDER_RESPONSE
        return {
            "id": f"msg_{uuid4().hex}", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                      "cache_creation_input_tokens": ZERO_TOKENS, "cache_read_input_tokens": ZERO_TOKENS},
        }

    def _stream_events(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Encode the Messages stream consumed by the installed native CLI."""
        start = {**message, "content": [], "stop_reason": None,
                 "usage": {**message["usage"], "output_tokens": ONE_ROUND}}
        block = message["content"][BLOCK_INDEX]
        if block["type"] == "tool_use":
            content = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            content = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        return [
            {"type": "message_start", "message": start},
            {"type": "content_block_start", "index": BLOCK_INDEX, "content_block": content},
            {"type": "content_block_delta", "index": BLOCK_INDEX, "delta": delta},
            {"type": "content_block_stop", "index": BLOCK_INDEX},
            {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
             "usage": {"output_tokens": message["usage"]["output_tokens"]}},
            {"type": "message_stop"},
        ]

    def _write_response(self, text: str, content_type: str) -> None:
        """Send one complete response with an explicit byte length."""
        payload = text.encode(TEXT_ENCODING)
        self.send_response(HTTP_OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
