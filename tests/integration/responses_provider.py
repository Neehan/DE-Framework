"""Deterministic Responses stream behind the real LiteLLM ChatGPT adapter."""

import json
from typing import Any

from harness.self_refine.constants import NO_GAPS_VERDICT
from harness.session.constants import STRUCTURED_OUTPUT_TOOL
from harness.utils.constants import INITIAL_COUNT

from tests.audit.constants import CORRECTNESS_RESULT, STEP_RESULT
from tests.codex.constants import (
    RESPONSE_ID,
    RESPONSES_CREATED,
    TOOL_CALL_ID,
    TOOL_CONTENT,
    TOOL_FILENAME,
)
from tests.constants import ONE_ROUND
from tests.integration.constants import INPUT_TOKENS, OUTPUT_TOKENS, PROVIDER_RESPONSE
from tests.integration.local_provider import LocalProvider


class ResponsesProvider(LocalProvider):
    """Return one Write call, then phase replies with exact usage; reuse local server ownership."""

    def do_POST(self) -> None:
        """Serve complete Responses events, requiring all calls to use the streaming endpoint."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/responses" and body["stream"]
        self.requests.append(body)
        item = self._output_item(body)
        response = {"id": RESPONSE_ID, "object": "response", "created_at": RESPONSES_CREATED,
                    "model": body["model"], "status": "completed", "output": [item],
                    "parallel_tool_calls": False, "tool_choice": "auto", "tools": body.get("tools", []),
                    "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                              "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS}}
        events = [{"type": "response.created", "response": {**response, "output": [], "status": "in_progress"}},
                  *self._item_events(item), {"type": "response.completed", "response": response}]
        for index, event in enumerate(events):
            event["sequence_number"] = index
        self._write_response("".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events), "text/event-stream")

    def _output_item(self, body: dict[str, Any]) -> dict[str, Any]:
        """Require a real Write tool result before returning phase text."""
        structured = [tool for tool in body.get("tools", []) if tool["name"] == STRUCTURED_OUTPUT_TOOL]
        if structured:
            tool, = structured
            verdict = STEP_RESULT if "steps" in tool["parameters"]["properties"] else CORRECTNESS_RESULT
            return {"type": "function_call", "id": "fc_audit", "call_id": TOOL_CALL_ID, "name": STRUCTURED_OUTPUT_TOOL,
                    "arguments": json.dumps(verdict), "status": "completed"}
        if body.get("tools") and not any(request.get("tools") for request in self.requests[:-ONE_ROUND]):
            return {"type": "function_call", "id": "fc_gateway", "call_id": TOOL_CALL_ID, "name": "Write",
                    "arguments": json.dumps({"file_path": TOOL_FILENAME, "content": TOOL_CONTENT}), "status": "completed"}
        prompt = json.dumps(body["input"][-ONE_ROUND])
        text = NO_GAPS_VERDICT if "Critically review" in prompt else PROVIDER_RESPONSE
        return {"type": "message", "id": "msg_gateway", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}

    def _item_events(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Provide text or tool argument deltas and completed items for LiteLLM translation."""
        common = {"output_index": INITIAL_COUNT, "item_id": item["id"]}
        if item["type"] == "function_call":
            return [
                {"type": "response.output_item.added", "output_index": INITIAL_COUNT, "item": {**item, "arguments": ""}},
                {"type": "response.function_call_arguments.delta", **common, "delta": item["arguments"]},
                {"type": "response.function_call_arguments.done", **common, "arguments": item["arguments"]},
                {"type": "response.output_item.done", "output_index": INITIAL_COUNT, "item": item},
            ]
        part = item["content"][INITIAL_COUNT]
        return [
            {"type": "response.output_item.added", "output_index": INITIAL_COUNT, "item": {**item, "content": []}},
            {"type": "response.content_part.added", **common, "content_index": INITIAL_COUNT, "part": {**part, "text": ""}},
            {"type": "response.output_text.delta", **common, "content_index": INITIAL_COUNT, "delta": part["text"]},
            {"type": "response.output_text.done", **common, "content_index": INITIAL_COUNT, "text": part["text"]},
            {"type": "response.content_part.done", **common, "content_index": INITIAL_COUNT, "part": part},
            {"type": "response.output_item.done", "output_index": INITIAL_COUNT, "item": item},
        ]
