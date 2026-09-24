"""Script native tool calls through the real CLI and filtering proxy."""

import json
from typing import Any

from tests.integration.constants import TOOL_PROBLEM, TOOL_STEPS
from tests.integration.local_provider import LocalProvider
from tests.integration.messages import iter_message_blocks


class ToolProvider(LocalProvider):
    """Reuse local inference transport; choose the next tool from conversation history without server state."""

    def _make_message(self, body: dict[str, Any]) -> dict[str, Any]:
        """Request each native tool once, then let the ordinary solve/critique responses finish the run."""
        message = super()._make_message(body)
        history = json.dumps(body["messages"])
        if TOOL_PROBLEM not in history or not body.get("tools"):
            return message
        completed = {block["id"] for block in iter_message_blocks(body["messages"]) if block["type"] == "tool_use"}
        for tool_id, (name, arguments) in TOOL_STEPS.items():
            if tool_id in completed:
                continue
            assert name in {tool["name"] for tool in body["tools"]}, f"Native tool missing: {name}"
            message["content"] = [{"type": "tool_use", "id": tool_id, "name": name, "input": arguments}]
            message["stop_reason"] = "tool_use"
            break
        return message
