"""Local provider that returns real SDK structured-output tool calls for both audits."""

from typing import Any

from harness.session.constants import STRUCTURED_OUTPUT_TOOL

from tests.audit.constants import CORRECTNESS_RESULT, STEP_RESULT
from tests.integration.local_provider import LocalProvider


class AuditProvider(LocalProvider):
    """Exercise the native CLI's structured output and tool policy without paid calls."""

    def _make_message(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return the requested schema through the CLI's local StructuredOutput tool."""
        message = super()._make_message(body)
        tool, = [tool for tool in body["tools"] if tool["name"] == STRUCTURED_OUTPUT_TOOL]
        verdict = STEP_RESULT if "steps" in tool["input_schema"]["properties"] else CORRECTNESS_RESULT
        message["content"] = [{"type": "tool_use", "id": "tool-audit-result", "name": STRUCTURED_OUTPUT_TOOL, "input": verdict}]
        message["stop_reason"] = "tool_use"
        return message
