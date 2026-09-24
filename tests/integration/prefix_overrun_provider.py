"""Force a native prefix across its cutoff and mark reasoning that must not reach forks."""

import json
from typing import Any

from experiments.constants import PREFIX_TOKENS

from tests.constants import ONE_ROUND
from tests.integration.constants import OUTPUT_TOKENS, PREFIX_OVERRUN_RESPONSE
from tests.integration.local_provider import LocalProvider


class PrefixOverrunProvider(LocalProvider):
    """Inject an over-budget first critique; continuation branches use ordinary local responses."""

    def _make_message(self, body: dict[str, Any]) -> dict[str, Any]:
        """Identify prefix work from its conversation and report a deliberately oversized response."""
        message = super()._make_message(body)
        if ("Critically review" in json.dumps(body["messages"][-ONE_ROUND]["content"])
                and "additional output tokens" not in json.dumps(body["messages"])):
            message["content"] = [{"type": "text", "text": PREFIX_OVERRUN_RESPONSE}]
            message["usage"]["output_tokens"] = PREFIX_TOKENS + OUTPUT_TOKENS
        return message
