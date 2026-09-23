"""Validate the narrow inference API before any request reaches the trusted provider."""

import json
from typing import Any, Never, cast

from jsonschema import Draft202012Validator

from harness.proxy.constants import ALLOWED_BETAS, REQUEST_SCHEMA


class RequestPolicy:
    """Validate JSON and beta flags against explicit allowlists; unknown capabilities fail closed."""

    def __init__(self) -> None:
        """Compile the shared inference schema once for this proxy's lifetime."""
        self._validator = Draft202012Validator(REQUEST_SCHEMA)

    def parse(self, content: bytes) -> dict[str, Any]:
        """Return validated JSON for re-encoding, avoiding parser disagreements with upstreams."""
        body = json.loads(content, parse_constant=self._reject_nonfinite)
        self._validator.validate(body)
        return cast(dict[str, Any], body)

    def validate_betas(self, value: str) -> str:
        """Allow only reviewed inference features, never arbitrary experimental provider capabilities."""
        betas = {part.strip() for part in value.split(",") if part.strip()}
        if betas - ALLOWED_BETAS:
            raise ValueError("unapproved provider beta")
        return ",".join(sorted(betas))

    @staticmethod
    def _reject_nonfinite(value: str) -> Never:
        """Reject nonstandard JSON numbers before validating or forwarding the body."""
        raise ValueError("nonfinite JSON number")
