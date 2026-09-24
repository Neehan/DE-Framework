"""Local rate-limit and spend-limit injection for real worker credential recovery."""

import json
from http import HTTPStatus

from tests.constants import ONE_ROUND
from tests.integration.constants import (
    LIMITED_KEYS,
    RATE_LIMIT_STREAM_ERROR,
    SPEND_LIMIT_MODEL,
    SPEND_LIMIT_STREAM_ERROR,
    STREAM_RATE_LIMIT_MODEL,
    STREAM_SPEND_LIMIT_MODEL,
)
from tests.integration.local_provider import LocalProvider
from tests.proxy.constants import SECOND_KEY


class CredentialFailureProvider(LocalProvider):
    """Inject rate limits immediately or spend limits after a saved solve; no overrides required."""

    def do_POST(self) -> None:
        """Select HTTP or streamed errors while leaving a healthy credential available for recovery."""
        credential = self.headers["x-api-key"]
        self.requests.append({"credential": credential})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if credential in LIMITED_KEYS:
            if body["model"] in (SPEND_LIMIT_MODEL, STREAM_SPEND_LIMIT_MODEL):
                if "Critically review" in json.dumps(body["messages"][-ONE_ROUND]):
                    self._reject_spent_key(body["model"])
                else:
                    self._respond(body)
                return
            if body["model"] == STREAM_RATE_LIMIT_MODEL:
                self._write_response(RATE_LIMIT_STREAM_ERROR, "text/event-stream")
                return
            self.send_response(HTTPStatus.TOO_MANY_REQUESTS)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        assert credential == SECOND_KEY
        self._respond(body)

    def _reject_spent_key(self, model: str) -> None:
        """Exercise SDK-level SSE failures and proxy-side HTTP error sanitization independently."""
        if model == STREAM_SPEND_LIMIT_MODEL:
            self._write_response(SPEND_LIMIT_STREAM_ERROR, "text/event-stream")
            return
        payload = json.dumps({"error": {"type": "invalid_request_error", "message": "Your credit balance is too low"}}).encode()
        self.send_response(HTTPStatus.BAD_REQUEST)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
