"""SDK protocol transport with simulated native transcript persistence."""

import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions
from harness.session.constants import (
    CLAUDE_CONFIG_DIR_ENV,
    TRANSCRIPT_EXTENSION,
)
from harness.utils.constants import TEXT_ENCODING

from tests.constants import SDK_PROJECT_DIRECTORY
from tests.support.sdk_transport import SdkTransport


class PersistentSdkTransport(SdkTransport):
    """Exercise archive restoration with the real SDK and locally simulated CLI files."""

    def __init__(self, options: ClaudeAgentOptions) -> None:
        """Bind transcript storage to the options supplied by SessionManager."""
        super().__init__()
        self.options = options
        session_id = options.session_id or Path(str(options.resume)).stem
        self.transcript = (
            Path(options.env[CLAUDE_CONFIG_DIR_ENV])
            / SDK_PROJECT_DIRECTORY
            / f"{session_id}{TRANSCRIPT_EXTENSION}"
        )

    async def connect(self) -> None:
        """Require restored history for resume; real native forks are covered by the CLI integration test."""
        if self.options.resume is not None:
            assert Path(self.options.resume) == self.transcript
            assert self.transcript.is_file()
        else:
            self.transcript.parent.mkdir(parents=True, exist_ok=True)
            self.transcript.touch(exist_ok=False)
        await super().connect()

    async def write(self, data: str) -> None:
        """Record submitted user messages as a CLI would before processing them."""
        await super().write(data)
        message = json.loads(data)
        if message["type"] == "user":
            self._record_message(message)

    def _record_message(self, message: dict[str, Any]) -> None:
        """Append a simulated conversation record to the native session file."""
        with self.transcript.open("a", encoding=TEXT_ENCODING) as handle:
            handle.write(json.dumps(message) + "\n")

    def send(self, message: dict[str, Any]) -> None:
        """Persist simulated CLI output before delivering it to the SDK reader."""
        if message["type"] == "result":
            message = {**message, "session_id": self.transcript.stem}
        self._record_message(message)
        super().send(message)
