"""Create saved verdicts and drive structured SDK results without provider calls."""

import json
from pathlib import Path
from typing import Any

from audit.constants import AUDIT_FILENAME

from tests.constants import ONE_ROUND, SDK_FINAL_TOKENS, SDK_MESSAGE_ID, SDK_SESSION_ID
from tests.support.sdk_transport import SdkTransport


def save_audit(directory: Path, checkpoints: dict[str, Any]) -> Path:
    """Create a per-seed record using the production filename."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / AUDIT_FILENAME
    path.write_text(json.dumps({"checkpoints": checkpoints}))
    return path


def send_verdict(transport: SdkTransport, verdict: dict[str, Any]) -> None:
    """Send a metered structured-output result with no fallback text."""
    transport.start(SDK_MESSAGE_ID)
    transport.delta(SDK_FINAL_TOKENS)
    transport.send({
        "type": "result", "subtype": "success", "duration_ms": ONE_ROUND, "duration_api_ms": ONE_ROUND,
        "is_error": False, "num_turns": ONE_ROUND, "session_id": SDK_SESSION_ID,
        "usage": {"output_tokens": SDK_FINAL_TOKENS}, "result": "", "structured_output": verdict,
    })
