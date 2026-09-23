"""Container entrypoint for one selected stage of the fixed audit sequence."""

import asyncio
import json
import sys
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from harness.sandbox.constants import CONTAINER_RUN_DIRECTORY, SUCCESS_EXIT_CODE
from harness.session.connection_manager import ConnectionManager
from harness.utils.worker import run_worker

from audit.models import AuditRequest
from audit.registry import AUDITS


def _create_connection(options: ClaudeAgentOptions) -> ConnectionManager:
    """Reuse mandatory connection and tool policy with audit-specific options."""
    return ConnectionManager(options, ClaudeSDKClient)


async def _run_audit(request: AuditRequest) -> int:
    """Select the audit implementation and persist its validated result in this container's directory."""
    audit = AUDITS[request.kind](_create_connection, asyncio.sleep)
    await audit.run(request, Path(CONTAINER_RUN_DIRECTORY))
    return SUCCESS_EXIT_CODE


if __name__ == "__main__":
    run_worker(_run_audit(AuditRequest(**json.load(sys.stdin))))
