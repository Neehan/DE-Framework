"""Container entrypoint receiving only the selected attempt over standard input."""

import asyncio
import json
import sys
from contextlib import nullcontext
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from experiments.registry import EXPERIMENTS
from harness.sandbox.constants import CONTAINER_RUN_DIRECTORY, SUCCESS_EXIT_CODE
from harness.self_refine.models import RefinementConfig
from harness.self_refine.recovery import Recovery
from harness.session.connection_manager import ConnectionManager
from harness.session.constants import LOCAL_TOOLS
from harness.session.session_manager import SessionManager
from harness.utils.constants import (
    DEFAULT_MIN_NO_GAP_CRITIQUES,
)
from harness.utils.worker import run_worker

from launcher.models import RunRequest


async def _run_experiment(request: RunRequest) -> int:
    """Compose the selected experiment with shared session and recovery components."""
    manager = SessionManager(Path(CONTAINER_RUN_DIRECTORY), nullcontext)
    options = ClaudeAgentOptions(model=request.model, allowed_tools=list(LOCAL_TOOLS))
    recovery = Recovery(manager, ConnectionManager(options, ClaudeSDKClient), asyncio.sleep)
    experiment_class = EXPERIMENTS[request.experiment]
    experiment = experiment_class(recovery, RefinementConfig(experiment_class.min_rounds, DEFAULT_MIN_NO_GAP_CRITIQUES), request.sketch)
    await experiment.run(request.initial_state)
    return SUCCESS_EXIT_CODE


if __name__ == "__main__":
    run_worker(_run_experiment(RunRequest(**json.load(sys.stdin))))
