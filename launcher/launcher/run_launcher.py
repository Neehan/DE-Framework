"""Prepare solver checkpoints and execute ordinary experiment attempts."""

import logging
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from harness.sandbox.sandbox import Sandbox
from harness.self_refine.models import RunStatus
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import COUNT_INCREMENT

from launcher.launcher.base import BaseLauncher
from launcher.models import BatchResult, Problem, RunRequest


class RunLauncher(BaseLauncher[RunRequest]):
    """Prepare and execute solver attempts; ContinuationLauncher overrides request preparation and execution."""

    async def _prepare_attempts(self, problems: list[Problem], provider_model: str, result: BatchResult) -> list[tuple[Path, RunRequest]]:
        """Validate saved identity, repair stopped exports, and collect only work needing a container."""
        pending = []
        for problem, directory in self._iter_attempts(problems):
            with self._report_attempt_errors(directory, result), RunLock(directory):
                request = self._prepare_request(directory, problem, provider_model)
                state = SessionManager(directory, nullcontext).prepare_attempt(request.initial_state)
                if state.status != RunStatus.RUNNING:
                    result.skipped += COUNT_INCREMENT
                    logging.getLogger(__name__).info("Skipped %s (%s)", directory, state.status)
                else:
                    pending.append((directory, request))
        return pending

    def _prepare_request(self, directory: Path, problem: Problem, provider_model: str) -> RunRequest:
        """Build the selected input before checking for an existing result."""
        return RunRequest(self._config.experiment, provider_model, problem.statement, self._config.budget_tokens, problem.sketch, None)

    async def _run_attempt(self, directory: Path, request: RunRequest, sandbox_factory: Callable[[], Sandbox]) -> None:
        """Execute one prepared attempt; subclasses may prepare experiment dependencies first."""
        await sandbox_factory().run(directory, request.to_json(), RunLock)
