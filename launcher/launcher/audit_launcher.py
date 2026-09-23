"""Audit completed solver checkpoints in a fixed, independently resumable sequence."""

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from pathlib import Path

from audit.audit_compiler import AuditCompiler
from audit.audit_runner import AuditRunner
from audit.models import AuditReference, SeedAudit
from harness.sandbox.sandbox import Sandbox
from harness.self_refine.models import RunStatus
from harness.session.constants import SESSION_FILENAME
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import COUNT_INCREMENT

from launcher.launcher.base import BaseLauncher
from launcher.models import AuditConfig, BatchResult, Problem


class AuditLauncher(BaseLauncher[SeedAudit]):
    """Run correctness then step recognition without sharing their containers or judgments.

    Inject validated host-only references and the existing sandbox environment; no overrides required.
    """

    def __init__(self, config: AuditConfig, environment: Callable[[], AbstractAsyncContextManager[Callable[[], Sandbox]]],
                 results_directory: Path, references: dict[str, AuditReference]) -> None:
        """Keep selected references on the host until projecting each stage's request."""
        super().__init__(config, environment, results_directory)
        self._runner = AuditRunner(config.audit_model, references)

    async def run(self, problems: list[Problem], provider_model: str) -> BatchResult:
        """Finish the batch, then compile all saved completed audits, including previously skipped work."""
        result = await super().run(problems, provider_model)
        AuditCompiler(self._experiment_directory).compile()
        return result

    async def _prepare_attempts(self, problems: list[Problem], provider_model: str, result: BatchResult) -> list[tuple[Path, SeedAudit]]:
        """Snapshot only completed solver states and skip already validated audit pairs."""
        pending = []
        for problem, directory in self._iter_attempts(problems):
            with self._report_attempt_errors(directory, result):
                if not (directory / SESSION_FILENAME).is_file():
                    raise FileNotFoundError(f"solver checkpoint is missing: {directory}")
                with RunLock(directory):
                    request = self._prepare_request(directory, problem, provider_model)
                if request is None:
                    result.skipped += COUNT_INCREMENT
                    logging.getLogger(__name__).info("Skipped %s (solver incomplete, empty solution, or audits completed)", directory)
                else:
                    pending.append((directory, request))
        return pending

    def _prepare_request(self, source: Path, problem: Problem, model: str) -> SeedAudit | None:
        """Read authoritative budget solutions and prepare missing checkpoint judgments."""
        state = SessionManager(source, nullcontext).read_checkpoint().refinement
        if state.problem != problem.statement:
            raise ValueError("checkpoint problem does not match the local dataset")
        if state.status not in (RunStatus.FINISHED, RunStatus.EXHAUSTED):
            return None
        if len(state.solution_checkpoints) != state.checkpoint_count:
            raise ValueError("completed solver is missing solution checkpoints")
        if not self._runner.prepare(source, state.solution_checkpoints):
            return None
        return SeedAudit(model, problem.problem_id, problem.statement, state.solution_checkpoints)

    async def _run_attempt(self, directory: Path, request: SeedAudit, sandbox_factory: Callable[[], Sandbox]) -> None:
        """Own the seed while the audit runner publishes each checkpoint's missing stages."""
        with RunLock(directory):
            await self._runner.run(directory, request, sandbox_factory)
