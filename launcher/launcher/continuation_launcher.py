"""Prepare a shared unaided prefix and atomically fork independent continuation attempts."""

from asyncio import sleep
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

from experiments.constants import (
    CONTINUATION_TOTAL_MULTIPLIER,
    PREFIX_DIRECTORY,
    PREFIX_TOKENS,
    UNAIDED,
)
from harness.sandbox.sandbox import Sandbox
from harness.self_refine.models import RunStatus
from harness.session.constants import SESSION_FILENAME
from harness.session.errors import AttemptAlreadyRunning
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import OUTPUT_TOKENS_PER_BLOCK

from launcher.constants import PREFIX_LOCK_POLL_SECONDS
from launcher.launcher.run_launcher import RunLauncher
from launcher.models import Problem, RunRequest


class ContinuationLauncher(RunLauncher):
    """Reuse scheduling and failure isolation while sharing a locked prefix between late arms.

    Only the host accesses the source archive. Each worker receives its own fork and permitted sketch; no overrides are required.
    """

    def _prepare_request(self, directory: Path, problem: Problem, provider_model: str) -> RunRequest:
        """Use the branch's own saved allowance when checking whether it needs execution."""
        request = super()._prepare_request(directory, problem, provider_model)
        if not (directory / SESSION_FILENAME).exists():
            return request
        return self._restore_branch_budget(directory, request)

    async def _run_attempt(self, directory: Path, request: RunRequest, sandbox_factory: Callable[[], Sandbox]) -> None:
        """Resume an existing branch directly; only new branches wait for and fork the prefix."""
        with RunLock(directory):
            if (directory / SESSION_FILENAME).exists():
                request = self._restore_branch_budget(directory, request)
            else:
                async with self._wait_for_prefix_lock(directory):
                    request = await self._prepare_fork(directory, request, sandbox_factory)
            await sandbox_factory().run(directory, request.to_json(), nullcontext)

    def _restore_branch_budget(self, directory: Path, request: RunRequest) -> RunRequest:
        """Recover the allowance committed at fork creation, including prefixes that finished early."""
        state = SessionManager(directory, nullcontext).read_checkpoint().refinement
        return replace(request, budget_tokens=state.budget_tokens)

    async def _prepare_fork(self, directory: Path, request: RunRequest, sandbox_factory: Callable[[], Sandbox]) -> RunRequest:
        """Finish the exclusively owned prefix once and derive the branch's extra allowance."""
        prefix = self._prefix_request(request)
        source = SessionManager(self._prefix_directory(directory), nullcontext)
        state = source.prepare_attempt(prefix.initial_state)
        if state.status == RunStatus.RUNNING:
            await sandbox_factory().run(self._prefix_directory(directory), prefix.to_json(), nullcontext)
        state = source.open(prefix.initial_state)
        try:
            if state.status == RunStatus.RUNNING:
                raise RuntimeError("prefix worker returned without a stopped checkpoint")
            request = replace(request, budget_tokens=state.output_tokens + OUTPUT_TOKENS_PER_BLOCK)
            source.fork(SessionManager(directory, nullcontext), request.budget_tokens)
            return request
        finally:
            source.close()

    @asynccontextmanager
    async def _wait_for_prefix_lock(self, directory: Path) -> AsyncIterator[None]:
        """Wait for the other branch's shared dependency; cancellation releases branch ownership."""
        lock = RunLock(self._prefix_directory(directory))
        while True:
            try:
                lock.__enter__()
                break
            except AttemptAlreadyRunning:
                await sleep(PREFIX_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            lock.__exit__(None, None, None)

    def _prefix_directory(self, directory: Path) -> Path:
        """Use one host-only prefix namespace for both continuation arms and the same problem/seed."""
        return directory.parent.parent.parent / PREFIX_DIRECTORY / directory.parent.name / directory.name

    def _prefix_request(self, request: RunRequest) -> RunRequest:
        """Advertise four blocks from the start, pause at three, and exclude every sketch."""
        return replace(request, experiment=UNAIDED, budget_tokens=CONTINUATION_TOTAL_MULTIPLIER * OUTPUT_TOKENS_PER_BLOCK,
                       sketch=None, pause_at_tokens=PREFIX_TOKENS)
