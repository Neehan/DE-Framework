"""Check saved attempts, then schedule pending work with bounded Docker concurrency."""

import logging
from asyncio import TaskGroup
from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager, contextmanager
from itertools import product
from pathlib import Path
from subprocess import CalledProcessError
from typing import Generic, TypeVar

from harness.sandbox.sandbox import Sandbox
from harness.session.errors import AttemptAlreadyRunning
from harness.utils.constants import COUNT_INCREMENT, TEXT_ENCODING

from launcher.models import BatchResult, LaunchConfig, Problem

Request = TypeVar("Request")


class BaseLauncher(Generic[Request]):
    """Share scheduling, lazy infrastructure, and per-attempt failure reporting.

    Subclasses implement _prepare_attempts and _run_attempt; _report_attempt_errors and _iter_attempts are shared helpers.
    """

    def __init__(
        self, config: LaunchConfig,
        environment: Callable[[], AbstractAsyncContextManager[Callable[[], Sandbox]]], results_directory: Path,
    ) -> None:
        """Accept lazy environment ownership so completed batches need no Docker setup."""
        self._config = config
        self._environment = environment
        self._results_directory = results_directory

    async def run(self, problems: list[Problem], provider_model: str) -> BatchResult:
        """Prepare checkpoints before Docker; preserve independent failures and propagate cancellation."""
        result = BatchResult()
        pending = await self._prepare_attempts(problems, provider_model, result)
        if pending:
            async with self._environment() as sandbox_factory:
                await self._run_pending_attempts(pending, sandbox_factory, result)
        logging.getLogger(__name__).info("Completed: %d; skipped: %d; failed: %d", result.completed, result.skipped, result.failed)
        return result

    def _iter_attempts(self, problems: list[Problem]) -> Iterator[tuple[Problem, Path]]:
        """Resolve problem/seed selections and result paths identically for both launchers."""
        for problem, seed in product(problems, self._config.seeds):
            yield problem, self._experiment_directory / problem.problem_id / f"seed_{seed}"

    @property
    def _experiment_directory(self) -> Path:
        """Resolve the shared experiment root for attempt storage and audit compilation."""
        return self._results_directory / self._config.dataset / self._config.model_directory / self._config.experiment_directory

    async def _prepare_attempts(self, problems: list[Problem], provider_model: str, result: BatchResult) -> list[tuple[Path, Request]]:
        """Subclasses validate inputs and collect pending attempts before starting Docker."""
        raise NotImplementedError

    async def _run_attempt(self, directory: Path, request: Request, sandbox_factory: Callable[[], Sandbox]) -> None:
        """Subclasses execute one attempt using the shared batch infrastructure."""
        raise NotImplementedError

    async def _run_pending_attempts(
        self, pending: list[tuple[Path, Request]], sandbox_factory: Callable[[], Sandbox], result: BatchResult,
    ) -> None:
        """Admit at most the configured number of attempts without creating a task for every seed."""
        attempts = iter(pending)

        async def consume_attempts() -> None:
            """Take the next attempt after the current container finishes or fails."""
            for directory, request in attempts:
                with self._report_attempt_errors(directory, result):
                    logging.getLogger(__name__).info("Running %s", directory)
                    await self._run_attempt(directory, request, sandbox_factory)
                    result.completed += COUNT_INCREMENT
                    logging.getLogger(__name__).info("Completed %s", directory)

        async with TaskGroup() as group:
            for _ in range(min(self._config.max_concurrency, len(pending))):
                group.create_task(consume_attempts())

    @contextmanager
    def _report_attempt_errors(self, directory: Path, result: BatchResult) -> Iterator[None]:
        """Classify ownership contention and report per-attempt failures consistently in both stages."""
        logger = logging.getLogger(__name__)
        try:
            yield
        except AttemptAlreadyRunning:
            result.skipped += COUNT_INCREMENT
            logger.info("Skipped %s (already running)", directory)
        except CalledProcessError as error:
            result.failed += COUNT_INCREMENT
            logger.error("Failed %s: %s", directory, error.stderr.decode(TEXT_ENCODING))
        except Exception:
            result.failed += COUNT_INCREMENT
            logger.exception("Failed %s", directory)
