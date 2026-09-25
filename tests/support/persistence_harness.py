"""Drive production refinement and recovery through a simulated SDK transport."""

from asyncio import Task, create_task, gather
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4
from zipfile import ZipFile

from harness.self_refine.models import RefinementConfig, RefinementState
from harness.self_refine.recovery import Recovery
from harness.self_refine.self_refine import SelfRefine
from harness.session.constants import (
    CHECKPOINT_FILENAME,
    RUNTIME_DIRECTORY,
    SESSION_FILENAME,
    WORKSPACE_DIRECTORY,
)
from harness.session.models import SessionState
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import TEXT_ENCODING

from tests.constants import ONE_ROUND, PHASE_TOKENS, SDK_INITIAL_TOKENS
from tests.support.persistent_sdk_transport import PersistentSdkTransport
from tests.support.sdk_harness import SdkHarness


class PersistenceHarness:
    """Supply scripted SDK responses and inspect checkpoints; production classes own all coordination."""

    def __init__(
        self, sdk: SdkHarness[PersistentSdkTransport], directory: Path, state: RefinementState
    ) -> None:
        """Keep fixture inputs and track storage owners for failure-safe teardown."""
        self.sdk = sdk
        self.directory = directory
        self.state = state
        self.wait = AsyncMock()
        self._tasks: list[Task[RefinementState]] = []
        self._owners: list[SessionManager] = []
        self.manager = self.new_manager()

    def new_manager(self) -> SessionManager:
        """Register another owner for the same seed so teardown releases every lock."""
        manager = SessionManager(self.directory, RunLock)
        self._owners.append(manager)
        return manager

    async def cleanup(self) -> None:
        """Close SDK activity before releasing storage owners."""
        for task in self._tasks:
            task.cancel()
        await gather(*self._tasks, return_exceptions=True)
        try:
            await self.sdk.connection.close()
        finally:
            for manager in reversed(self._owners):
                manager.close()

    def start(self, manager: SessionManager, state: RefinementState) -> Task[RefinementState]:
        """Run production coordination and register the task for fixture cleanup."""
        runner = SelfRefine(Recovery(manager, self.sdk.connection, self.wait), RefinementConfig(ONE_ROUND, ONE_ROUND))
        task = create_task(runner.run(state))
        self._tasks.append(task)
        return task

    async def wait_for_prompt(self, connection_count: int) -> dict[str, Any]:
        """Wait for a prompt on a specific connection, including reconnects and later rounds."""
        await self.sdk.wait_for_connection(connection_count)
        return await self.sdk.transports[-ONE_ROUND].prompts.get()

    def send_result(self, transport: PersistentSdkTransport, text: str) -> None:
        """Persist and deliver a fresh assistant message and its terminal result."""
        transport.assistant(str(uuid4()), text, SDK_INITIAL_TOKENS)
        transport.result(text, PHASE_TOKENS, ONE_ROUND, False)

    def checkpoint(self) -> SessionState:
        """Read persisted state independently of the manager's mutable state."""
        with ZipFile(self.directory / SESSION_FILENAME) as archive:
            return SessionState.from_json(archive.read(CHECKPOINT_FILENAME).decode(TEXT_ENCODING))

    def workspace(self) -> Path:
        """Locate the disposable files visible to the test model."""
        return self.directory / RUNTIME_DIRECTORY / WORKSPACE_DIRECTORY
