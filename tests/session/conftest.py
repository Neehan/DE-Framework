"""SDK protocol fixtures and independent file-only session storage fixtures."""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from harness.self_refine.models import Phase, RefinementState, RunStatus
from harness.session.agent_session import AgentSession
from harness.session.connection_manager import ConnectionManager
from harness.session.constants import (
    SDK_DIRECTORY,
    TRANSCRIPT_EXTENSION,
    WORKSPACE_DIRECTORY,
)
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.session.token_usage import TokenUsage
from harness.utils.constants import TEXT_ENCODING

from tests.constants import (
    BUDGET_TOKENS,
    INITIAL_SOLUTION,
    NATIVE_SESSION_ID,
    ONE_ROUND,
    PHASE_TOKENS,
    SDK_MESSAGE_ID,
    SDK_PROJECT_DIRECTORY,
    WORKSPACE_FILENAME,
)
from tests.support.sdk_harness import SdkHarness
from tests.support.sdk_transport import SdkTransport


@pytest.fixture
def mock_client() -> MagicMock:
    """Provide the SDK interface for lifecycle fault injection."""
    return create_autospec(ClaudeSDKClient, instance=True)


@pytest.fixture
async def client_connection(
    sdk_options: ClaudeAgentOptions,
    mock_client: MagicMock,
) -> AsyncIterator[ConnectionManager]:
    """Own a real connection manager around an injected SDK client double."""
    connection = ConnectionManager(sdk_options, lambda options: mock_client)
    try:
        yield connection
    finally:
        await connection.close()


@pytest.fixture
async def sdk(sdk_options: ClaudeAgentOptions) -> AsyncIterator[SdkHarness[SdkTransport]]:
    """Own SDK shutdown even when a protocol or lifecycle assertion fails."""
    harness = SdkHarness(sdk_options, lambda options: SdkTransport())
    try:
        yield harness
    finally:
        await harness.connection.close()


@pytest.fixture
async def transport(sdk: SdkHarness[SdkTransport], seed_directory: Path) -> SdkTransport:
    """Connect the protocol fixture without starting a CLI or making provider calls."""
    await sdk.connection.connect(seed_directory, None, False, BUDGET_TOKENS)
    return sdk.transports[-ONE_ROUND]


@pytest.fixture
def session(sdk: SdkHarness[SdkTransport], transport: SdkTransport) -> AgentSession:
    """Bind a fresh interpreter to the fixture's connected SDK."""
    return AgentSession(sdk.connection, set())


@pytest.fixture
def usage() -> TokenUsage:
    """Start an isolated token ledger without SDK dependencies."""
    return TokenUsage(set())


@pytest.fixture
def manager(seed_directory: Path) -> Iterator[SessionManager]:
    """Own storage without constructing any SDK client or connection fixture."""
    manager = SessionManager(seed_directory, RunLock)
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture
def saved_session(manager: SessionManager, state: RefinementState) -> SessionManager:
    """Save one completed phase with opaque native transcript bytes and workspace contents."""
    manager.open(state)
    state.phase = Phase.CRITIQUE
    state.output_tokens = PHASE_TOKENS
    state.solution = INITIAL_SOLUTION
    state.pause_at_tokens = PHASE_TOKENS
    state.status = RunStatus.PAUSED
    state.solution_checkpoints = {ONE_ROUND: INITIAL_SOLUTION}
    manager.session_state.session_id = NATIVE_SESSION_ID
    manager.session_state.completed_message_ids.add(SDK_MESSAGE_ID)
    transcript = manager.runtime_directory / SDK_DIRECTORY / SDK_PROJECT_DIRECTORY / f"{NATIVE_SESSION_ID}{TRANSCRIPT_EXTENSION}"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(INITIAL_SOLUTION, encoding=TEXT_ENCODING)
    (manager.runtime_directory / WORKSPACE_DIRECTORY / WORKSPACE_FILENAME).write_text(INITIAL_SOLUTION, encoding=TEXT_ENCODING)
    manager.checkpoint()
    return manager
