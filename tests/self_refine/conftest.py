"""Refinement configuration, recovery doubles, and production persistence driver fixtures."""

from asyncio import Event
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from harness.self_refine.constants import (
    CRITIQUE_PROMPT_FILE,
    GAPS_FOUND_VERDICT,
    NO_GAPS_VERDICT,
    REVISE_PROMPT_FILE,
)
from harness.self_refine.models import RefinementConfig, RefinementState
from harness.self_refine.recovery import Recovery
from harness.session.connection_manager import ConnectionManager
from harness.session.models import SessionEvent, SessionState
from harness.session.session_manager import SessionManager
from harness.utils.constants import (
    DEFAULT_MIN_NO_GAP_CRITIQUES,
    DEFAULT_MIN_ROUNDS,
    ZERO_TOKENS,
)
from harness.utils.prompt_loader import load_prompt

from tests.constants import (
    BUDGET_TOKENS,
    PROBLEM,
    SDK_SESSION_ID,
)
from tests.support.async_test_agent_session import AsyncTestAgentSession
from tests.support.persistence_harness import PersistenceHarness
from tests.support.persistent_sdk_transport import PersistentSdkTransport
from tests.support.sdk_harness import SdkHarness


@pytest.fixture
def revise_prompt() -> str:
    """Read the shared revision instruction for phase-order assertions."""
    return load_prompt(REVISE_PROMPT_FILE, {})


@pytest.fixture
def refinement_config() -> RefinementConfig:
    """Use the agreed convergence settings unless a test explicitly overrides them."""
    return RefinementConfig(DEFAULT_MIN_ROUNDS, DEFAULT_MIN_NO_GAP_CRITIQUES)


@pytest.fixture
def restored_state() -> RefinementState:
    """Represent checkpoint restoration with an object distinct from mutable live progress."""
    return RefinementState(PROBLEM, BUDGET_TOKENS, None)


@pytest.fixture
def session_store(state: RefinementState, restored_state: RefinementState) -> MagicMock:
    """Inject storage ownership and checkpoint replacement into focused recovery tests."""
    store = create_autospec(SessionManager, instance=True)
    store.open.return_value = state
    store.session_state = SessionState(state, None)
    store.resume_target = None

    def restore() -> RefinementState:
        """Replace metadata just as a real archive restore does."""
        store.session_state = SessionState(restored_state, SDK_SESSION_ID)
        store.resume_target = SDK_SESSION_ID
        return restored_state

    store.restore.side_effect = restore
    return store


@pytest.fixture
def mock_connections() -> MagicMock:
    """Provide observable connection operations without a real SDK lifecycle."""
    connections = create_autospec(ConnectionManager, instance=True)
    connections.connect.return_value = SDK_SESSION_ID
    return connections


@pytest.fixture
def retry_wait() -> AsyncMock:
    """Record retry delays without sleeping."""
    return AsyncMock()


@pytest.fixture
def recovery(session_store: MagicMock, mock_connections: MagicMock, retry_wait: AsyncMock) -> Recovery:
    """Compose the real retry coordinator with independently controllable dependencies."""
    return Recovery(session_store, mock_connections, retry_wait)


@pytest.fixture
async def persistence(
    sdk_options: ClaudeAgentOptions,
    seed_directory: Path,
    state: RefinementState,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[PersistenceHarness]:
    """Own native files and locks; tests opt into retries when exercising live recovery."""
    monkeypatch.setattr("harness.self_refine.recovery.RECOVERY_MAX_RETRIES", ZERO_TOKENS)
    harness = PersistenceHarness(SdkHarness(sdk_options, PersistentSdkTransport), seed_directory, state)
    try:
        yield harness
    finally:
        await harness.cleanup()


@pytest.fixture
async def make_async_agent() -> AsyncIterator[Callable[[list[SessionEvent]], AsyncTestAgentSession]]:
    """Retain and close fault-injection streams even when their test fails."""
    sessions: list[AsyncTestAgentSession] = []

    def create(events: list[SessionEvent]) -> AsyncTestAgentSession:
        """Create one controllable stream and register its lifetime with the fixture."""
        agent = AsyncTestAgentSession([events])
        sessions.append(agent)
        return agent

    try:
        yield create
    finally:
        for agent in sessions:
            await agent.close_streams()


@pytest.fixture
def warning_started() -> Event:
    """Expose the point where a warning operation blocks."""
    return Event()


@pytest.fixture
def blocked_warning(warning_started: Event) -> AsyncMock:
    """Hold warning submission so tests can cancel or attempt a concurrent run."""

    async def block(prompt: str) -> None:
        """Signal entry and wait until the owning task is cancelled."""
        warning_started.set()
        await Event().wait()

    return AsyncMock(side_effect=block)


@pytest.fixture
def critique_prompt() -> str:
    """Render the shared critique with the verdicts recognized by the harness."""
    return load_prompt(CRITIQUE_PROMPT_FILE, {
        "no_gaps_verdict": NO_GAPS_VERDICT, "gaps_found_verdict": GAPS_FOUND_VERDICT,
    })
