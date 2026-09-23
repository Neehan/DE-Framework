"""SDK connection ownership, lifecycle, and transport error classification."""

from asyncio import CancelledError, Event, create_task, timeout
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    CLIConnectionError,
    CLINotFoundError,
    ProcessError,
)
from harness.session.connection_manager import ConnectionManager
from harness.session.constants import (
    API_TIMEOUT_ENV,
    AUTO_COMPACT_ARG,
    BASH_DENIALS,
    CLAUDE_CONFIG_DIR_ENV,
    CLI_PATH_ENV,
    DEFAULT_SYSTEM_PROMPT_FILE,
    DISABLE_NONSTREAMING_FALLBACK_ENV,
    DISALLOWED_TOOLS,
    ENV_ENABLED,
    MAX_API_RETRIES_ENV,
    MAX_OUTPUT_TOKENS_ENV,
    REPLAY_USER_MESSAGES,
    SDK_DIRECTORY,
    SETTING_SOURCES_ARG,
    STREAM_IDLE_TIMEOUT_ENV,
    STREAM_WATCHDOG_ENV,
    TOOL_POLICY_ARGS,
    WORKSPACE_DIRECTORY,
)
from harness.session.models import ConnectionStatus
from harness.session.rate_limit_error import RateLimitError
from harness.session.spend_limit_error import SpendLimitError
from harness.utils.constants import (
    AUTO_COMPACT_WINDOW,
    MAX_OUTPUT_TOKENS_PER_RESPONSE,
    MAX_TURNS_PER_PHASE,
    MILLISECONDS_PER_SECOND,
    MUSE_PREFIX,
    PROVIDER_MIN_TASK_BUDGET_TOKENS,
    PROVIDER_TIMEOUT_SECONDS,
    REASONING_EFFORT,
    SDK_MAX_API_RETRIES,
    ZERO_TOKENS,
)
from harness.utils.prompt_loader import load_prompt

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    BUDGET_TOKENS,
    FATAL_PROCESS_EXIT_CODE,
    KILLED_PROCESS_EXIT_CODE,
    ONE_ROUND,
    PHASE_TOKENS,
    SDK_CLI_PATH,
    SDK_MODEL,
    TRANSIENT_PROVIDER_ERROR,
)
from tests.support.sdk_harness import SdkHarness
from tests.support.sdk_transport import SdkTransport


async def test_web_subagent_and_workflow_tools_stay_denied_after_reconnect(
    sdk_options: ClaudeAgentOptions, seed_directory: Path,
) -> None:
    """Mandatory denials survive caller allowances, project settings, and native resume."""
    sdk_options.allowed_tools = [
        "Read", "WebSearch", "WebFetch", "Agent", "Task", "Workflow", "Skill", "TaskCreate", "CronCreate",
    ]
    sdk_options.disallowed_tools = ["NotebookEdit"]
    sdk_options.setting_sources = ["project"]
    sdk_options.extra_args[SETTING_SOURCES_ARG] = "project"
    sdk = SdkHarness(sdk_options, lambda options: SdkTransport())
    try:
        native_id = await sdk.connection.connect(seed_directory, None, False, BUDGET_TOKENS)
        await sdk.connection.close()
        await sdk.connection.connect(seed_directory, native_id, False, BUDGET_TOKENS)
        for client in sdk.clients:
            assert DISALLOWED_TOOLS <= set(client.options.disallowed_tools)
            assert BASH_DENIALS <= set(client.options.disallowed_tools)
            assert "NotebookEdit" in client.options.disallowed_tools
            assert client.options.allowed_tools == ["Read"]
            assert set(client.options.tools or []) == {"Bash", "Read", "Write", "Edit", "Glob", "Grep"}
            assert client.options.setting_sources == []
            assert client.options.extra_args[SETTING_SOURCES_ARG] == ""
    finally:
        await sdk.connection.close()


@pytest.mark.parametrize("argument", sorted(TOOL_POLICY_ARGS))
def test_extra_arguments_cannot_override_tool_denials(
    sdk_options: ClaudeAgentOptions, mock_client: MagicMock, argument: str,
) -> None:
    """Reject duplicate CLI denial flags that could override mandatory exclusions."""
    sdk_options.extra_args[argument] = ""
    with pytest.raises(ValueError, match="tool denials"):
        ConnectionManager(sdk_options, lambda options: mock_client)


@pytest.mark.parametrize(("model", "budgeted"), ((SDK_MODEL, True), (f"{MUSE_PREFIX}test", True), (SDK_MODEL, False)))
async def test_resume_preserves_identity_provider_options_and_uniform_limits(
    sdk: SdkHarness[SdkTransport], sdk_options: ClaudeAgentOptions, seed_directory: Path,
    monkeypatch: pytest.MonkeyPatch, model: str, budgeted: bool,
) -> None:
    """Refresh supported pacing hints on resume, keep disabled providers disabled, and enforce uniform limits."""
    monkeypatch.setenv(CLI_PATH_ENV, SDK_CLI_PATH)
    sdk_options.model = model
    budget_hint_enabled = budgeted and not model.startswith(MUSE_PREFIX)
    native_id = await sdk.connection.connect(seed_directory, None, False, BUDGET_TOKENS if budgeted else None)
    await sdk.connection.close()
    resumed_id = await sdk.connection.connect(seed_directory, native_id, False, PHASE_TOKENS if budgeted else None)
    await sdk.connection.close()
    assert resumed_id == native_id
    for client in sdk.clients:
        options = client.options
        assert options.cli_path == SDK_CLI_PATH
        assert options.model == model
        assert options.env["PROVIDER_SETTING"] == "preserved"
        assert options.cwd == seed_directory / WORKSPACE_DIRECTORY
        assert options.env[CLAUDE_CONFIG_DIR_ENV] == str(seed_directory / SDK_DIRECTORY)
        assert options.env[MAX_OUTPUT_TOKENS_ENV] == str(MAX_OUTPUT_TOKENS_PER_RESPONSE)
        assert options.extra_args[AUTO_COMPACT_ARG] == str(AUTO_COMPACT_WINDOW)
        assert options.effort == REASONING_EFFORT
        assert options.max_turns == MAX_TURNS_PER_PHASE
        assert options.env[API_TIMEOUT_ENV] == str(PROVIDER_TIMEOUT_SECONDS * MILLISECONDS_PER_SECOND)
        assert options.env[STREAM_IDLE_TIMEOUT_ENV] == options.env[API_TIMEOUT_ENV]
        assert options.env[STREAM_WATCHDOG_ENV] == ENV_ENABLED
        assert options.env[DISABLE_NONSTREAMING_FALLBACK_ENV] == ENV_ENABLED
        assert options.env[MAX_API_RETRIES_ENV] == str(SDK_MAX_API_RETRIES)
        assert options.include_partial_messages
        assert options.extra_args[REPLAY_USER_MESSAGES] is None
        assert options.system_prompt == load_prompt(DEFAULT_SYSTEM_PROMPT_FILE, {})
    assert sdk.clients[ZERO_TOKENS].options.session_id == native_id
    assert sdk.clients[-ONE_ROUND].options.resume == native_id
    assert sdk.clients[-ONE_ROUND].options.session_id is None
    assert sdk.clients[ZERO_TOKENS].options.task_budget == ({"total": BUDGET_TOKENS} if budget_hint_enabled else None)
    assert sdk.clients[-ONE_ROUND].options.task_budget == ({"total": PROVIDER_MIN_TASK_BUDGET_TOKENS} if budget_hint_enabled else None)


def test_sdk_options_cannot_supply_a_second_budget(sdk_options: ClaudeAgentOptions, mock_client: MagicMock) -> None:
    """Reject ambiguous budget configuration instead of silently ignoring its numeric value."""
    sdk_options.task_budget = {"total": BUDGET_TOKENS}
    with pytest.raises(ValueError, match="remaining_tokens"):
        ConnectionManager(sdk_options, lambda options: mock_client)


async def test_close_disconnects_exactly_once(
    seed_directory: Path, mock_client: MagicMock, client_connection: ConnectionManager
) -> None:
    """Repeated cleanup cannot disconnect the same SDK client twice."""
    await client_connection.connect(seed_directory, None, False, BUDGET_TOKENS)
    await client_connection.close()
    await client_connection.close()
    mock_client.disconnect.assert_awaited_once()
    assert client_connection._status == ConnectionStatus.DISCONNECTED
    assert client_connection._reader is None


async def test_cancelled_connect_releases_client(
    seed_directory: Path, mock_client: MagicMock, client_connection: ConnectionManager
) -> None:
    """Cancellation while connecting cleans up once and never leaves a reader task."""
    started = Event()
    release = Event()

    async def connect() -> None:
        """Wait until the test cancels startup."""
        started.set()
        await release.wait()

    mock_client.connect.side_effect = connect
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(client_connection.connect(seed_directory, None, False, BUDGET_TOKENS))
        await started.wait()
        task.cancel()
        with pytest.raises(CancelledError):
            await task
    await client_connection.close()
    mock_client.disconnect.assert_awaited_once()
    assert client_connection._reader is None


@pytest.mark.parametrize("error", (CancelledError(), CLINotFoundError("missing CLI")))
async def test_failed_startup_preserves_original_error_when_shutdown_fails(
    seed_directory: Path, error: BaseException, mock_client: MagicMock, client_connection: ConnectionManager
) -> None:
    """Startup cancellation and permanent failures survive a secondary disconnect error."""
    mock_client.connect.side_effect = error
    mock_client.disconnect.side_effect = CLIConnectionError("shutdown disconnected")
    with pytest.raises(type(error)) as raised:
        await client_connection.connect(seed_directory, None, False, BUDGET_TOKENS)
    assert raised.value is error
    assert "shutdown disconnected" in "\n".join(error.__notes__)
    assert client_connection._status == ConnectionStatus.DISCONNECTED
    assert client_connection._client is None
    assert client_connection._reader is None
    await client_connection.close()
    mock_client.disconnect.assert_awaited_once()


async def test_duplicate_connect_does_not_close_owner(
    sdk: SdkHarness[SdkTransport], seed_directory: Path
) -> None:
    """A second connect fails without replacing the active client or reader."""
    await sdk.connection.connect(seed_directory, None, False, BUDGET_TOKENS)
    with pytest.raises(RuntimeError, match="already active"):
        await sdk.connection.connect(seed_directory, None, False, BUDGET_TOKENS)
    assert len(sdk.clients) == ONE_ROUND
    assert sdk.transports[ZERO_TOKENS].ready


@pytest.mark.parametrize(
    "error, expected",
    [
        (CLINotFoundError("missing CLI"), CLINotFoundError),
        (ProcessError("bad settings", exit_code=FATAL_PROCESS_EXIT_CODE), ProcessError),
        (ProcessError("killed", exit_code=KILLED_PROCESS_EXIT_CODE), ConnectionError),
        (CLIConnectionError("startup disconnected"), ConnectionError),
        (ProcessError(TRANSIENT_PROVIDER_ERROR, exit_code=FATAL_PROCESS_EXIT_CODE), ConnectionError),
        (ProcessError("API Error: 429 rate_limit_error", exit_code=FATAL_PROCESS_EXIT_CODE), RateLimitError),
        (ProcessError("Spend limit reached", exit_code=FATAL_PROCESS_EXIT_CODE), SpendLimitError),
    ],
)
async def test_startup_failures_are_classified_and_cleaned_up(
    seed_directory: Path,
    error: Exception,
    expected: type[Exception],
    mock_client: MagicMock,
    client_connection: ConnectionManager,
) -> None:
    """Missing CLI and configuration errors stay permanent; killed/transient CLI failures retry."""
    mock_client.connect.side_effect = error
    with pytest.raises(expected):
        await client_connection.connect(seed_directory, None, False, BUDGET_TOKENS)
    await client_connection.close()
    mock_client.disconnect.assert_awaited_once()
    assert client_connection._status == ConnectionStatus.DISCONNECTED
    assert client_connection._reader is None


async def test_shutdown_failure_still_stops_reader_and_releases_client(
    sdk: SdkHarness[SdkTransport], seed_directory: Path
) -> None:
    """Failed SDK disconnect cannot leave a background reader or active lifecycle behind."""
    await sdk.connection.connect(seed_directory, None, False, BUDGET_TOKENS)
    client = sdk.clients[ZERO_TOKENS]
    disconnect = client.disconnect

    async def fail_after_disconnect() -> None:
        """Release the SDK transport before simulating its shutdown exception."""
        await disconnect()
        raise CLIConnectionError("shutdown disconnected")

    client.disconnect = AsyncMock(side_effect=fail_after_disconnect)
    with pytest.raises(ConnectionError):
        await sdk.connection.close()
    await sdk.connection.close()
    client.disconnect.assert_awaited_once()
    assert sdk.connection._reader is None
    assert sdk.connection._client is None


@pytest.mark.parametrize("startup", (True, False))
async def test_stderr_only_spend_failure_reaches_host_and_resets_for_next_connection(
    sdk_options: ClaudeAgentOptions, seed_directory: Path, mock_client: MagicMock, startup: bool,
) -> None:
    """A generic startup exit or live stderr-only failure selects permanent rotation without retaining raw stderr."""
    observer = Mock()
    sdk_options.stderr = observer
    factory = Mock(return_value=mock_client)
    connection = ConnectionManager(sdk_options, factory)

    def report_spend_limit() -> None:
        """Report the provider's only useful diagnostic through the actual configured stderr callback."""
        options = factory.call_args.args[0]
        options.stderr("Provider spend limit reached")

    async def fail_connect() -> None:
        """Emit a diagnostic before the SDK raises its otherwise uninformative process failure."""
        report_spend_limit()
        raise ProcessError("CLI exited", exit_code=FATAL_PROCESS_EXIT_CODE)

    try:
        if startup:
            mock_client.connect.side_effect = fail_connect
            with pytest.raises(SpendLimitError):
                await connection.connect(seed_directory, None, False, BUDGET_TOKENS)
        else:
            await connection.connect(seed_directory, None, False, BUDGET_TOKENS)
            report_spend_limit()
            with pytest.raises(SpendLimitError):
                await connection.receive()
        observer.assert_called_once_with("Provider spend limit reached")
        await connection.close()
        mock_client.connect.side_effect = None
        await connection.connect(seed_directory, None, False, BUDGET_TOKENS)
        assert connection._spend_limit_error is None
    finally:
        await connection.close()
