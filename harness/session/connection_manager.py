"""SDK connection ownership and buffered transport for one active conversation."""

import logging
import os
from asyncio import CancelledError, Queue, Task, create_task
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, Message

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
    LOCAL_TOOLS,
    MAX_API_RETRIES_ENV,
    MAX_OUTPUT_TOKENS_ENV,
    REPLAY_USER_MESSAGES,
    SDK_DIRECTORY,
    SESSION_NAME_ARG,
    SETTING_SOURCES_ARG,
    SPEND_LIMIT_MESSAGE,
    STREAM_IDLE_TIMEOUT_ENV,
    STREAM_WATCHDOG_ENV,
    TOOL_POLICY_ARGS,
    WORKSPACE_DIRECTORY,
)
from harness.session.errors import is_spend_limit_error, normalize_connection_error
from harness.session.models import ConnectionStatus
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


class ConnectionManager:
    """Own connect, close, query, receive, and interrupt for a configured SDK client.

    Inject provider options and the SDK client factory; connect receives the remaining allowance or None for unbudgeted tasks. Meta budget hints are omitted. close is idempotent after completion; close_after_failure preserves an existing error. No overrides are required.
    """

    def __init__(
        self, options: ClaudeAgentOptions,
        client_factory: Callable[[ClaudeAgentOptions], ClaudeSDKClient],
    ) -> None:
        """Accept caller options while reserving mandatory transport and tool settings."""
        if TOOL_POLICY_ARGS.intersection(options.extra_args):
            raise ValueError("configure tool denials through disallowed_tools, not extra_args")
        if options.task_budget is not None:
            raise ValueError("provide remaining_tokens to connect, not task_budget in SDK options")
        self._options = options
        self._client_factory = client_factory
        self._client: ClaudeSDKClient | None = None
        self._status = ConnectionStatus.DISCONNECTED
        self._reader: Task[None] | None = None
        self._messages: Queue[Message | Exception | None] = Queue()
        self._spend_limit_error: SpendLimitError | None = None

    async def connect(self, runtime: Path, resume_target: str | None, fork_session: bool, remaining_tokens: int | None) -> str:
        """Start, resume, or fork a native conversation with the current remaining output allowance."""
        if self._status != ConnectionStatus.DISCONNECTED:
            raise RuntimeError("connection manager is already active")
        if fork_session and resume_target is None:
            raise ValueError("forking requires a source session ID")
        if remaining_tokens is not None and remaining_tokens <= ZERO_TOKENS:
            raise ValueError("remaining_tokens must be positive")
        native_id = str(uuid4()) if resume_target is None or fork_session else Path(resume_target).stem
        options = self._get_options(runtime, native_id, resume_target, fork_session, remaining_tokens)
        self._status = ConnectionStatus.CONNECTING
        self._messages = Queue()
        self._spend_limit_error = None
        try:
            self._client = self._client_factory(options)
            await self._call_sdk(self._client.connect())
            self._reader = create_task(self._read_messages())
            self._status = ConnectionStatus.CONNECTED
            return native_id
        except BaseException as error:
            await self.close_after_failure(error)
            raise

    async def close(self) -> None:
        """Disconnect once, draining SDK output until shutdown before stopping the reader."""
        if self._status == ConnectionStatus.DISCONNECTED:
            return
        if self._status == ConnectionStatus.CLOSING:
            raise RuntimeError("connection manager is already closing")
        self._status = ConnectionStatus.CLOSING
        try:
            if self._client is not None:
                await self._call_sdk(self._client.disconnect())
        finally:
            try:
                if self._reader is not None:
                    self._reader.cancel()
                    with suppress(CancelledError):
                        await self._reader
            finally:
                self._reader = None
                self._client = None
                self._status = ConnectionStatus.DISCONNECTED

    async def close_after_failure(self, error: BaseException) -> None:
        """Close a failed connection while retaining the primary error and shutdown diagnostics."""
        try:
            await self.close()
        except Exception as shutdown_error:
            error.add_note(f"Connection shutdown also failed: {shutdown_error}")
            logging.getLogger(__name__).warning(
                "Connection shutdown failed while handling %s: %s", type(error).__name__, shutdown_error,
            )

    async def query(self, message: AsyncGenerator[dict[str, Any], None]) -> None:
        """Submit encoded conversation input through the active SDK connection."""
        await self._call_sdk(self._require_client().query(message))

    async def interrupt(self) -> None:
        """Request termination while the background reader drains control acknowledgements."""
        await self._call_sdk(self._require_client().interrupt())

    async def receive(self) -> Message:
        """Read buffered SDK output or raise the original stream failure."""
        self._require_client()
        message = await self._messages.get()
        match message:
            case None:
                raise ConnectionError("SDK stream ended before the phase completed")
            case Exception():
                raise message
            case _:
                return message

    async def _read_messages(self) -> None:
        """Keep SDK control replies flowing independently of the phase consumer."""
        assert self._client is not None
        try:
            stream = cast(AsyncGenerator[Message, None], self._client.receive_messages())
            async with aclosing(stream) as messages:
                async for message in messages:
                    self._messages.put_nowait(message)
        except Exception as error:
            self._messages.put_nowait(self._spend_limit_error or normalize_connection_error(error))
        finally:
            self._messages.put_nowait(None)

    async def _call_sdk(self, operation: Awaitable[None]) -> None:
        """Normalize SDK errors once for startup, writes, interrupts, and shutdown."""
        try:
            await operation
        except Exception as error:
            raise self._spend_limit_error or normalize_connection_error(error)

    def _read_stderr(self, line: str) -> None:
        """Capture startup-only spend failures without retaining stderr, and preserve any injected observer."""
        if self._spend_limit_error is None and is_spend_limit_error(line):
            self._spend_limit_error = SpendLimitError(SPEND_LIMIT_MESSAGE)
            self._messages.put_nowait(self._spend_limit_error)
        if self._options.stderr is not None:
            self._options.stderr(line)

    def _require_client(self) -> ClaudeSDKClient:
        """Reject transport operations outside a connected lifecycle."""
        if self._status != ConnectionStatus.CONNECTED or self._client is None:
            raise RuntimeError("agent session is not connected")
        return self._client

    def _get_options(
        self, runtime: Path, native_id: str, resume_id: str | None, fork_session: bool, remaining_tokens: int | None,
    ) -> ClaudeAgentOptions:
        """Apply storage, uniform limits, and mandatory tool denials on every connection."""
        return replace(
            self._options, cwd=runtime / WORKSPACE_DIRECTORY,
            stderr=self._read_stderr,
            cli_path=self._options.cli_path or os.environ.get(CLI_PATH_ENV),
            include_partial_messages=True, effort=REASONING_EFFORT,
            max_turns=min(self._options.max_turns, MAX_TURNS_PER_PHASE) if self._options.max_turns is not None else MAX_TURNS_PER_PHASE,
            task_budget=(None if remaining_tokens is None or (self._options.model or "").startswith(MUSE_PREFIX)
                         else {"total": max(PROVIDER_MIN_TASK_BUDGET_TOKENS, remaining_tokens)}),
            tools=list(LOCAL_TOOLS),
            system_prompt=(self._options.system_prompt if self._options.system_prompt is not None
                           else load_prompt(DEFAULT_SYSTEM_PROMPT_FILE, {})),
            allowed_tools=[tool for tool in self._options.allowed_tools if tool not in DISALLOWED_TOOLS],
            disallowed_tools=sorted(set(self._options.disallowed_tools) | DISALLOWED_TOOLS | BASH_DENIALS),
            setting_sources=[],
            env={
                **self._options.env, CLAUDE_CONFIG_DIR_ENV: str(runtime / SDK_DIRECTORY),
                MAX_OUTPUT_TOKENS_ENV: str(MAX_OUTPUT_TOKENS_PER_RESPONSE),
                API_TIMEOUT_ENV: str(PROVIDER_TIMEOUT_SECONDS * MILLISECONDS_PER_SECOND),
                STREAM_IDLE_TIMEOUT_ENV: str(PROVIDER_TIMEOUT_SECONDS * MILLISECONDS_PER_SECOND),
                STREAM_WATCHDOG_ENV: ENV_ENABLED,
                DISABLE_NONSTREAMING_FALLBACK_ENV: ENV_ENABLED,
                MAX_API_RETRIES_ENV: str(SDK_MAX_API_RETRIES),
            },
            extra_args={
                **self._options.extra_args, AUTO_COMPACT_ARG: str(AUTO_COMPACT_WINDOW),
                SETTING_SOURCES_ARG: "",
                REPLAY_USER_MESSAGES: None,
                SESSION_NAME_ARG: native_id,
            },
            session_id=native_id if resume_id is None or fork_session else None,
            resume=resume_id, continue_conversation=False, fork_session=fork_session,
        )
