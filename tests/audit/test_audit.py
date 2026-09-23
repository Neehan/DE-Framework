"""Validate real SDK structured verdicts, fresh-stage recovery, and cleanup before publication."""

import json
from asyncio import CancelledError, Queue, create_task, timeout
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from audit.base_audit import BaseAudit
from audit.constants import RESULT_FILENAME, STEP_RECOGNITION
from audit.correctness_audit import CorrectnessAudit
from audit.models import AuditRequest
from audit.step_recognition_audit import StepRecognitionAudit
from claude_agent_sdk import ClaudeAgentOptions
from harness.session.connection_manager import ConnectionManager
from harness.session.constants import RUNTIME_DIRECTORY, WORKSPACE_DIRECTORY
from harness.session.rate_limit_error import RateLimitError
from harness.session.spend_limit_error import SpendLimitError
from harness.utils.constants import DEFAULT_AUDIT_MAX_TURNS
from jsonschema import ValidationError

from tests.audit.constants import CORRECTNESS_RESULT, STEP_RESULT, STEPS
from tests.audit.helpers import send_verdict
from tests.constants import ASYNC_TEST_TIMEOUT_SECONDS, ONE_ROUND, SDK_MESSAGE_ID
from tests.support.sdk_harness import SdkHarness
from tests.support.sdk_transport import SdkTransport


@pytest.mark.parametrize("audit_type", [CorrectnessAudit, StepRecognitionAudit])
async def test_structured_verdict_replaces_uncommitted_worker_output(
    audit_type: type[BaseAudit], audit_request: AuditRequest, tmp_path: Path,
    connection_factory: Callable[[ClaudeAgentOptions], ConnectionManager], audit_connections: Queue[SdkHarness[SdkTransport]],
) -> None:
    """Only host-published audit.json marks completion; abandoned worker output must be judged again."""
    request = replace(audit_request, kind=audit_type.name, steps=STEPS if audit_type.name == STEP_RECOGNITION else None)
    expected = STEP_RESULT if request.steps else CORRECTNESS_RESULT
    audit = audit_type(connection_factory, AsyncMock())
    (tmp_path / RESULT_FILENAME).write_text(json.dumps({"abandoned": "previous judge output"}))
    stale = tmp_path / RUNTIME_DIRECTORY / WORKSPACE_DIRECTORY / "abandoned-judgment"
    stale.parent.mkdir(parents=True)
    stale.write_text("discard partial audit scratch from a killed worker")
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(audit.run(request, tmp_path))
        harness = await audit_connections.get()
        transport = harness.transports[-ONE_ROUND]
        prompt = await transport.prompts.get()
        assert not stale.exists()
        options = harness.clients[-ONE_ROUND].options
        assert options.max_turns == DEFAULT_AUDIT_MAX_TURNS and options.task_budget is None
        assert options.output_format == {"type": "json_schema", "schema": audit_type.schema}
        assert (STEPS[0] in prompt["message"]["content"]) == bool(request.steps)
        send_verdict(transport, expected)
        await task
    assert audit_connections.empty() and not transport.ready
    assert json.loads((tmp_path / RESULT_FILENAME).read_text()) == expected
    assert list(tmp_path.iterdir()) == [tmp_path / RESULT_FILENAME]


async def test_stream_disconnect_restarts_only_this_audit_with_fresh_session(
    audit_request: AuditRequest, tmp_path: Path, connection_factory: Callable[[ClaudeAgentOptions], ConnectionManager],
    audit_connections: Queue[SdkHarness[SdkTransport]],
) -> None:
    """Interrupted judgment text is discarded and the original prompt is retried in a new session."""
    wait = AsyncMock()
    audit = CorrectnessAudit(connection_factory, wait)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(audit.run(audit_request, tmp_path))
        first = await audit_connections.get()
        transport = first.transports[-ONE_ROUND]
        prompt = await transport.prompts.get()
        transport.incoming.put_nowait(None)
        second = await audit_connections.get()
        resumed = second.transports[-ONE_ROUND]
        retry = await resumed.prompts.get()
        assert retry["message"] == prompt["message"]
        assert first.clients[-ONE_ROUND].options.session_id != second.clients[-ONE_ROUND].options.session_id
        assert second.clients[-ONE_ROUND].options.resume is None and not transport.ready
        send_verdict(resumed, CORRECTNESS_RESULT)
        await task
    wait.assert_awaited_once()
    assert list(tmp_path.iterdir()) == [tmp_path / RESULT_FILENAME]


@pytest.mark.parametrize(("diagnostic", "error_type"), [("API Error: 429 rate limit", RateLimitError), ("API Error: spend limit reached", SpendLimitError)])
async def test_credential_failure_propagates_without_local_retry(
    diagnostic: str, error_type: type[Exception], audit_request: AuditRequest, tmp_path: Path,
    connection_factory: Callable[[ClaudeAgentOptions], ConnectionManager], audit_connections: Queue[SdkHarness[SdkTransport]],
) -> None:
    """Credential rotation belongs to the host; a failed worker leaves no completed result or scratch."""
    wait = AsyncMock()
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(CorrectnessAudit(connection_factory, wait).run(audit_request, tmp_path))
        harness = await audit_connections.get()
        transport = harness.transports[-ONE_ROUND]
        await transport.prompts.get()
        transport.assistant(SDK_MESSAGE_ID, diagnostic, ONE_ROUND)
        with pytest.raises(error_type):
            await task
    assert not transport.ready and not list(tmp_path.iterdir())
    wait.assert_not_awaited()


@pytest.mark.parametrize("cancel", [False, True])
async def test_invalid_verdict_or_cancellation_never_marks_audit_complete(
    cancel: bool, audit_request: AuditRequest, tmp_path: Path, connection_factory: Callable[[ClaudeAgentOptions], ConnectionManager],
    audit_connections: Queue[SdkHarness[SdkTransport]],
) -> None:
    """Cleanup runs before any publication, including malformed schema output and cancellation."""
    wait = AsyncMock()
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(CorrectnessAudit(connection_factory, wait).run(audit_request, tmp_path))
        harness = await audit_connections.get()
        transport = harness.transports[-ONE_ROUND]
        await transport.prompts.get()
        if cancel:
            task.cancel()
        else:
            send_verdict(transport, STEP_RESULT)
        with pytest.raises(CancelledError if cancel else ValidationError):
            await task
    assert not transport.ready and not list(tmp_path.iterdir())
    wait.assert_not_awaited()
