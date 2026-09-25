"""Audit protocol fixtures reuse the existing real-SDK transport harness."""

from asyncio import Queue
from collections.abc import Callable
from pathlib import Path

import pytest
from audit.constants import CORRECTNESS
from audit.models import AuditRequest
from claude_agent_sdk import ClaudeAgentOptions
from harness.session.connection_manager import ConnectionManager

from tests.audit.constants import (
    AUDIT_MODEL,
    COMPILER_EXPERIMENT,
    REFERENCE,
    SUBMISSION,
)
from tests.constants import PROBLEM
from tests.support.sdk_harness import SdkHarness
from tests.support.sdk_transport import SdkTransport


@pytest.fixture
def experiment_directory(tmp_path: Path) -> Path:
    """Isolate a canonical experiment tree without Docker or datasets."""
    return tmp_path / COMPILER_EXPERIMENT


@pytest.fixture
def audit_request() -> AuditRequest:
    """Supply one independent correctness task without outline or prior judgments."""
    return AuditRequest(CORRECTNESS, AUDIT_MODEL, PROBLEM, REFERENCE, SUBMISSION, None)


@pytest.fixture
def audit_connections() -> Queue[SdkHarness[SdkTransport]]:
    """Expose each new SDK lifetime to deterministic retry tests."""
    return Queue()


@pytest.fixture
def connection_factory(audit_connections: Queue[SdkHarness[SdkTransport]]) -> Callable[[ClaudeAgentOptions], ConnectionManager]:
    """Capture production connection options while replacing only the external CLI transport."""
    def create(options: ClaudeAgentOptions) -> ConnectionManager:
        """Build a fresh real SDK client and transport for each judging attempt."""
        harness = SdkHarness(options, lambda options: SdkTransport())
        audit_connections.put_nowait(harness)
        return harness.connection
    return create
