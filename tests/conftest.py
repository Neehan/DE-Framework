"""Fixtures shared by session and refinement component tests."""

from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from harness.self_refine.models import RefinementState

from tests.constants import (
    BUDGET_TOKENS,
    PROBLEM,
    SDK_MODEL,
    TEST_ATTEMPT_DIRECTORY,
)


@pytest.fixture
def state() -> RefinementState:
    """Create independent initial progress for each scenario."""
    return RefinementState(PROBLEM, BUDGET_TOKENS, None)


@pytest.fixture
def seed_directory(tmp_path: Path) -> Path:
    """Keep every seed under pytest's per-test temporary directory."""
    return tmp_path / TEST_ATTEMPT_DIRECTORY


@pytest.fixture
def sdk_options() -> ClaudeAgentOptions:
    """Supply caller settings; ConnectionManager owns mandatory SDK options."""
    return ClaudeAgentOptions(
        model=SDK_MODEL,
        env={"PROVIDER_SETTING": "preserved"},
    )
