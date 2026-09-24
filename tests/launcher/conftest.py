"""Local data and injected sandbox fixtures for launcher behavior."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from harness.proxy.constants import API_KEY_ENV
from harness.proxy.models import ProviderConfig
from launcher.cli import parse_arguments
from launcher.dataset import Dataset
from launcher.models import LaunchConfig, Problem, ProviderRoute

from tests.launcher.constants import (
    CLI_ARGUMENTS,
    PROBLEM_ROWS,
    TEST_ROUTE_KEY,
    TEST_ROUTE_URL,
)


@pytest.fixture
def launch_config() -> LaunchConfig:
    """Use the real CLI defaults and an explicit seed array."""
    return parse_arguments(CLI_ARGUMENTS)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Keep forbidden references in a host-only JSONL fixture."""
    (tmp_path / "aobench.jsonl").write_text("\n".join(json.dumps(row) for row in PROBLEM_ROWS))
    return Dataset(tmp_path)


@pytest.fixture
def problems(dataset: Dataset) -> list[Problem]:
    """Select ordinary problem inputs through the production local loader."""
    return dataset.load("aobench", None, None, None)


@pytest.fixture
def sandbox_environment() -> Mock:
    """Fail if stopped work unnecessarily prepares Docker."""
    return Mock(side_effect=AssertionError("Docker must not be prepared"))


@pytest.fixture
def provider() -> ProviderRoute:
    """Use a validated host-only provider without contacting it."""
    return ProviderRoute("claude-test", (ProviderConfig(TEST_ROUTE_URL, API_KEY_ENV, TEST_ROUTE_KEY),), True)
