"""Temporary host files and injected setup dependencies; no real credentials or Docker operations."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from aiohttp import ClientSession
from codex.gateway import Gateway
from harness.sandbox.docker import Docker
from scripts.setup import Setup

from scripts import setup as setup_module
from tests.audit.constants import STEPS
from tests.codex.constants import GATEWAY_KEY
from tests.launcher.constants import PROBLEM_ROWS


@pytest.fixture
def setup_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every setup-owned host path and preserve unrelated environment entries."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"# User settings\nLITELLM_API_KEY={GATEWAY_KEY}\nANTHROPIC_API_KEY=keep-this-value\n")
    monkeypatch.setattr(setup_module, "DOTENV_FILE", dotenv)
    monkeypatch.setattr(setup_module, "DATASETS_DIRECTORY", tmp_path / "datasets")
    return tmp_path


@pytest.fixture
def dataset_content() -> bytes:
    """Use a real local-dataset row with a valid three-step reference."""
    row = json.loads(json.dumps(PROBLEM_ROWS[0]))
    row["solutions"][0]["steps"] = STEPS
    return (json.dumps(row) + "\n").encode()


@pytest.fixture
def setup_client(dataset_content: bytes) -> MagicMock:
    """Serve synthetic JSONL downloads through the same HTTP interface as production."""
    client = create_autospec(ClientSession, instance=True)
    response = client.get.return_value.__aenter__.return_value
    response.raise_for_status = MagicMock()
    response.read = AsyncMock(return_value=dataset_content)
    return client


@pytest.fixture
def setup_docker() -> MagicMock:
    """Record Docker health and image builds without contacting an engine."""
    docker = create_autospec(Docker, instance=True)
    docker.run.return_value = b""
    return docker


@pytest.fixture
def setup_gateway() -> MagicMock:
    """Keep account import, build, and readiness calls injectable."""
    return create_autospec(Gateway, instance=True)


@pytest.fixture
def repository_setup(setup_directory: Path, setup_docker: MagicMock, setup_client: MagicMock, setup_gateway: MagicMock) -> Setup:
    """Compose the production orchestrator with isolated files and external services."""
    return Setup(setup_docker, setup_client, setup_gateway)
