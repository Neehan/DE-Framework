"""Opt in with HARNESS_TEST_IMAGE=<built-image> pytest tests/integration; no image builds or paid calls."""

import os
import time
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from harness.utils.constants import TEXT_ENCODING

from tests.integration.constants import (
    ABSOLUTE_ESCAPE_LINK,
    ENDPOINT_POLL_SECONDS,
    ENDPOINT_START_TIMEOUT_SECONDS,
    INTEGRATION_IMAGE_ENV,
    LOCAL_DATASET,
    OWN_RUN_CONTENT,
    OWN_RUN_FILE,
    PROVIDER_SERVER_CODE,
    RELATIVE_ESCAPE_LINK,
    SIBLING_SOLUTION,
)
from tests.integration.docker import docker
from tests.integration.provider_server import serve_provider
from tests.integration.tool_provider import ToolProvider


@pytest.fixture
def docker_image() -> str:
    """Skip by default; an explicitly selected image or unavailable daemon fails visibly."""
    image = os.environ.get(INTEGRATION_IMAGE_ENV)
    if image is None:
        pytest.skip(f"set {INTEGRATION_IMAGE_ENV} to a built harness image to run Docker checks")
    docker(["image", "inspect", image])
    return image


@pytest.fixture
def run_directory(tmp_path: Path) -> Path:
    """Create owned scratch and forbidden sibling/data sentinels without mounting their shared parent."""
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / OWN_RUN_FILE).write_text(OWN_RUN_CONTENT, encoding=TEXT_ENCODING)
    for relative in (SIBLING_SOLUTION, LOCAL_DATASET):
        path = tmp_path / relative
        path.parent.mkdir()
        path.write_text("forbidden reference sentinel", encoding=TEXT_ENCODING)
    (directory / ABSOLUTE_ESCAPE_LINK).symlink_to(tmp_path / SIBLING_SOLUTION)
    (directory / RELATIVE_ESCAPE_LINK).symlink_to(Path("..") / SIBLING_SOLUTION)
    return directory


@pytest.fixture
def local_provider() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Serve deterministic inference on the host, reachable only through the run's proxy from Docker."""
    with serve_provider(ToolProvider) as provider:
        yield provider


@pytest.fixture
def provider_network(docker_image: str) -> Iterator[str]:
    """Own one fresh network and two local endpoints; remove only resources created here."""
    network = f"harness-test-{uuid4().hex}"
    with ExitStack() as cleanup:
        docker(["network", "create", network])
        cleanup.callback(docker, ["network", "rm", network])
        for alias in ("provider", "forbidden-peer"):
            container = docker([
                "run", "-d", "--name", f"{network}-{alias}", "--network", network, "--network-alias", alias,
                "--cap-drop", "ALL", "--read-only", "--entrypoint", "python", docker_image,
                "-u", "-c", PROVIDER_SERVER_CODE,
            ])
            cleanup.callback(docker, ["rm", "--force", container])
            wait_for_endpoint(container)
        yield network


def wait_for_endpoint(container: str) -> None:
    """Wait for the endpoint's readiness message after both listeners have bound."""
    deadline = time.monotonic() + ENDPOINT_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if "ready" in docker(["logs", container]):
            return
        time.sleep(ENDPOINT_POLL_SECONDS)
    raise TimeoutError("local test endpoint did not start")
