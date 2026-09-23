"""Bounded Docker commands and read-only mounts for opt-in local integration checks."""

import subprocess

from harness.sandbox.constants import DOCKER_EXECUTABLE
from harness.utils.constants import INITIAL_COUNT

from tests.integration.constants import (
    DOCKER_TIMEOUT_SECONDS,
    IMPLEMENTATION_DIRECTORY,
    PYTHONPATH,
)


def docker(arguments: list[str]) -> str:
    """Run a Docker control command and include its diagnostic output on failure."""
    result = subprocess.run(
        [DOCKER_EXECUTABLE, *arguments], capture_output=True, text=True, timeout=DOCKER_TIMEOUT_SECONDS,
    )
    if result.returncode:
        raise RuntimeError(f"Docker {arguments[INITIAL_COUNT]} failed: {result.stderr}\n{result.stdout}")
    return result.stdout.strip()


def checkout_mounts() -> list[str]:
    """Exercise the current code and prompts, while keeping datasets outside every container."""
    return [
        "--mount", f"type=bind,source={IMPLEMENTATION_DIRECTORY / 'harness'},target=/app/harness,readonly",
        "--mount", f"type=bind,source={IMPLEMENTATION_DIRECTORY / 'experiments'},target=/app/experiments,readonly",
        "--mount", f"type=bind,source={IMPLEMENTATION_DIRECTORY / 'launcher'},target=/app/launcher,readonly",
        "--mount", f"type=bind,source={IMPLEMENTATION_DIRECTORY / 'prompts'},target=/app/prompts,readonly",
        "--mount", f"type=bind,source={IMPLEMENTATION_DIRECTORY / 'tests'},target=/test-support/tests,readonly",
        "--env", f"PYTHONPATH={PYTHONPATH}",
    ]
