"""Install container egress rules, then launch the selected command without privileges."""

import os
import socket
import subprocess
import sys

from harness.proxy.models import ProviderEndpoint
from harness.sandbox.constants import (
    COMMAND_ARGUMENT_OFFSET,
    CONTROL_TIMEOUT_SECONDS,
    HOSTS_FILE,
    PRIVILEGE_EXECUTABLE,
    PROVIDER_BASE_URL_ENV,
    ROOT_USER_ID,
    RUN_GID_ENV,
    RUN_UID_ENV,
)
from harness.sandbox.firewall import Firewall
from harness.utils.constants import INITIAL_COUNT


def _execute(command: list[str]) -> None:
    """Apply a firewall command inside this container or abort startup."""
    subprocess.run(command, check=True, timeout=CONTROL_TIMEOUT_SECONDS)


def _resolve(hostname: str, port: int) -> list[str]:
    """Resolve all provider addresses before removing DNS access from the agent."""
    return [str(address[INITIAL_COUNT]) for _, _, _, _, address in socket.getaddrinfo(
        hostname, port, type=socket.SOCK_STREAM,
    )]


def start_sandbox() -> None:
    """Require a target command, enforce networking, and irreversibly drop privileges."""
    command = sys.argv[COMMAND_ARGUMENT_OFFSET:]
    if not command:
        raise ValueError("sandbox bootstrap requires a command to execute")
    uid, gid = int(os.environ[RUN_UID_ENV]), int(os.environ[RUN_GID_ENV])
    if uid <= ROOT_USER_ID or gid <= ROOT_USER_ID:
        raise ValueError("sandbox must execute with a non-root user and group")
    endpoint = ProviderEndpoint.from_url(os.environ[PROVIDER_BASE_URL_ENV])
    Firewall(_execute, _resolve).restrict(endpoint, HOSTS_FILE)
    os.execvp(PRIVILEGE_EXECUTABLE, [
        PRIVILEGE_EXECUTABLE, "--reuid", str(uid), "--regid", str(gid), "--clear-groups",
        "--inh-caps=-all", "--ambient-caps=-all", "--bounding-set=-all", "--no-new-privs",
        "--", *command,
    ])


if __name__ == "__main__":
    start_sandbox()
