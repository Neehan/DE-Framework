"""Probe file isolation, the production firewall, and dropped privileges inside the actual sandbox."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import urllib.request
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError

from claude_agent_sdk import ClaudeAgentOptions
from harness.proxy.constants import AUTH_ENV_HEADERS, PROVIDER_CREDENTIAL_COUNT
from harness.sandbox.constants import (
    CONTAINER_RUN_DIRECTORY,
    PROVIDER_BASE_URL_ENV,
    ROOT_USER_ID,
    SUCCESS_EXIT_CODE,
)
from harness.self_refine.models import RefinementState, RunStatus
from harness.session.constants import (
    BASH_DENIALS,
    CLI_PATH_ENV,
    DISALLOWED_TOOLS,
    LOCAL_TOOLS,
    MANAGED_SETTINGS_FILE,
)
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import COUNT_INCREMENT, TEXT_ENCODING, ZERO_TOKENS

from tests.constants import BUDGET_TOKENS, PROBLEM
from tests.integration.constants import (
    ABSOLUTE_ESCAPE_LINK,
    BLOCKED_PORT,
    CAPABILITY_FIELDS,
    DNS_QUERY,
    DNS_RESPONSE_BYTES,
    DOCKER_DNS_ENDPOINT,
    DOCKER_SOCKET,
    HEXADECIMAL_BASE,
    NETWORK_REPORT_FILENAME,
    OUTPUT_TOKENS,
    OWN_RUN_CONTENT,
    OWN_RUN_FILE,
    PROVIDER_MODEL,
    RELATIVE_ESCAPE_LINK,
    SIBLING_SOLUTION,
    SOCKET_TIMEOUT_SECONDS,
    TEST_UPSTREAM_KEY,
    TOOL_FILE,
    TOOL_PROBLEM,
)
from tests.integration.native_session_check import run_refinement


def require_blocked_tcp(host: str, port: int) -> None:
    """Fail if any non-provider TCP destination can be reached."""
    try:
        connection = socket.create_connection((host, port), timeout=SOCKET_TIMEOUT_SECONDS)
    except OSError:
        return
    connection.close()
    raise AssertionError(f"Forbidden endpoint reachable: {host}:{port}")


def require_blocked_dns() -> None:
    """Send a valid query to Docker's DNS server and require that no answer arrives."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as dns:
        dns.settimeout(SOCKET_TIMEOUT_SECONDS)
        try:
            dns.sendto(DNS_QUERY, DOCKER_DNS_ENDPOINT)
            dns.recvfrom(DNS_RESPONSE_BYTES)
        except OSError:
            return
    raise AssertionError("Arbitrary DNS remains reachable")


def require_dropped_privileges() -> None:
    """Check all Linux capability sets, privilege escalation, and firewall/hosts access."""
    status = dict(line.split(":", COUNT_INCREMENT) for line in Path("/proc/self/status").read_text().splitlines())
    for field in CAPABILITY_FIELDS:
        assert int(status[field].strip(), HEXADECIMAL_BASE) == ZERO_TOKENS, field
    assert status["NoNewPrivs"].strip() == "1"
    assert subprocess.run(["iptables", "-L"], capture_output=True).returncode != SUCCESS_EXIT_CODE
    assert not os.access("/etc/hosts", os.W_OK)
    assert json.loads(MANAGED_SETTINGS_FILE.read_text())["permissions"]["deny"] == sorted(DISALLOWED_TOOLS | BASH_DENIALS)
    for path in (MANAGED_SETTINGS_FILE, MANAGED_SETTINGS_FILE.parent):
        assert path.stat().st_uid == ROOT_USER_ID
        assert not os.access(path, os.W_OK)


def require_isolated_files(hidden_files: list[str]) -> None:
    """Read owned scratch, but reject host files, sibling paths, and symlink escape attempts."""
    directory = Path(CONTAINER_RUN_DIRECTORY)
    assert (directory / OWN_RUN_FILE).read_text(encoding=TEXT_ENCODING) == OWN_RUN_CONTENT
    forbidden = [
        *map(Path, hidden_files), directory.parent / SIBLING_SOLUTION,
        directory / ABSOLUTE_ESCAPE_LINK, directory / RELATIVE_ESCAPE_LINK,
    ]
    assert (directory / ABSOLUTE_ESCAPE_LINK).is_symlink()
    assert (directory / RELATIVE_ESCAPE_LINK).is_symlink()
    assert not DOCKER_SOCKET.exists()
    for path in forbidden:
        try:
            path.read_bytes()
        except (FileNotFoundError, PermissionError):
            continue
        raise AssertionError(f"Forbidden file readable: {path}")


def require_filtered_requests() -> None:
    """Try hosted search through raw Python; possessing the proxy token must not bypass the policy."""
    assert TEST_UPSTREAM_KEY not in os.environ.values()
    credentials = [(name, os.environ[name]) for name in AUTH_ENV_HEADERS if name in os.environ]
    assert len(credentials) == PROVIDER_CREDENTIAL_COUNT
    name, token = credentials.pop()
    header = AUTH_ENV_HEADERS[name]
    body = {"model": PROVIDER_MODEL, "max_tokens": OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": PROBLEM}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]}
    request = urllib.request.Request(
        f"{os.environ[PROVIDER_BASE_URL_ENV]}/v1/messages", data=json.dumps(body).encode(TEXT_ENCODING),
        headers={"Content-Type": "application/json", header: f"Bearer {token}" if header == "authorization" else token},
    )
    try:
        urllib.request.urlopen(request, timeout=SOCKET_TIMEOUT_SECONDS)
    except HTTPError as error:
        assert error.code == HTTPStatus.BAD_REQUEST
    else:
        raise AssertionError("Raw Python enabled provider-side search")


async def require_native_inference() -> None:
    """Execute native tools, stream their results through the proxy, and finish solve/critique."""
    manager = SessionManager(Path(CONTAINER_RUN_DIRECTORY) / "native", RunLock)
    result = await run_refinement(
        manager, RefinementState(TOOL_PROBLEM, BUDGET_TOKENS, None),
        ClaudeAgentOptions(model=PROVIDER_MODEL, allowed_tools=list(LOCAL_TOOLS)),
    )
    assert result.status == RunStatus.FINISHED
    assert Path(TOOL_FILE).read_text() == "after proxy edit\n"


def main() -> None:
    """Check reachable and forbidden endpoints, then export evidence through the owned run directory."""
    request = json.load(sys.stdin)
    assert os.getuid() == request["uid"] and os.getgid() == request["gid"]
    require_isolated_files(request["hidden_files"])
    require_filtered_requests()
    asyncio.run(require_native_inference())
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
        listener.bind(("::1", BLOCKED_PORT))
        listener.listen()
        for host, port in request["blocked"]:
            require_blocked_tcp(host, port)
    require_blocked_dns()
    require_dropped_privileges()
    version = subprocess.check_output([os.environ[CLI_PATH_ENV], "--version"], text=True).strip()
    report = {"blocked": request["blocked"], "cli_version": version, "file_isolation_checked": True}
    (Path(CONTAINER_RUN_DIRECTORY) / NETWORK_REPORT_FILENAME).write_text(json.dumps(report), encoding=TEXT_ENCODING)


if __name__ == "__main__":
    main()
