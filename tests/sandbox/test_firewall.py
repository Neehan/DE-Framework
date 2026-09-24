"""Provider-only firewall rules and fail-closed bootstrap behavior without host firewall changes."""

from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import Mock

import pytest
from harness.proxy.models import ProviderEndpoint
from harness.sandbox import bootstrap
from harness.sandbox.constants import (
    FIREWALL_EXECUTABLES,
    PROVIDER_BASE_URL_ENV,
    RUN_GID_ENV,
    RUN_UID_ENV,
)
from harness.sandbox.firewall import Firewall
from harness.utils.constants import COUNT_INCREMENT, INITIAL_COUNT, TEXT_ENCODING

from tests.constants import (
    FATAL_PROCESS_EXIT_CODE,
    FIREWALL_TEST_COMMAND,
    PROVIDER_TEST_ADDRESSES,
    PROVIDER_TEST_HOST,
    PROVIDER_TEST_PORT,
    SANDBOX_COMMAND,
    SANDBOX_PROVIDER_ENV,
)


def test_only_provider_tcp_addresses_are_allowed(tmp_path: Path) -> None:
    """Both families default to DROP; DNS, arbitrary ports, and loopback receive no exemptions."""
    execute = Mock()
    resolve = Mock(return_value=[*PROVIDER_TEST_ADDRESSES, *PROVIDER_TEST_ADDRESSES])
    hosts = tmp_path / "hosts"
    endpoint = ProviderEndpoint(PROVIDER_TEST_HOST, PROVIDER_TEST_PORT)
    Firewall(execute, resolve).restrict(endpoint, hosts)
    resolve.assert_called_once_with(PROVIDER_TEST_HOST, PROVIDER_TEST_PORT)
    expected = []
    for executable in FIREWALL_EXECUTABLES.values():
        expected.extend([[executable, "-P", "OUTPUT", "DROP"], [executable, "-F", "OUTPUT"]])
    for executable, address in zip(FIREWALL_EXECUTABLES.values(), PROVIDER_TEST_ADDRESSES):
        expected.append([
            executable, "-A", "OUTPUT", "-d", address, "-p", "tcp",
            "--dport", str(PROVIDER_TEST_PORT), "-j", "ACCEPT",
        ])
    assert [call.args[INITIAL_COUNT] for call in execute.call_args_list] == expected
    assert hosts.read_text(encoding=TEXT_ENCODING).splitlines()[COUNT_INCREMENT:] == [
        f"{address} {PROVIDER_TEST_HOST}" for address in PROVIDER_TEST_ADDRESSES
    ]


@pytest.mark.parametrize("addresses", ([], ["0.0.0.0"], ["::"], ["224.0.0.1"]))
def test_missing_or_unrestricted_provider_addresses_fail(tmp_path: Path, addresses: list[str]) -> None:
    """Invalid DNS answers cannot create a broad allow rule."""
    execute = Mock()
    with pytest.raises(ValueError, match="concrete unicast"):
        Firewall(execute, Mock(return_value=addresses)).restrict(
            ProviderEndpoint(PROVIDER_TEST_HOST, PROVIDER_TEST_PORT), tmp_path / "hosts",
        )
    execute.assert_not_called()


@pytest.mark.parametrize("url", ("", "provider", "ftp://provider", "http://user:secret@provider", "http://provider:0"))
def test_invalid_provider_url_fails_before_container_creation(url: str) -> None:
    """Require one explicit endpoint instead of a wildcard or implicit provider choice."""
    with pytest.raises(ValueError):
        ProviderEndpoint.from_url(url)


def test_bootstrap_never_launches_after_a_firewall_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed rule installation exits before either the SDK or a user command starts."""
    monkeypatch.setenv(RUN_UID_ENV, str(PROVIDER_TEST_PORT))
    monkeypatch.setenv(RUN_GID_ENV, str(PROVIDER_TEST_PORT))
    monkeypatch.setenv(PROVIDER_BASE_URL_ENV, SANDBOX_PROVIDER_ENV[PROVIDER_BASE_URL_ENV])
    monkeypatch.setattr(bootstrap.sys, "argv", ["bootstrap", *SANDBOX_COMMAND])
    failure = CalledProcessError(FATAL_PROCESS_EXIT_CODE, FIREWALL_TEST_COMMAND)
    restrict = Mock(side_effect=failure)
    launch = Mock()
    monkeypatch.setattr(Firewall, "restrict", restrict)
    monkeypatch.setattr(bootstrap.os, "execvp", launch)
    with pytest.raises(CalledProcessError):
        bootstrap.start_sandbox()
    launch.assert_not_called()
