"""Container-only egress policy for the run's filtering proxy endpoint."""

from collections.abc import Callable
from ipaddress import ip_address
from pathlib import Path

from harness.proxy.models import ProviderEndpoint
from harness.sandbox.constants import FIREWALL_EXECUTABLES
from harness.utils.constants import TEXT_ENCODING


class Firewall:
    """restrict pins proxy DNS and permits only its TCP endpoint; inject commands and DNS for tests."""

    def __init__(
        self, execute: Callable[[list[str]], None], resolve: Callable[[str, int], list[str]],
    ) -> None:
        """Accept the container command executor and pre-lockdown hostname resolver."""
        self._execute = execute
        self._resolve = resolve

    def restrict(self, endpoint: ProviderEndpoint, hosts_file: Path) -> None:
        """Fail before launch unless both IP families block all other outbound traffic."""
        addresses = sorted({ip_address(value) for value in self._resolve(endpoint.hostname, endpoint.port)}, key=str)
        if not addresses or any(address.is_unspecified or address.is_multicast for address in addresses):
            raise ValueError("provider must resolve to concrete unicast addresses")
        with hosts_file.open("a", encoding=TEXT_ENCODING) as hosts:
            hosts.write("\n" + "".join(f"{address} {endpoint.hostname}\n" for address in addresses))
        for executable in FIREWALL_EXECUTABLES.values():
            self._execute([executable, "-P", "OUTPUT", "DROP"])
            self._execute([executable, "-F", "OUTPUT"])
        for address in addresses:
            self._execute([
                FIREWALL_EXECUTABLES[address.version], "-A", "OUTPUT", "-d", str(address),
                "-p", "tcp", "--dport", str(endpoint.port), "-j", "ACCEPT",
            ])
