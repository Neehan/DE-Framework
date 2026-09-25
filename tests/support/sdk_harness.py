"""Shared SDK factory and connection lifecycle for protocol and persistence tests."""

from asyncio import Event
from collections.abc import Callable

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from harness.session.connection_manager import ConnectionManager

from tests.support.sdk_transport import SdkTransport


class SdkHarness[TransportType: SdkTransport]:
    """Capture real SDK clients and injected transports; wait_for_connection synchronizes retries."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        transport_factory: Callable[[ClaudeAgentOptions], TransportType],
    ) -> None:
        """Build a connection owner with the test's chosen in-memory or persistent transport."""
        self.clients: list[ClaudeSDKClient] = []
        self.transports: list[TransportType] = []
        self._transport_factory = transport_factory
        self._created = Event()
        self.connection = ConnectionManager(options, self._create_client)

    def _create_client(self, options: ClaudeAgentOptions) -> ClaudeSDKClient:
        """Record each fresh client so tests can inspect resume options and drive its output."""
        transport = self._transport_factory(options)
        client = ClaudeSDKClient(options, transport=transport)
        self.transports.append(transport)
        self.clients.append(client)
        self._created.set()
        return client

    async def wait_for_connection(self, count: int) -> None:
        """Wait for a specific connection without relying on elapsed time."""
        while len(self.transports) < count:
            await self._created.wait()
            self._created.clear()
