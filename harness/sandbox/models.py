"""Validated settings for one isolated container."""

from dataclasses import dataclass

from harness.sandbox.constants import UNRESTRICTED_NETWORKS
from harness.utils.constants import INITIAL_COUNT


@dataclass(frozen=True)
class SandboxConfig:
    """Supply the image, network, and container command."""

    image: str
    network: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject empty image/network names, unrestricted networks, and empty commands."""
        if not self.image.strip() or not self.network.strip():
            raise ValueError("sandbox image and network must not be empty")
        if self.network in UNRESTRICTED_NETWORKS:
            raise ValueError("sandbox requires a dedicated restricted network")
        if not self.command or not self.command[INITIAL_COUNT].strip():
            raise ValueError("sandbox requires a command to execute")
