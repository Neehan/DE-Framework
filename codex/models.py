"""Validated account identity and derived gateway addresses."""

from dataclasses import dataclass

from codex.constants import (
    AUTH_VOLUME_PREFIX,
    BASE_PORT,
    CONTAINER_PREFIX,
    HOST,
    MAX_PORT,
    MIN_ACCOUNT,
)


@dataclass(frozen=True)
class Account:
    """Identify one subscription slot and derive its resources without redundant fields."""

    number: int

    def __post_init__(self) -> None:
        """Reject invalid slots before touching Docker or credential files."""
        if not MIN_ACCOUNT <= self.number <= MAX_PORT - BASE_PORT:
            raise ValueError(f"account must be between {MIN_ACCOUNT} and {MAX_PORT - BASE_PORT}")

    @property
    def container(self) -> str:
        """Return the stable container name owned by this account."""
        return f"{CONTAINER_PREFIX}-{self.number}"

    @property
    def volume(self) -> str:
        """Return the private credential volume, retained when the container stops."""
        return f"{AUTH_VOLUME_PREFIX}-{self.number}"

    @property
    def port(self) -> int:
        """Return the account's loopback-only host port."""
        return BASE_PORT + self.number

    @property
    def url(self) -> str:
        """Return the endpoint reachable by the host filtering proxy."""
        return f"http://{HOST}:{self.port}"
