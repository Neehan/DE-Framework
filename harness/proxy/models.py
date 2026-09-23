"""Provider endpoint parsing and host-only upstream authentication."""

from dataclasses import dataclass, field
from urllib.parse import urlsplit

from harness.proxy.constants import (
    ANTHROPIC_VERSION,
    AUTH_ENV_HEADERS,
    OAUTH_BETA,
    OAUTH_TOKEN_ENV,
)
from harness.sandbox.constants import (
    MIN_PROVIDER_PORT,
    PROVIDER_PORTS,
)


@dataclass(frozen=True)
class ProviderEndpoint:
    """An HTTP endpoint; the container firewall receives the filtering proxy's endpoint."""

    hostname: str
    port: int

    @classmethod
    def from_url(cls, url: str) -> "ProviderEndpoint":
        """Require an explicit HTTP(S) provider URL without embedded credentials."""
        parsed = urlsplit(url)
        if parsed.scheme not in PROVIDER_PORTS or not parsed.hostname:
            raise ValueError("provider requires an absolute http:// or https:// base URL")
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("provider base URL must not contain credentials, query, or fragment")
        port = parsed.port if parsed.port is not None else PROVIDER_PORTS[parsed.scheme]
        if port < MIN_PROVIDER_PORT:
            raise ValueError("provider port must be positive")
        return cls(parsed.hostname.encode("idna").decode("ascii"), port)


@dataclass(frozen=True)
class ProviderConfig:
    """Bind a fixed upstream URL and authentication headers; never pass this object into Docker."""

    url: str
    auth_variable: str
    credential: str = field(repr=False)

    def __post_init__(self) -> None:
        """Validate the fixed endpoint and credential once at the host boundary."""
        ProviderEndpoint.from_url(self.url)
        if self.auth_variable not in AUTH_ENV_HEADERS or not self.credential.strip():
            raise ValueError("provider requires one supported, nonempty credential")
        if any(character in self.credential for character in "\r\n"):
            raise ValueError("provider credential must be a single header value")

    @property
    def headers(self) -> dict[str, str]:
        """Create upstream headers while preserving the caller's API-key, bearer, or OAuth mode."""
        header, value = self.get_auth_header(self.credential)
        headers = {header: value, "anthropic-version": ANTHROPIC_VERSION}
        if self.auth_variable == OAUTH_TOKEN_ENV:
            headers["anthropic-beta"] = OAUTH_BETA
        return headers

    def get_auth_header(self, credential: str) -> tuple[str, str]:
        """Format either the upstream credential or run token using the configured authentication mode."""
        header = AUTH_ENV_HEADERS[self.auth_variable]
        return header, f"Bearer {credential}" if header == "authorization" else credential
