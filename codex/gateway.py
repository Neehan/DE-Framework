"""Manage long-lived LiteLLM account containers independently of experiment runs."""

import json
from asyncio import timeout
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import Awaitable, Callable
from io import BytesIO
from pathlib import Path
from tarfile import TarInfo
from tarfile import open as open_tar
from typing import Any

from aiohttp import ClientConnectionError, ClientSession
from harness.sandbox.docker import Docker
from harness.utils.asyncio import finish_before_cancelling
from harness.utils.constants import (
    INITIAL_COUNT,
    PROVIDER_TIMEOUT_SECONDS,
    SDK_MAX_API_RETRIES,
    TEXT_ENCODING,
)
from launcher.constants import IMPLEMENTATION_DIRECTORY

from codex.auth import read_credentials
from codex.constants import (
    ACCOUNT_LABEL,
    AUTH_DIRECTORY,
    BIND_HOST,
    CONFIG_DIRECTORY,
    CONFIG_FILENAME,
    HEALTH_POLL_SECONDS,
    HOST,
    IDENTITY_COMMAND,
    IMAGE,
    MODEL_ALIAS,
    PORT,
    PRIVATE_FILE_MODE,
    PYTHON,
    STARTUP_TIMEOUT_SECONDS,
)
from codex.models import Account


class Gateway:
    """Build, add, start, check, and stop account gateways using injected Docker, HTTP, and timing.

    Credentials persist in private volumes. No method deletes or replaces authentication; no overrides are required.
    """

    def __init__(self, docker: Docker, client: ClientSession, wait: Callable[[float], Awaitable[None]]) -> None:
        """Accept infrastructure dependencies without owning experiment or session state."""
        self._docker = docker
        self._client = client
        self._wait = wait

    async def build(self) -> None:
        """Build the pinned adapter with its verified system-content correction."""
        await self._docker.run([
            "build", "-t", IMAGE, "-f", str(IMPLEMENTATION_DIRECTORY / "docker/Dockerfile.codex"),
            str(IMPLEMENTATION_DIRECTORY),
        ], None, PIPE)

    async def add(self, account: Account, source: Path) -> None:
        """Import over stdin or reuse the same account's refreshed credentials without overwriting them."""
        content = read_credentials(source)
        await self._docker.run(["image", "inspect", IMAGE], None, DEVNULL)
        await self._docker.run(["volume", "create", account.volume], None, DEVNULL)
        await self._docker.run([
            "run", "--rm", "--interactive", "--network", "none", "--log-driver", "none",
            "--mount", self._auth_mount(account, False), "--entrypoint", PYTHON, IMAGE, "-m", "codex.auth",
        ], content, DEVNULL)

    async def start(self, accounts: list[Account], key: str) -> None:
        """Validate all account identities before starting one reusable gateway per subscription."""
        identities = [await self._read_identity(account) for account in accounts]
        if len(set(identities)) != len(identities):
            raise ValueError("selected accounts contain duplicate subscriptions")
        image = json.loads(await self._docker.run(["image", "inspect", IMAGE], None, PIPE))[INITIAL_COUNT]["Id"]
        for account in accounts:
            await self._start_account(account, key, image)

    async def check(self, account: Account, key: str) -> None:
        """Require an authenticated model-list response without spending inference tokens."""
        async with self._client.get(f"{account.url}/v1/models", headers={"Authorization": f"Bearer {key}"}) as response:
            response.raise_for_status()
            payload = await response.json()
        if not payload["data"]:
            raise ValueError(f"account {account.number} exposes no models")

    async def stop(self, account: Account) -> None:
        """Stop an owned gateway while retaining its container configuration and private volume."""
        state = await self._inspect_account(account)
        if state is not None and state["State"]["Running"]:
            await self._docker.run(["stop", account.container], None, DEVNULL)

    async def _start_account(self, account: Account, key: str, image: str) -> None:
        """Reuse the pinned gateway; cancellation leaves persistent account services available for retry."""
        state = await self._inspect_account(account)
        if state is None:
            await finish_before_cancelling(self._create_account(account, key), "Codex gateway creation")
        elif state["Image"] != image:
            raise ValueError(f"account {account.number} already uses a different image")
        if state is None or not state["State"]["Running"]:
            await self._docker.run(["start", account.container], None, DEVNULL)
        await self._wait_until_ready(account, key)

    async def _create_account(self, account: Account, key: str) -> None:
        """Create a gateway without run mounts, then copy its private configuration over stdin."""
        await self._docker.run([
            "create", "--name", account.container, "--restart", "unless-stopped", "--log-driver", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--label", f"{ACCOUNT_LABEL}={account.number}",
            "--env", f"CHATGPT_TOKEN_DIR={AUTH_DIRECTORY}",
            "--publish", f"{HOST}:{account.port}:{PORT}", "--mount", self._auth_mount(account, False), IMAGE,
            "--config", f"{CONFIG_DIRECTORY}/{CONFIG_FILENAME}", "--host", BIND_HOST, "--port", str(PORT),
        ], None, DEVNULL)
        try:
            await self._docker.run(["cp", "-", f"{account.container}:{CONFIG_DIRECTORY}"], self._configuration_archive(key), DEVNULL)
        except BaseException as error:
            try:
                await self._docker.run(["rm", account.container], None, DEVNULL)
            except Exception as cleanup_error:
                error.add_note(f"New gateway cleanup also failed: {cleanup_error}")
            raise

    def _configuration_archive(self, key: str) -> bytes:
        """Keep the gateway key out of Docker command arguments, environment metadata, and host files."""
        config = {
            "model_list": [{"model_name": MODEL_ALIAS, "model_info": {"mode": "responses"},
                            "litellm_params": {"model": f"chatgpt/responses/{MODEL_ALIAS}", "supports_system_message": False}}],
            "router_settings": {"num_retries": SDK_MAX_API_RETRIES, "timeout": PROVIDER_TIMEOUT_SECONDS,
                                "stream_timeout": PROVIDER_TIMEOUT_SECONDS},
            "general_settings": {"master_key": key},
        }
        content = json.dumps(config).encode(TEXT_ENCODING)
        output = BytesIO()
        with open_tar(fileobj=output, mode="w") as archive:
            entry = TarInfo(CONFIG_FILENAME)
            entry.size, entry.mode = len(content), PRIVATE_FILE_MODE
            archive.addfile(entry, BytesIO(content))
        return output.getvalue()

    async def _inspect_account(self, account: Account) -> dict[str, Any] | None:
        """Find the named container and verify ownership before operating on it."""
        found = await self._docker.run([
            "container", "ls", "--all", "--filter", f"name=^/{account.container}$", "--format", "{{.ID}}",
        ], None, PIPE)
        if not found.strip():
            return None
        state = json.loads(await self._docker.run(["inspect", account.container], None, PIPE))[INITIAL_COUNT]
        if state["Config"]["Labels"].get(ACCOUNT_LABEL) != str(account.number):
            raise ValueError(f"container {account.container} is not owned by this account")
        return state

    async def _read_identity(self, account: Account) -> str:
        """Require an existing volume and read only its account identity, never its tokens."""
        await self._docker.run(["volume", "inspect", account.volume], None, DEVNULL)
        result = await self._docker.run([
            "run", "--rm", "--network", "none", "--log-driver", "none", "--entrypoint", PYTHON,
            "--mount", self._auth_mount(account, True), IMAGE, "-c", IDENTITY_COMMAND,
        ], None, PIPE)
        identity = result.decode(TEXT_ENCODING).strip()
        if not identity:
            raise ValueError(f"account {account.number} has no subscription identity")
        return identity

    async def _wait_until_ready(self, account: Account, key: str) -> None:
        """Wait for local startup only; authentication and protocol errors fail immediately."""
        async with timeout(STARTUP_TIMEOUT_SECONDS):
            while True:
                state = await self._inspect_account(account)
                if state is None or not state["State"]["Running"]:
                    raise RuntimeError(f"account {account.number} exited during startup")
                try:
                    await self.check(account, key)
                    return
                except (ClientConnectionError, TimeoutError):
                    await self._wait(HEALTH_POLL_SECONDS)

    def _auth_mount(self, account: Account, readonly: bool) -> str:
        """Mount only this account's credentials; helpers reading identity use read-only access."""
        mount = f"type=volume,source={account.volume},target={AUTH_DIRECTORY}"
        return f"{mount},readonly" if readonly else mount
