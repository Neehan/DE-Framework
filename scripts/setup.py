"""Prepare local datasets, Docker images, and optional Codex gateways on the host."""

import asyncio
import json
import logging
import os
from asyncio.subprocess import DEVNULL
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory

from aiohttp import ClientError, ClientResponseError, ClientSession, ClientTimeout
from codex.auth import read_credentials
from codex.constants import HEALTH_TIMEOUT_SECONDS, MODEL_ALIAS
from codex.gateway import Gateway
from codex.models import Account
from dotenv import dotenv_values, load_dotenv, set_key
from experiments.constants import REFERENCE_ROLE
from harness.sandbox.constants import CONTROL_TIMEOUT_SECONDS, SUCCESS_EXIT_CODE
from harness.sandbox.docker import Docker
from harness.utils.asyncio import run_process
from harness.utils.constants import COUNT_INCREMENT, TEXT_ENCODING
from launcher.constants import (
    DATASET_NAMES,
    DATASETS_DIRECTORY,
    DOTENV_FILE,
    FAILED_EXIT_CODE,
    LITELLM_PREFIX,
    LITELLM_URL_ENV,
    LOG_FORMAT,
)
from launcher.dataset import Dataset
from launcher.docker_environment import build_images
from launcher.provider import resolve_provider

from scripts.constants import (
    AUTH_FILE_PATTERN,
    AUTH_FILE_PREFIX,
    DATASET_FILE_URL,
    DATASET_REPO_URL_ENV,
    DOWNLOAD_TIMEOUT_SECONDS,
)


class Setup:
    """Prepare the repository using injected Docker, HTTP, and gateway clients; no overrides required."""

    def __init__(self, docker: Docker, client: ClientSession, gateway: Gateway) -> None:
        """Reuse production infrastructure without launching experiments or reading credentials into workers."""
        self._docker = docker
        self._client = client
        self._gateway = gateway

    async def run(self, env: Mapping[str, str]) -> None:
        """Validate local configuration, prepare data and images, then start configured account gateways."""
        accounts = self._read_accounts(env)
        key = self._read_gateway_key(accounts, env) if accounts else None
        async with asyncio.timeout(CONTROL_TIMEOUT_SECONDS):
            await self._docker.run(["info"], None, DEVNULL)
        await self._prepare_datasets(env.get(DATASET_REPO_URL_ENV))
        logging.info("Building run and audit images")
        await build_images(self._docker, True)
        if key is not None:
            logging.info("Building LiteLLM and preparing %d account gateways", len(accounts))
            await self._gateway.build()
            for account, path in accounts.items():
                await self._gateway.add(account, path)
            await self._gateway.start(list(accounts), key)
            self._save_gateway_urls(accounts)
        else:
            logging.info("No CODEX_AUTH_FILE entries configured; local LiteLLM setup skipped")

    def _read_accounts(self, env: Mapping[str, str]) -> dict[Account, Path]:
        """Validate numbered file paths and distinct subscriptions before touching Docker."""
        accounts = {}
        identities = set()
        for name, value in env.items():
            if not name.startswith(AUTH_FILE_PREFIX) or not value.strip():
                continue
            match = AUTH_FILE_PATTERN.fullmatch(name)
            if match is None:
                raise ValueError(f"use {AUTH_FILE_PREFIX}_<positive account number>: {name}")
            account = Account(int(match["account"]))
            path = (DOTENV_FILE.parent / Path(value.strip()).expanduser()).resolve()
            identity = json.loads(read_credentials(path))["account_id"]
            if identity in identities:
                raise ValueError("CODEX_AUTH_FILE entries contain duplicate subscriptions")
            identities.add(identity)
            accounts[account] = path
        return dict(sorted(accounts.items(), key=lambda item: item[0].number))

    def _read_gateway_key(self, accounts: dict[Account, Path], env: Mapping[str, str]) -> str:
        """Require one shared gateway key and reject endpoints that conflict with local account setup."""
        urls = {f"{LITELLM_URL_ENV}_{account.number}": account.url for account in accounts}
        for name, value in env.items():
            if name.startswith(LITELLM_URL_ENV) and value.strip() and value.strip().rstrip("/") != urls.get(name):
                raise ValueError(f"{name} conflicts with the configured local Codex accounts")
        route = resolve_provider(f"{LITELLM_PREFIX}{MODEL_ALIAS}", {**env, **urls})
        keys = {provider.credential for provider in route.providers}
        if len(keys) != COUNT_INCREMENT:
            raise ValueError("local Codex gateways require one shared LITELLM_API_KEY")
        return keys.pop()

    async def _prepare_datasets(self, repository_url: str | None) -> None:
        """Use local subsets first; warn about missing files without a configured repository URL."""
        DATASETS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        missing = []
        for name in DATASET_NAMES:
            destination = DATASETS_DIRECTORY / f"{name}.jsonl"
            if destination.exists():
                self._validate_dataset(DATASETS_DIRECTORY, name)
            else:
                missing.append(name)
        if not missing:
            return
        if repository_url is None or not repository_url.strip():
            logging.warning("Missing datasets: %s. Set %s or place their JSONL files in %s.",
                            ", ".join(missing), DATASET_REPO_URL_ENV, DATASETS_DIRECTORY)
            return
        for name in missing:
            await self._download_dataset(name, repository_url.strip().rstrip("/"))

    async def _download_dataset(self, name: str, repository_url: str) -> None:
        """Warn on a missing download; validate successful responses before publishing a local file."""
        url = DATASET_FILE_URL.format(repository=repository_url, name=name)
        logging.info("Downloading dataset: %s", name)
        try:
            async with self._client.get(url, timeout=ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS)) as response:
                response.raise_for_status()
                content = await response.read()
        except ClientResponseError as error:
            if error.status != HTTPStatus.NOT_FOUND:
                raise
            logging.warning("Dataset %s was not found at the configured repository; place %s.jsonl in %s.",
                            name, name, DATASETS_DIRECTORY)
            return
        with TemporaryDirectory(dir=DATASETS_DIRECTORY) as temporary:
            staged = Path(temporary) / f"{name}.jsonl"
            staged.write_bytes(content)
            self._validate_dataset(Path(temporary), name)
            os.link(staged, DATASETS_DIRECTORY / staged.name)

    def _validate_dataset(self, directory: Path, name: str) -> None:
        """Use the launcher's contracts to check problems, required sketches, reference proofs, and outlines."""
        dataset = Dataset(directory)
        problems = dataset.load(name, None, None, REFERENCE_ROLE)
        dataset.load_references(name, problems)

    def _save_gateway_urls(self, accounts: dict[Account, Path]) -> None:
        """Record ready gateway endpoints while preserving unrelated .env values and comments."""
        saved = dotenv_values(DOTENV_FILE)
        for account in accounts:
            name = f"{LITELLM_URL_ENV}_{account.number}"
            if saved.get(name) != account.url:
                set_key(DOTENV_FILE, name, account.url, quote_mode="never")
            logging.info("Account %d ready: %s", account.number, account.url)


async def prepare_repository() -> int:
    """Create production clients and load only the repository-root environment file."""
    if not DOTENV_FILE.is_file():
        raise FileNotFoundError("Copy .env.example to .env at the repository root and configure your providers first")
    load_dotenv(DOTENV_FILE, override=False)
    async with ClientSession(timeout=ClientTimeout(total=HEALTH_TIMEOUT_SECONDS), trust_env=False) as client:
        docker = Docker(asyncio.create_subprocess_exec)
        gateway = Gateway(docker, client, asyncio.sleep)
        await Setup(docker, client, gateway).run(os.environ)
    logging.info("Setup complete. Activate .venv and run launcher commands from the repository root.")
    return SUCCESS_EXIT_CODE


def main() -> None:
    """Report setup failures without exposing credential-bearing Docker input."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        code = run_process(prepare_repository())
    except CalledProcessError as error:
        logging.error("Docker setup failed (exit %s): %s", error.returncode, error.stderr.decode(TEXT_ENCODING))
        code = FAILED_EXIT_CODE
    except (ValueError, KeyError, OSError, ClientError, RuntimeError, TimeoutError) as error:
        logging.error("Setup failed: %s", error)
        code = FAILED_EXIT_CODE
    raise SystemExit(code)


if __name__ == "__main__":
    main()
