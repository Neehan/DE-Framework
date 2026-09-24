"""Setup reruns, input failures, and dataset publication exercise the real host orchestration."""

import json
import logging
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponseError
from codex.models import Account
from dotenv import dotenv_values
from launcher.constants import DATASET_NAMES
from scripts.constants import DATASET_REPO_URL_ENV
from scripts.setup import Setup

from tests.codex.constants import ACCOUNT_NUMBER, AUTH_TOKENS, GATEWAY_KEY
from tests.constants import DATASET_REPO_URL, ONE_ROUND


async def test_setup_rerun_preserves_local_files_and_records_ready_gateways(
    repository_setup: Setup, setup_directory: Path, setup_client: MagicMock, setup_gateway: MagicMock, dataset_content: bytes,
) -> None:
    """Downloads happen once; auth paths and credentials stay intact and generated URLs work without manual copying."""
    auth = setup_directory / "account login.json"
    auth.write_text(json.dumps({"tokens": AUTH_TOKENS}))
    original = auth.read_bytes()
    env = {"CODEX_AUTH_FILE_1": auth.name, "LITELLM_API_KEY": GATEWAY_KEY, DATASET_REPO_URL_ENV: DATASET_REPO_URL}
    await repository_setup.run(env)
    dotenv = setup_directory / ".env"
    configured = dotenv.read_bytes()
    saved = dotenv_values(dotenv)
    assert saved["LITELLM_BASE_URL_1"] == Account(ACCOUNT_NUMBER).url
    assert saved["LITELLM_API_KEY"] == GATEWAY_KEY and saved["ANTHROPIC_API_KEY"] == "keep-this-value"
    await repository_setup.run({**env, "LITELLM_BASE_URL_1": Account(ACCOUNT_NUMBER).url})
    assert dotenv.read_bytes() == configured and auth.read_bytes() == original
    assert setup_client.get.call_count == len(DATASET_NAMES)
    for name in DATASET_NAMES:
        assert (setup_directory / "datasets" / f"{name}.jsonl").read_bytes() == dataset_content
    setup_gateway.add.assert_awaited_with(Account(ACCOUNT_NUMBER), auth)
    setup_gateway.start.assert_awaited_with([Account(ACCOUNT_NUMBER)], GATEWAY_KEY)


@pytest.mark.parametrize("invalid", ["duplicate", "missing-file", "bad-slot", "conflicting-url", "missing-key"])
async def test_invalid_account_configuration_fails_before_docker(
    repository_setup: Setup, setup_directory: Path, setup_docker: MagicMock, setup_client: MagicMock, invalid: str,
) -> None:
    """Reject ambiguous credentials and endpoints before downloads, builds, or account imports."""
    auth = setup_directory / "auth.json"
    auth.write_text(json.dumps({"tokens": AUTH_TOKENS}))
    env = {"CODEX_AUTH_FILE_1": str(auth), "LITELLM_API_KEY": GATEWAY_KEY}
    if invalid == "duplicate":
        env["CODEX_AUTH_FILE_2"] = str(auth)
    elif invalid == "missing-file":
        env["CODEX_AUTH_FILE_1"] = str(setup_directory / "missing.json")
    elif invalid == "bad-slot":
        env["CODEX_AUTH_FILE_0"] = env.pop("CODEX_AUTH_FILE_1")
    elif invalid == "conflicting-url":
        env["LITELLM_BASE_URL"] = "https://remote.example"
    else:
        env.pop("LITELLM_API_KEY")
    with pytest.raises((ValueError, FileNotFoundError)):
        await repository_setup.run(env)
    setup_docker.run.assert_not_awaited()
    setup_client.get.assert_not_called()


@pytest.mark.parametrize("existing", [False, True])
async def test_invalid_dataset_is_not_published_or_replaced(
    repository_setup: Setup, setup_directory: Path, setup_docker: MagicMock, setup_client: MagicMock, existing: bool,
) -> None:
    """A corrupt existing file or failed download stops setup without publishing partial data or building images."""
    destination = setup_directory / "datasets" / f"{DATASET_NAMES[0]}.jsonl"
    content = b"not a dataset"
    if existing:
        destination.parent.mkdir()
        destination.write_bytes(content)
    setup_client.get.return_value.__aenter__.return_value.read.return_value = content
    with pytest.raises(ValueError):
        await repository_setup.run({DATASET_REPO_URL_ENV: DATASET_REPO_URL})
    if existing:
        assert destination.read_bytes() == content
        setup_client.get.assert_not_called()
    else:
        assert not destination.exists()
        assert not list(destination.parent.iterdir())
    assert setup_docker.run.await_count == ONE_ROUND


async def test_external_provider_setup_does_not_manage_local_gateways(
    repository_setup: Setup, setup_directory: Path, setup_gateway: MagicMock,
) -> None:
    """Anthropic, Meta, and externally managed LiteLLM need images and local data but no imported Codex account."""
    dotenv = setup_directory / ".env"
    original = dotenv.read_bytes()
    await repository_setup.run({"LITELLM_BASE_URL": "https://remote.example", "LITELLM_API_KEY": GATEWAY_KEY,
                                DATASET_REPO_URL_ENV: DATASET_REPO_URL})
    setup_gateway.build.assert_not_awaited()
    setup_gateway.add.assert_not_awaited()
    setup_gateway.start.assert_not_awaited()
    assert dotenv.read_bytes() == original


@pytest.mark.parametrize("repository_url", [None, "", "   ", DATASET_REPO_URL])
async def test_local_datasets_need_no_repository_or_download(
    repository_setup: Setup, setup_directory: Path, setup_client: MagicMock, dataset_content: bytes,
    caplog: pytest.LogCaptureFixture, repository_url: str | None,
) -> None:
    """Complete local data suppresses downloads and warnings regardless of whether a repository is configured."""
    directory = setup_directory / "datasets"
    directory.mkdir()
    for name in DATASET_NAMES:
        (directory / f"{name}.jsonl").write_bytes(dataset_content)
    env = {} if repository_url is None else {DATASET_REPO_URL_ENV: repository_url}
    with caplog.at_level(logging.WARNING):
        await repository_setup.run(env)
    setup_client.get.assert_not_called()
    assert not caplog.records
    assert all((directory / f"{name}.jsonl").read_bytes() == dataset_content for name in DATASET_NAMES)


@pytest.mark.parametrize("repository_url", [None, "", "   "])
async def test_missing_repository_warns_once_and_continues_setup(
    repository_setup: Setup, setup_directory: Path, setup_docker: MagicMock, setup_client: MagicMock,
    caplog: pytest.LogCaptureFixture, repository_url: str | None,
) -> None:
    """Missing data without a repository does not prevent image setup or attempt any HTTP requests."""
    env = {} if repository_url is None else {DATASET_REPO_URL_ENV: repository_url}
    with caplog.at_level(logging.WARNING):
        await repository_setup.run(env)
    setup_client.get.assert_not_called()
    assert len(caplog.records) == ONE_ROUND
    assert all(name in caplog.text for name in DATASET_NAMES) and DATASET_REPO_URL_ENV in caplog.text
    assert not list((setup_directory / "datasets").iterdir())
    assert any(call.args[0][0] == "build" for call in setup_docker.run.await_args_list)


@pytest.mark.parametrize("status", [HTTPStatus.OK, HTTPStatus.NOT_FOUND, HTTPStatus.FORBIDDEN])
async def test_only_missing_subset_is_downloaded_and_missing_link_warns(
    repository_setup: Setup, setup_directory: Path, setup_client: MagicMock, dataset_content: bytes,
    caplog: pytest.LogCaptureFixture, status: HTTPStatus,
) -> None:
    """Preserve local data, publish valid downloads, warn on 404, and fail on other HTTP errors."""
    directory = setup_directory / "datasets"
    directory.mkdir()
    local_name, missing_name = DATASET_NAMES
    local = directory / f"{local_name}.jsonl"
    local.write_bytes(dataset_content)
    if status != HTTPStatus.OK:
        response = setup_client.get.return_value.__aenter__.return_value
        response.raise_for_status.side_effect = ClientResponseError(MagicMock(), (), status=status)
    env = {DATASET_REPO_URL_ENV: f" {DATASET_REPO_URL}/ "}
    with caplog.at_level(logging.WARNING):
        if status == HTTPStatus.FORBIDDEN:
            with pytest.raises(ClientResponseError):
                await repository_setup.run(env)
        else:
            await repository_setup.run(env)
    assert setup_client.get.call_count == ONE_ROUND
    assert setup_client.get.call_args.args[0] == f"{DATASET_REPO_URL}/resolve/main/{missing_name}.jsonl"
    assert local.read_bytes() == dataset_content
    if status == HTTPStatus.OK:
        assert (directory / f"{missing_name}.jsonl").read_bytes() == dataset_content
        assert not caplog.records
    else:
        assert list(directory.iterdir()) == [local]
        assert len(caplog.records) == (ONE_ROUND if status == HTTPStatus.NOT_FOUND else 0)
