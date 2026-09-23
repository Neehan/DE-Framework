"""Account ownership, duplicate prevention, safe configuration, and persistent lifecycle."""

import json
from asyncio.subprocess import DEVNULL
from io import BytesIO
from subprocess import CalledProcessError
from tarfile import open as open_tar
from unittest.mock import MagicMock

import pytest
from codex.constants import (
    ACCOUNT_LABEL,
    CONFIG_FILENAME,
    HOST,
    MODEL_ALIAS,
    PRIVATE_FILE_MODE,
)
from codex.gateway import Gateway
from codex.main import parse_arguments
from codex.models import Account

from tests.codex.constants import (
    ACCOUNT_NUMBER,
    CONFIG_ERROR,
    CONFIG_ERROR_CODE,
    GATEWAY_KEY,
    IMAGE_ID,
    SECOND_ACCOUNT_NUMBER,
)


async def test_gateway_creation_keeps_key_out_of_arguments_and_retains_auth_on_stop(gateway: Gateway, docker_client: MagicMock) -> None:
    """Start a loopback-only gateway with private stdin configuration; stop never removes its volume."""
    account = Account(ACCOUNT_NUMBER)
    state = {"Image": IMAGE_ID, "State": {"Running": True}, "Config": {"Labels": {ACCOUNT_LABEL: str(account.number)}}}
    docker_client.run.side_effect = [b"", b"identity", json.dumps([{"Id": IMAGE_ID}]).encode(), b"", b"", b"", b"",
                                     b"container", json.dumps([state]).encode()]
    await gateway.start([account], GATEWAY_KEY)
    calls = docker_client.run.call_args_list
    assert GATEWAY_KEY not in repr([call.args[0] for call in calls])
    creation = next(call.args[0] for call in calls if call.args[0][0] == "create")
    assert f"{HOST}:{account.port}:4000" in creation
    payload = next(call.args[1] for call in calls if call.args[0][0] == "cp")
    with open_tar(fileobj=BytesIO(payload)) as archive:
        entry = archive.getmember(CONFIG_FILENAME)
        handle = archive.extractfile(entry)
        assert handle is not None and entry.mode == PRIVATE_FILE_MODE
        config = json.load(handle)
    assert config["general_settings"]["master_key"] == GATEWAY_KEY
    assert config["model_list"][0]["model_name"] == MODEL_ALIAS
    docker_client.run.reset_mock(side_effect=True)
    docker_client.run.side_effect = [b"container", json.dumps([state]).encode(), b""]
    await gateway.stop(account)
    docker_client.run.assert_called_with(["stop", account.container], None, DEVNULL)
    assert all("rm" not in call.args[0] for call in docker_client.run.call_args_list)


async def test_duplicate_subscriptions_fail_before_container_creation(gateway: Gateway, docker_client: MagicMock) -> None:
    """Two slots with the same account cannot overweight the shared rate-limit pool."""
    docker_client.run.side_effect = [b"", b"same-account", b"", b"same-account"]
    with pytest.raises(ValueError, match="duplicate"):
        await gateway.start([Account(ACCOUNT_NUMBER), Account(SECOND_ACCOUNT_NUMBER)], GATEWAY_KEY)
    assert all("create" not in call.args[0] for call in docker_client.run.call_args_list)


@pytest.mark.parametrize("create_fails", [True, False])
async def test_failed_creation_never_removes_someone_elses_container(
    gateway: Gateway, docker_client: MagicMock, create_fails: bool,
) -> None:
    """Remove only a newly created container whose config copy failed; retain all auth volumes."""
    error = CalledProcessError(CONFIG_ERROR_CODE, "docker", stderr=CONFIG_ERROR)
    prefix = [b"", b"identity", json.dumps([{"Id": IMAGE_ID}]).encode(), b""]
    docker_client.run.side_effect = prefix + ([error] if create_fails else [b"", error, b""])
    account = Account(ACCOUNT_NUMBER)
    with pytest.raises(CalledProcessError) as caught:
        await gateway.start([account], GATEWAY_KEY)
    assert caught.value is error
    removals = [call.args[0] for call in docker_client.run.call_args_list if call.args[0][0] == "rm"]
    assert removals == ([] if create_fails else [["rm", account.container]])


async def test_foreign_container_is_never_stopped(gateway: Gateway, docker_client: MagicMock) -> None:
    """Name collisions alone do not authorize operating on someone else's container."""
    docker_client.run.side_effect = [b"foreign", json.dumps([{"Config": {"Labels": {}}}]).encode()]
    with pytest.raises(ValueError, match="not owned"):
        await gateway.stop(Account(ACCOUNT_NUMBER))
    assert all("stop" not in call.args[0] for call in docker_client.run.call_args_list)


def test_cli_requires_explicit_unique_accounts() -> None:
    """Reject ambiguous account selection and unnecessary model configuration at gateway startup."""
    assert parse_arguments(["start", "--accounts", "1", "2"]).accounts == [Account(ACCOUNT_NUMBER), Account(SECOND_ACCOUNT_NUMBER)]
    with pytest.raises(SystemExit):
        parse_arguments(["start", "--accounts", "1", "1"])
    with pytest.raises(ValueError):
        parse_arguments(["stop", "--accounts", "0"])
