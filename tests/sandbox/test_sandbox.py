"""Host-side container ownership, selected input, and cancellation without a Docker daemon."""

import os
from asyncio import CancelledError, Event, create_task, timeout
from asyncio.subprocess import Process
from pathlib import Path
from subprocess import CalledProcessError
from typing import cast
from unittest.mock import AsyncMock, Mock, create_autospec

import pytest
from harness.sandbox.constants import (
    BOOTSTRAP_CAPABILITIES,
    BOOTSTRAP_MODULE,
    CONTAINER_NAME_PREFIX,
    CONTAINER_RUN_DIRECTORY,
    MISSING_CONTAINER_ERROR_PREFIX,
    ROOT_USER_ID,
    RUN_GID_ENV,
    RUN_UID_ENV,
)
from harness.sandbox.models import SandboxConfig
from harness.sandbox.sandbox import Sandbox
from harness.session.errors import AttemptAlreadyRunning
from harness.session.run_lock import RunLock
from harness.utils.constants import (
    INITIAL_COUNT,
    RATE_LIMIT_EXIT_CODE,
    RATE_LIMIT_RESET_PREFIX,
    SPEND_LIMIT_EXIT_CODE,
    TEXT_ENCODING,
)

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    FATAL_PROCESS_EXIT_CODE,
    ONE_ROUND,
    SANDBOX_COMMAND,
    SANDBOX_CONTAINER_ID,
    SANDBOX_IMAGE,
    SANDBOX_PROVIDER_ENV,
    SANDBOX_PROXY_ENV,
    SANDBOX_REQUEST,
)
from tests.support.helpers import make_process
from tests.support.models import SandboxProcesses


def _container_name(start_process: AsyncMock) -> str:
    """Return the unique name allocated before Docker creation began."""
    arguments = next(call.args for call in start_process.await_args_list if "create" in call.args)
    return arguments[arguments.index("--name") + ONE_ROUND]


async def test_mounts_only_attempt_and_passes_only_proxy_credentials(
    processes: SandboxProcesses, sandbox: Sandbox, seed_directory: Path, start_process: AsyncMock, proxy_factory: Mock,
) -> None:
    """Real provider credentials stay in the proxy; the container receives only its disposable run token."""
    await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)
    inspect, create, start, remove = start_process.await_args_list
    expected_mount = f"type=bind,source={seed_directory.resolve()},target={CONTAINER_RUN_DIRECTORY}"
    assert expected_mount in create.args
    assert create.args.count("--mount") == ONE_ROUND
    assert f"{ROOT_USER_ID}:{ROOT_USER_ID}" in create.args
    assert f"{RUN_UID_ENV}={os.getuid()}" in create.args
    assert f"{RUN_GID_ENV}={os.getgid()}" in create.args
    assert all(capability in create.args for capability in BOOTSTRAP_CAPABILITIES)
    assert create.args[-len(SANDBOX_COMMAND):] == SANDBOX_COMMAND
    assert BOOTSTRAP_MODULE in create.args
    assert SANDBOX_REQUEST not in create.args
    for name, value in SANDBOX_PROXY_ENV.items():
        assert f"{name}={value}" in create.args
    assert all(value not in " ".join(create.args) for value in SANDBOX_PROVIDER_ENV.values())
    proxy_factory.assert_called_once_with()
    proxy_factory.return_value.__aexit__.assert_awaited_once()
    container = _container_name(start_process)
    assert container.startswith(CONTAINER_NAME_PREFIX)
    assert start.args == ("docker", "start", "--attach", "--interactive", container)
    assert remove.args == ("docker", "rm", "--force", container)
    cast(AsyncMock, processes.execution.communicate).assert_awaited_once_with(
        SANDBOX_REQUEST.encode(TEXT_ENCODING)
    )
    assert seed_directory.is_dir()


async def test_failed_container_execution_still_removes_container(
    processes: SandboxProcesses, sandbox: Sandbox, seed_directory: Path, start_process: AsyncMock
) -> None:
    """Nonzero harness exits preserve the error and mounted storage while cleaning up."""
    processes.execution = make_process(b"", FATAL_PROCESS_EXIT_CODE)
    start_process.side_effect = [processes.inspection, processes.creation, processes.execution, processes.removal]
    with pytest.raises(CalledProcessError):
        await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)
    assert start_process.await_args_list[-ONE_ROUND].args == ("docker", "rm", "--force", _container_name(start_process))
    assert seed_directory.is_dir()


@pytest.mark.parametrize("failure", ("exit", "empty_output", "timeout"))
async def test_failed_creation_cleans_up_the_owned_name(
    processes: SandboxProcesses, sandbox: Sandbox, seed_directory: Path,
    start_process: AsyncMock, failure: str,
) -> None:
    """Creation can succeed at the daemon even when its ID never reaches the caller."""
    expected_error: type[Exception]
    if failure == "exit":
        processes.creation = make_process(b"", FATAL_PROCESS_EXIT_CODE)
        expected_error = CalledProcessError
    elif failure == "empty_output":
        cast(AsyncMock, processes.creation.communicate).return_value = (b"", b"")
        expected_error = RuntimeError
    else:
        cast(AsyncMock, processes.creation.communicate).side_effect = [TimeoutError(), (b"", b"")]
        expected_error = TimeoutError
    start_process.side_effect = [processes.inspection, processes.creation, processes.removal]
    with pytest.raises(expected_error):
        await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)
    assert start_process.await_args_list[-ONE_ROUND].args == ("docker", "rm", "--force", _container_name(start_process))


async def test_cancellation_reaps_client_and_removes_owned_container(
    processes: SandboxProcesses, sandbox: Sandbox, seed_directory: Path, start_process: AsyncMock
) -> None:
    """Stopping a live run terminates the Docker client and its container."""
    execution = create_autospec(Process, instance=True)
    execution.returncode = None
    execution.communicate.side_effect = [CancelledError(), (b"", b"")]
    start_process.side_effect = [processes.inspection, processes.creation, execution, processes.removal]
    with pytest.raises(CancelledError):
        await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)
    execution.kill.assert_called_once()
    assert start_process.await_args_list[-ONE_ROUND].args == ("docker", "rm", "--force", _container_name(start_process))


@pytest.mark.parametrize("creation_fails", (False, True))
async def test_cancellation_during_creation_waits_then_removes(
    processes: SandboxProcesses, sandbox: Sandbox, seed_directory: Path,
    start_process: AsyncMock, creation_fails: bool,
) -> None:
    """A cancelled create is allowed to return ownership before cleanup begins."""
    creating = Event()
    release = Event()

    async def communicate(content: bytes | None) -> tuple[bytes, bytes]:
        """Hold creation until the caller has cancelled the sandbox task."""
        creating.set()
        await release.wait()
        return (SANDBOX_CONTAINER_ID.encode(TEXT_ENCODING), b"")

    cast(AsyncMock, processes.creation.communicate).side_effect = communicate
    if creation_fails:
        cast(AsyncMock, processes.creation.wait).return_value = FATAL_PROCESS_EXIT_CODE
    start_process.side_effect = [processes.inspection, processes.creation, processes.removal]
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock))
        await creating.wait()
        task.cancel()
        release.set()
        with pytest.raises(CancelledError) as cancelled:
            await task
    if creation_fails:
        assert "Container creation also failed" in cancelled.value.__notes__[INITIAL_COUNT]
    assert start_process.await_args_list[-ONE_ROUND].args == ("docker", "rm", "--force", _container_name(start_process))


async def test_cleanup_failure_preserves_the_creation_error(
    processes: SandboxProcesses, sandbox: Sandbox, seed_directory: Path, start_process: AsyncMock,
) -> None:
    """Failure to remove an ambiguously created container must not replace its original error."""
    failure = TimeoutError("create reply lost")
    cast(AsyncMock, processes.creation.communicate).side_effect = [failure, (b"", b"")]
    start_process.side_effect = [processes.inspection, processes.creation, make_process(b"", FATAL_PROCESS_EXIT_CODE)]
    with pytest.raises(TimeoutError) as raised:
        await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)
    assert raised.value is failure
    assert "Container cleanup also failed" in raised.value.__notes__[INITIAL_COUNT]


@pytest.mark.parametrize("owned_container_missing", (True, False))
async def test_removal_ignores_only_absence_of_the_owned_container(
    processes: SandboxProcesses, sandbox: Sandbox, seed_directory: Path,
    start_process: AsyncMock, owned_container_missing: bool,
) -> None:
    """Only Docker's exact missing-owned-name diagnostic counts as successful cleanup."""
    processes.removal = make_process(b"", FATAL_PROCESS_EXIT_CODE)

    async def remove_output(content: bytes | None) -> tuple[bytes, bytes]:
        """Report either the owned name or an unrelated missing container."""
        name = _container_name(start_process) if owned_container_missing else SANDBOX_CONTAINER_ID
        return b"", f"{MISSING_CONTAINER_ERROR_PREFIX}{name}".encode(TEXT_ENCODING)

    cast(AsyncMock, processes.removal.communicate).side_effect = remove_output
    start_process.side_effect = [processes.inspection, processes.creation, processes.execution, processes.removal]
    if owned_container_missing:
        await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)
    else:
        with pytest.raises(CalledProcessError):
            await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)


async def test_overlapping_run_is_rejected_without_stopping_owner(
    processes: SandboxProcesses, sandbox: Sandbox, seed_directory: Path
) -> None:
    """A second invocation cannot take over the first invocation's container."""
    running = Event()
    release = Event()

    async def communicate(content: bytes | None) -> tuple[bytes, bytes]:
        """Expose an in-progress container execution."""
        running.set()
        await release.wait()
        return (b"", b"")

    cast(AsyncMock, processes.execution.communicate).side_effect = communicate
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock))
        await running.wait()
        with pytest.raises(RuntimeError, match="already running"):
            await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)
        release.set()
        await task


@pytest.mark.parametrize("network", ("host", "bridge", "default"))
def test_unrestricted_builtin_networks_are_rejected(network: str) -> None:
    """The wrapper cannot accidentally opt into host or default bridge networking."""
    with pytest.raises(ValueError, match="restricted network"):
        SandboxConfig(SANDBOX_IMAGE, network, SANDBOX_COMMAND)


async def test_abandoned_worker_is_removed_before_starting_replacement(
    processes: SandboxProcesses, sandbox: Sandbox, seed_directory: Path, start_process: AsyncMock,
) -> None:
    """A killed launcher can leave Docker running; host ownership must stop that worker before reuse."""
    cast(AsyncMock, processes.inspection.communicate).return_value = (SANDBOX_CONTAINER_ID.encode(TEXT_ENCODING), b"")
    start_process.side_effect = [
        processes.inspection, processes.removal, processes.creation, processes.execution, processes.removal,
    ]
    await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)
    inspect, stop_abandoned, create, _start, _remove = start_process.await_args_list
    assert str(seed_directory.resolve()) in inspect.args[-ONE_ROUND]
    assert stop_abandoned.args == ("docker", "rm", "--force", SANDBOX_CONTAINER_ID)
    assert "create" in create.args


@pytest.mark.parametrize("cancel_wait", [False, True])
@pytest.mark.parametrize("exit_code", [RATE_LIMIT_EXIT_CODE, SPEND_LIMIT_EXIT_CODE])
async def test_rate_limit_waits_without_a_worker_and_keeps_run_ownership(
    sandbox: Sandbox, seed_directory: Path, start_process: AsyncMock, processes: SandboxProcesses,
    proxy_factory: Mock, cancel_wait: bool, exit_code: int,
) -> None:
    """Cooldown waits outlive inference deadlines because the old worker is gone and its replacement has not started."""
    limited = make_process(b"", exit_code)
    reset_at = float(ONE_ROUND) if exit_code == RATE_LIMIT_EXIT_CODE else None
    limited.communicate.return_value = (b"", f"diagnostic\n{RATE_LIMIT_RESET_PREFIX}{reset_at}\n".encode(TEXT_ENCODING))
    start_process.side_effect = [processes.inspection, processes.creation, limited, processes.removal,
                                processes.creation, processes.execution, processes.removal]
    waiting, release = Event(), Event()

    async def recover_credential(spend_limited: bool, resets_at: float | None) -> None:
        """Hold credential availability independently of any SDK or Docker control timeout."""
        assert spend_limited == (exit_code == SPEND_LIMIT_EXIT_CODE)
        assert resets_at == reset_at
        waiting.set()
        await release.wait()

    proxy_factory.return_value.recover_credential.side_effect = recover_credential
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        task = create_task(sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock))
        await waiting.wait()
        assert [call.args[ONE_ROUND] for call in start_process.await_args_list] == ["ps", "create", "start", "rm"]
        with pytest.raises(AttemptAlreadyRunning), RunLock(seed_directory):
            pytest.fail("cooldown released the run lock")
        if cancel_wait:
            task.cancel()
            with pytest.raises(CancelledError):
                await task
        else:
            release.set()
            await task
    proxy_factory.return_value.__aexit__.assert_awaited_once()
    with RunLock(seed_directory):
        pass


async def test_rate_limited_worker_cleanup_failure_does_not_restart(
    sandbox: Sandbox, seed_directory: Path, start_process: AsyncMock, processes: SandboxProcesses, proxy_factory: Mock,
) -> None:
    """A credential change cannot hide failure to clean up its old worker."""
    limited = make_process(b"", RATE_LIMIT_EXIT_CODE)
    limited.communicate.return_value = (b"", f"{RATE_LIMIT_RESET_PREFIX}\n".encode(TEXT_ENCODING))
    start_process.side_effect = [processes.inspection, processes.creation, limited,
                                make_process(b"", FATAL_PROCESS_EXIT_CODE)]
    with pytest.raises(CalledProcessError):
        await sandbox.run(seed_directory, SANDBOX_REQUEST, RunLock)
    proxy_factory.return_value.recover_credential.assert_not_awaited()
