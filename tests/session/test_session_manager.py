"""Phase checkpoint, rollback, locking, and completion tests without provider calls."""

import os
import stat
import subprocess
import sys
from asyncio import create_subprocess_exec, timeout
from asyncio.subprocess import PIPE
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pytest
from harness.self_refine.models import Phase, RefinementState, RunStatus
from harness.session.constants import (
    CHECKPOINT_FILENAME,
    LOCK_FILENAME,
    RUNTIME_DIRECTORY,
    SDK_DEBUG_DIRECTORY,
    SESSION_FILENAME,
    WORKSPACE_DIRECTORY,
)
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import TEXT_ENCODING, ZERO_TOKENS

from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    BUDGET_TOKENS,
    FAILED_PHASE_FILENAME,
    FATAL_PROCESS_EXIT_CODE,
    FIRST_SOLUTION_FILENAME,
    FORK_CRASH_PROBE,
    INITIAL_SOLUTION,
    LOCK_HELD_EXIT_CODE,
    LOCK_HOLDER,
    LOCK_PROBE,
    NATIVE_SESSION_ID,
    OWNER_DIRECTORY_MODE,
    PHASE_TOKENS,
    PROBLEM,
    READ_ONLY_DIRECTORY_MODE,
    REVISED_SOLUTION,
    SDK_MESSAGE_ID,
    TWO_ROUNDS,
    WORKSPACE_FILENAME,
)


def test_storage_operations_need_no_sdk(manager: SessionManager, state: RefinementState) -> None:
    """Opening, saving, and restoring metadata require no SDK client or provider configuration."""
    manager.open(state)
    manager.checkpoint()
    restored = manager.restore()
    manager.close()
    assert restored.problem == PROBLEM


def test_read_checkpoint_preserves_export_and_does_not_restore_runtime(
    saved_session: SessionManager, seed_directory: Path,
) -> None:
    """Read authoritative saved progress without repairing exports or extracting native files."""
    saved_session.close()
    archive = (seed_directory / SESSION_FILENAME).read_bytes()
    (seed_directory / FIRST_SOLUTION_FILENAME).write_text(REVISED_SOLUTION, encoding=TEXT_ENCODING)
    saved = saved_session.read_checkpoint()
    assert saved.refinement.solution == INITIAL_SOLUTION
    assert saved.session_id == NATIVE_SESSION_ID
    assert (seed_directory / SESSION_FILENAME).read_bytes() == archive
    assert (seed_directory / FIRST_SOLUTION_FILENAME).read_text(encoding=TEXT_ENCODING) == REVISED_SOLUTION
    assert not (seed_directory / RUNTIME_DIRECTORY).exists()


@pytest.mark.parametrize("change", ("problem", "budget"))
def test_mismatched_attempt_does_not_restore_or_modify_files(
    manager: SessionManager, state: RefinementState, seed_directory: Path, change: str,
) -> None:
    """A wrong seed path fails before restoring files or silently returning another task."""
    manager.open(state)
    manager.close()
    original = (seed_directory / SESSION_FILENAME).read_bytes()
    requested = RefinementState(PROBLEM, BUDGET_TOKENS, None)
    if change == "problem":
        requested.problem = REVISED_SOLUTION
    else:
        requested.budget_tokens += PHASE_TOKENS
    with pytest.raises(ValueError, match="does not match"):
        manager.open(requested)
    assert (seed_directory / SESSION_FILENAME).read_bytes() == original
    assert not (seed_directory / RUNTIME_DIRECTORY).exists()


def test_fork_refuses_existing_destination(manager: SessionManager, state: RefinementState, seed_directory: Path) -> None:
    """Explicit forks never overwrite an existing seed's data."""
    manager.open(state)
    target = seed_directory.parent / "existing"
    target.mkdir()
    solution = target / FIRST_SOLUTION_FILENAME
    solution.write_text(INITIAL_SOLUTION, encoding=TEXT_ENCODING)
    with pytest.raises(FileExistsError):
        manager.fork(SessionManager(target, RunLock), None)
    assert solution.read_text(encoding=TEXT_ENCODING) == INITIAL_SOLUTION


def test_archive_preserves_links_without_copying_targets_or_debug_logs(
    manager: SessionManager, state: RefinementState, seed_directory: Path, tmp_path: Path,
) -> None:
    """Internal links stay portable; external link targets and disposable SDK debug files stay outside the ZIP."""
    manager.open(state)
    workspace = manager.runtime_directory / WORKSPACE_DIRECTORY
    work = workspace / WORKSPACE_FILENAME
    work.write_text(INITIAL_SOLUTION, encoding=TEXT_ENCODING)
    external = tmp_path / "external"
    external.write_text(REVISED_SOLUTION, encoding=TEXT_ENCODING)
    (workspace / "internal").symlink_to(work)
    (workspace / "external").symlink_to(external)
    debug = manager.runtime_directory / SDK_DEBUG_DIRECTORY
    debug.mkdir()
    (debug / "latest").symlink_to(external)
    manager.checkpoint()
    manager.close()
    manager.open(state)
    assert (workspace / "internal").readlink() == Path(WORKSPACE_FILENAME)
    assert (workspace / "internal").read_text(encoding=TEXT_ENCODING) == INITIAL_SOLUTION
    assert (workspace / "external").readlink() == external
    with ZipFile(seed_directory / SESSION_FILENAME) as archive:
        assert not any(name.startswith(SDK_DEBUG_DIRECTORY.as_posix()) for name in archive.namelist())
        assert REVISED_SOLUTION.encode(TEXT_ENCODING) not in [archive.read(name) for name in archive.namelist()]
    assert external.read_text(encoding=TEXT_ENCODING) == REVISED_SOLUTION


def test_archive_rejects_entries_beneath_links_before_restore(
    manager: SessionManager, state: RefinementState, seed_directory: Path, tmp_path: Path,
) -> None:
    """A malicious archive cannot extract a child through a link into external storage."""
    manager.open(state)
    workspace = manager.runtime_directory / WORKSPACE_DIRECTORY
    (workspace / "external").symlink_to(tmp_path)
    manager.checkpoint()
    with ZipFile(seed_directory / SESSION_FILENAME, "a") as archive:
        archive.writestr("workspace/external/escaped", INITIAL_SOLUTION)
    with pytest.raises(ValueError, match="unsafe session archive"):
        manager.restore()
    assert not (tmp_path / "escaped").exists()
    assert (workspace / "external").is_symlink()


def test_second_process_cannot_acquire_live_seed_lock(manager: SessionManager, state: RefinementState, seed_directory: Path) -> None:
    """The lock excludes a different Python process, not just another coroutine."""
    manager.open(state)
    command = [sys.executable, "-c", LOCK_PROBE, str(seed_directory / LOCK_FILENAME)]
    held = subprocess.run(command, timeout=ASYNC_TEST_TIMEOUT_SECONDS, check=False)
    assert held.returncode == LOCK_HELD_EXIT_CODE
    manager.close()
    released = subprocess.run(command, timeout=ASYNC_TEST_TIMEOUT_SECONDS, check=False)
    assert released.returncode == ZERO_TOKENS
    assert (seed_directory / LOCK_FILENAME).exists()


async def test_killed_process_releases_lock_without_deleting_file(manager: SessionManager, state: RefinementState, seed_directory: Path) -> None:
    """A forced process death releases ownership even though .lock remains on disk."""
    seed_directory.mkdir(parents=True)
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        process = await create_subprocess_exec(
            sys.executable,
            "-c",
            LOCK_HOLDER,
            str(seed_directory / LOCK_FILENAME),
            stdin=PIPE,
            stdout=PIPE,
        )
        try:
            assert process.stdout is not None
            await process.stdout.readline()
            with pytest.raises(RuntimeError, match="already running"):
                manager.open(state)
        finally:
            process.kill()
            await process.wait()
        manager.open(state)
    assert (seed_directory / LOCK_FILENAME).exists()


def test_corrupt_checkpoint_fails_without_overwriting_it(manager: SessionManager, state: RefinementState, seed_directory: Path) -> None:
    """Corrupt saved state must never become a new paid run."""
    manager.open(state)
    manager.close()
    with ZipFile(seed_directory / SESSION_FILENAME, "w") as archive:
        archive.writestr(CHECKPOINT_FILENAME, "{}")
    original = (seed_directory / SESSION_FILENAME).read_bytes()
    with pytest.raises(KeyError):
        manager.open(state)
    assert (seed_directory / SESSION_FILENAME).read_bytes() == original


def test_restore_rolls_back_native_files_workspace_and_usage(saved_session: SessionManager) -> None:
    """Only the completed phase's files, usage, and replay identities survive restoration."""
    state = saved_session.session_state.refinement
    workspace = saved_session.runtime_directory / WORKSPACE_DIRECTORY
    assert saved_session.resume_target is not None
    transcript = Path(saved_session.resume_target)
    transcript.write_text(REVISED_SOLUTION, encoding=TEXT_ENCODING)
    (workspace / WORKSPACE_FILENAME).write_text(REVISED_SOLUTION, encoding=TEXT_ENCODING)
    (workspace / FAILED_PHASE_FILENAME).write_text(REVISED_SOLUTION, encoding=TEXT_ENCODING)
    state.output_tokens += PHASE_TOKENS
    state.warning_sent = True
    saved_session.session_state.completed_message_ids.add(REVISED_SOLUTION)
    restored = saved_session.restore()
    assert restored.output_tokens == PHASE_TOKENS
    assert restored.solution == INITIAL_SOLUTION
    assert restored.phase == Phase.CRITIQUE
    assert not restored.warning_sent
    assert transcript.read_text(encoding=TEXT_ENCODING) == INITIAL_SOLUTION
    assert (workspace / WORKSPACE_FILENAME).read_text(encoding=TEXT_ENCODING) == INITIAL_SOLUTION
    assert not (workspace / FAILED_PHASE_FILENAME).exists()
    assert saved_session.session_state.completed_message_ids == {SDK_MESSAGE_ID}


def test_new_owner_restores_native_identity(saved_session: SessionManager, seed_directory: Path, state: RefinementState) -> None:
    """The compressed archive alone restores the conversation after its previous owner exits."""
    saved_session.close()
    assert not (seed_directory / RUNTIME_DIRECTORY).exists()
    owner = SessionManager(seed_directory, RunLock)
    try:
        restored = owner.open(state)
        assert restored.phase == Phase.CRITIQUE
        assert restored.solution == INITIAL_SOLUTION
        assert owner.resume_target is not None
        assert Path(owner.resume_target).stem == NATIVE_SESSION_ID
    finally:
        owner.close()


@pytest.mark.parametrize("status", (RunStatus.PAUSED, RunStatus.FINISHED, RunStatus.EXHAUSTED))
def test_explicit_continuation_preserves_history_and_extends_budget(
    saved_session: SessionManager, status: RunStatus, seed_directory: Path,
) -> None:
    """Continuation retains the saved solution and identity, clears pause, and renews the warning allowance."""
    state = saved_session.session_state.refinement
    state.status = status
    state.rounds = state.no_gap_critiques = TWO_ROUNDS
    state.pause_at_tokens = PHASE_TOKENS
    state.warning_sent = True
    saved_session.checkpoint()
    saved_session.close()
    saved_session.open(state)
    if status != RunStatus.PAUSED:
        assert not (seed_directory / RUNTIME_DIRECTORY).exists()
    continued = saved_session.continue_run(BUDGET_TOKENS + PHASE_TOKENS)
    assert continued.status == RunStatus.RUNNING
    assert continued.budget_tokens == BUDGET_TOKENS + PHASE_TOKENS
    assert continued.pause_at_tokens is None
    assert not continued.warning_sent
    assert continued.output_tokens == PHASE_TOKENS
    assert continued.solution == INITIAL_SOLUTION
    assert continued.rounds == TWO_ROUNDS
    assert continued.no_gap_critiques == ZERO_TOKENS
    assert saved_session.session_state.session_id == NATIVE_SESSION_ID
    assert not saved_session.session_state.fork_session
    assert saved_session.restore() == continued


def test_fork_copies_checkpoint_and_records_native_fork_intent(
    saved_session: SessionManager, seed_directory: Path,
) -> None:
    """Fork from durable history, without copying live mutations or changing the source archive."""
    original = (seed_directory / SESSION_FILENAME).read_bytes()
    saved_session.session_state.refinement.solution = REVISED_SOLUTION
    destination = seed_directory.parent / "branch"
    target = SessionManager(destination, RunLock)
    forked = saved_session.fork(target, None)
    try:
        restored = target.open(forked)
        assert restored.solution == INITIAL_SOLUTION
        assert target.session_state.fork_session
        assert target.session_state.session_id == NATIVE_SESSION_ID
        assert target.session_state.completed_message_ids == {SDK_MESSAGE_ID}
        assert target.resume_target is not None
        assert Path(target.resume_target).is_relative_to(destination)
    finally:
        target.close()
    assert (seed_directory / SESSION_FILENAME).read_bytes() == original


@pytest.mark.parametrize("crash_at", ("before_archive", "before_solution"))
def test_fork_recovers_process_crashes_around_publication(
    saved_session: SessionManager, seed_directory: Path, crash_at: str,
) -> None:
    """Abrupt exit leaves either a retryable destination or a fully prepared continuation archive."""
    state = saved_session.session_state.refinement
    state.status = RunStatus.PAUSED
    state.rounds = state.no_gap_critiques = TWO_ROUNDS
    saved_session.checkpoint()
    saved_session.close()
    original = (seed_directory / SESSION_FILENAME).read_bytes()
    destination = seed_directory.parent / "branch"
    crashed = subprocess.run(
        [sys.executable, "-c", FORK_CRASH_PROBE, str(seed_directory), str(destination), crash_at],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        timeout=ASYNC_TEST_TIMEOUT_SECONDS, capture_output=True, check=False,
    )
    assert crashed.returncode == FATAL_PROCESS_EXIT_CODE, crashed.stderr.decode()
    assert not crashed.stderr
    assert not (destination / RUNTIME_DIRECTORY).exists()
    assert not (destination / FIRST_SOLUTION_FILENAME).exists()
    target = SessionManager(destination, RunLock)
    if crash_at == "before_archive":
        assert not (destination / SESSION_FILENAME).exists()
        saved_session.open(state)
        saved_session.fork(target, BUDGET_TOKENS)
    else:
        assert (destination / SESSION_FILENAME).exists()
    try:
        restored = target.open(RefinementState(PROBLEM, BUDGET_TOKENS, None))
        assert restored.phase == Phase.CONTINUE
        assert restored.status == RunStatus.RUNNING
        assert restored.pause_at_tokens is None
        assert restored.output_tokens == PHASE_TOKENS
        assert restored.solution == INITIAL_SOLUTION
        assert restored.rounds == TWO_ROUNDS
        assert restored.no_gap_critiques == ZERO_TOKENS
        assert target.session_state.fork_session
        assert target.session_state.session_id == NATIVE_SESSION_ID
        assert target.resume_target is not None
        assert (destination / FIRST_SOLUTION_FILENAME).read_text() == INITIAL_SOLUTION
    finally:
        target.close()
    assert (seed_directory / SESSION_FILENAME).read_bytes() == original


def test_failed_archive_write_preserves_previous_checkpoint(saved_session: SessionManager, seed_directory: Path) -> None:
    """An unsuccessful atomic replacement leaves the committed solution and archive recoverable."""
    original = (seed_directory / SESSION_FILENAME).read_bytes()
    saved_session.session_state.refinement.solution = REVISED_SOLUTION
    with patch("harness.session.session_manager.os.replace", side_effect=OSError("write failed")):
        with pytest.raises(OSError, match="write failed"):
            saved_session.checkpoint()
    assert (seed_directory / SESSION_FILENAME).read_bytes() == original
    assert (seed_directory / FIRST_SOLUTION_FILENAME).read_text() == INITIAL_SOLUTION
    assert not list(seed_directory.glob("*.tmp"))
    assert saved_session.restore().solution == INITIAL_SOLUTION


def test_archive_recovers_solution_export_failure(saved_session: SessionManager, seed_directory: Path) -> None:
    """The committed archive regenerates solution_1x.md even when the previous export failed."""
    saved_session.session_state.refinement.status = RunStatus.FINISHED
    with patch.object(saved_session, "_write_solutions", side_effect=OSError("export failed")):
        with pytest.raises(OSError, match="export failed"):
            saved_session.checkpoint()
    restored = saved_session.restore()
    assert restored.status == RunStatus.FINISHED
    assert (seed_directory / FIRST_SOLUTION_FILENAME).read_text() == INITIAL_SOLUTION


def test_save_uses_owned_state_after_restore(saved_session: SessionManager) -> None:
    """Stale references to discarded live state cannot alter later checkpoints."""
    previous = saved_session.session_state.refinement
    restored = saved_session.restore()
    previous.solution = PROBLEM
    restored.solution = REVISED_SOLUTION
    saved_session.checkpoint()
    assert saved_session.restore().solution == REVISED_SOLUTION


def test_missing_transcript_does_not_replace_checkpoint(saved_session: SessionManager, seed_directory: Path) -> None:
    """Losing native history fails instead of silently starting a new conversation."""
    original = (seed_directory / SESSION_FILENAME).read_bytes()
    assert saved_session.resume_target is not None
    Path(saved_session.resume_target).unlink()
    with pytest.raises(FileNotFoundError, match="transcript"):
        saved_session.checkpoint()
    assert (seed_directory / SESSION_FILENAME).read_bytes() == original


def test_read_only_directories_restore_fork_and_clean_up(
    manager: SessionManager, state: RefinementState, tmp_path: Path,
) -> None:
    """Restore descendants before directory modes and remove owned directories without touching external links."""
    manager.open(state)
    runtime = manager.runtime_directory
    readonly = runtime / WORKSPACE_DIRECTORY / "readonly"
    child = readonly / "child"
    child.mkdir(parents=True)
    (child / "proof.txt").write_text(INITIAL_SOLUTION)
    external = tmp_path / "outside"
    external.mkdir(mode=OWNER_DIRECTORY_MODE)
    (readonly / "external").symlink_to(external, target_is_directory=True)
    child.chmod(READ_ONLY_DIRECTORY_MODE)
    readonly.chmod(READ_ONLY_DIRECTORY_MODE)
    manager.checkpoint()
    manager.close()
    assert not runtime.exists()
    manager.open(state)
    assert (child / "proof.txt").read_text() == INITIAL_SOLUTION
    assert stat.S_IMODE(child.stat().st_mode) == READ_ONLY_DIRECTORY_MODE
    assert stat.S_IMODE(readonly.stat().st_mode) == READ_ONLY_DIRECTORY_MODE
    manager.restore()
    manager.fork(SessionManager(tmp_path / "fork", RunLock), None)
    manager.close()
    assert not runtime.exists()
    assert stat.S_IMODE(external.stat().st_mode) == OWNER_DIRECTORY_MODE


def test_special_files_fail_before_opening_and_preserve_checkpoint(
    manager: SessionManager, state: RefinementState, seed_directory: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FIFO must be rejected before ZipFile tries a blocking open, leaving the durable checkpoint intact."""
    manager.open(state)
    original = (seed_directory / SESSION_FILENAME).read_bytes()
    pipe = manager.runtime_directory / WORKSPACE_DIRECTORY / "pipe"
    os.mkfifo(pipe)
    write = ZipFile.write

    def require_regular_file(archive: ZipFile, path: Path, relative: Path) -> None:
        """Fail immediately if the archiver attempts to open the FIFO instead of rejecting its type."""
        assert path != pipe, "would block opening a FIFO"
        write(archive, path, relative)

    monkeypatch.setattr(ZipFile, "write", require_regular_file)
    with pytest.raises(ValueError, match="unsupported session file type"):
        manager.checkpoint()
    assert (seed_directory / SESSION_FILENAME).read_bytes() == original
    manager.close()
