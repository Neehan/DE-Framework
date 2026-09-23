"""Exclusive attempt ownership and compressed phase-boundary persistence."""

import os
import shutil
import stat
from collections.abc import Callable
from contextlib import AbstractContextManager
from math import ceil
from pathlib import Path, PurePosixPath
from typing import IO
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from harness.self_refine.models import Phase, RefinementState, RunStatus
from harness.session.constants import (
    CHECKPOINT_FILENAME,
    RUNTIME_DIRECTORY,
    SDK_DEBUG_DIRECTORY,
    SDK_DIRECTORY,
    SESSION_FILENAME,
    SOLUTION_FILENAME_TEMPLATE,
    SOLUTION_GLOB,
    TRANSCRIPT_EXTENSION,
    WORKSPACE_DIRECTORY,
    ZIP_MODE_SHIFT,
)
from harness.session.models import SessionState
from harness.utils.constants import (
    INITIAL_COUNT,
    OUTPUT_TOKENS_PER_BLOCK,
    TEXT_ENCODING,
)
from harness.utils.storage import atomic_write, create_runtime, remove_runtime


class SessionManager:
    """Open, checkpoint, restore, and close one seed directory under injected ownership.

    open resumes ordinary work; continue_run explicitly reopens stopped work; fork copies a saved session to a fresh seed. Close the connection before storage operations. No overrides are required.
    """

    def __init__(self, directory: Path, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Inject RunLock for standalone use or nullcontext when a host launcher already owns this seed."""
        self._directory = directory.resolve()
        self._ownership = ownership(self._directory)
        self._session_state: SessionState | None = None

    @property
    def session_state(self) -> SessionState:
        """Return the owned checkpoint state while this seed is locked."""
        return self._require_open_state()

    @property
    def runtime_directory(self) -> Path:
        """Return the managed workspace and native-session root for a locked seed."""
        self._require_open_state()
        return self._directory / RUNTIME_DIRECTORY

    @property
    def resume_target(self) -> str | None:
        """Locate the saved transcript explicitly so moving or forking a workspace does not lose it."""
        transcript = self._require_transcript(self._require_open_state().session_id)
        return str(transcript) if transcript is not None else None

    def open(self, initial_state: RefinementState) -> RefinementState:
        """Lock this seed and restore progress or initialize its first checkpoint."""
        if self._session_state is not None:
            raise RuntimeError("session manager is already open")
        self._ownership.__enter__()
        try:
            initial_state.validate()
            if (self._directory / SESSION_FILENAME).exists():
                return self._restore_checkpoint(initial_state)
            self._require_new_attempt()
            session_state = SessionState(initial_state, None)
            self._session_state = session_state
            create_runtime(self._directory)
            self._write_checkpoint(session_state)
            self._write_solutions(initial_state)
            return initial_state
        except BaseException:
            self.close()
            raise

    def prepare_attempt(self, initial_state: RefinementState) -> RefinementState:
        """Check ownership and saved identity; repair stopped solution exports without opening native files."""
        initial_state.validate()
        with self._ownership:
            path = self._directory / SESSION_FILENAME
            if not path.exists():
                return initial_state
            with ZipFile(path) as archive:
                saved = self._read_checkpoint(archive).refinement
            self._validate_requested_state(saved, initial_state)
            if saved.status != RunStatus.RUNNING:
                self._write_solutions(saved)
            return saved

    def read_checkpoint(self) -> SessionState:
        """Read validated saved state under injected ownership without restoring or changing files."""
        with self._ownership, ZipFile(self._directory / SESSION_FILENAME) as archive:
            return self._read_checkpoint(archive)

    def checkpoint(self) -> None:
        """Persist owned progress and closed native files at a completed phase boundary."""
        session_state = self._require_open_state()
        session_state.refinement.validate()
        self._require_transcript(session_state.session_id)
        self._write_checkpoint(session_state)
        self._write_solutions(session_state.refinement)

    def restore(self) -> RefinementState:
        """Roll back runtime files and accounting to the last durable phase boundary."""
        self._require_open_state()
        return self._restore_checkpoint(None)

    def continue_run(self, budget_tokens: int) -> RefinementState:
        """Commit continuation with a total allowance; close this owner before passing the result to SelfRefine.run."""
        self._require_open_state()
        with ZipFile(self._directory / SESSION_FILENAME) as archive:
            saved = self._read_checkpoint(archive)
            state = saved.refinement
            self._continue_state(state, budget_tokens)
            self._restore_runtime(archive)
            self._require_transcript(saved.session_id)
        self._session_state = saved
        self.checkpoint()
        return state

    def fork(self, target: "SessionManager", continuation_budget: int | None) -> RefinementState:
        """Atomically fork saved history, optionally preparing its first continuation phase before publication."""
        self._require_open_state()
        with target._ownership:
            target._require_new_attempt()
            with ZipFile(self._directory / SESSION_FILENAME) as archive:
                saved = self._read_checkpoint(archive)
                if continuation_budget is not None:
                    self._continue_state(saved.refinement, continuation_budget)
                    saved.refinement.phase = Phase.CONTINUE
                saved.fork_session = saved.session_id is not None
                atomic_write(target._directory / SESSION_FILENAME, lambda handle: self._copy_archive(archive, handle, saved))
            target._write_solutions(saved.refinement)
            return saved.refinement

    def _copy_archive(self, source: ZipFile, handle: IO[bytes], saved: SessionState) -> None:
        """Publish a fork without extracting runtime files that could strand an interrupted destination."""
        with ZipFile(handle, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(CHECKPOINT_FILENAME, saved.to_json())
            for entry in source.infolist():
                if entry.filename != CHECKPOINT_FILENAME:
                    with source.open(entry) as reader, archive.open(entry, "w") as writer:
                        shutil.copyfileobj(reader, writer)

    def _continue_state(self, state: RefinementState, budget_tokens: int) -> None:
        """Reopen stopped progress with fresh critiques, preserving total rounds and the last solution."""
        if state.status == RunStatus.RUNNING:
            raise ValueError("only stopped checkpoints require explicit continuation")
        if budget_tokens <= state.output_tokens:
            raise ValueError("continuation budget must exceed spent tokens")
        if budget_tokens > state.budget_tokens:
            state.warning_sent = False
        state.checkpoint_count = len(state.solution_checkpoints) + ceil((budget_tokens - state.output_tokens) / OUTPUT_TOKENS_PER_BLOCK)
        state.budget_tokens = budget_tokens
        state.pause_at_tokens = None
        state.no_gap_critiques = INITIAL_COUNT
        state.status = RunStatus.RUNNING

    def close(self) -> None:
        """Remove disposable runtime and release the lock after the connection is closed."""
        try:
            if self._session_state is not None:
                remove_runtime(self._directory)
        finally:
            self._session_state = None
            self._ownership.__exit__(None, None, None)

    def _restore_checkpoint(self, expected: RefinementState | None) -> RefinementState:
        """Validate identity before changing files, then restore the authoritative checkpoint."""
        with ZipFile(self._directory / SESSION_FILENAME) as archive:
            session_state = self._read_checkpoint(archive)
            state = session_state.refinement
            if expected is not None:
                self._validate_requested_state(state, expected)
            if session_state.refinement.status not in (RunStatus.FINISHED, RunStatus.EXHAUSTED):
                self._restore_runtime(archive)
                self._require_transcript(session_state.session_id)
        self._session_state = session_state
        self._write_solutions(session_state.refinement)
        return session_state.refinement

    def _validate_requested_state(self, saved: RefinementState, expected: RefinementState) -> None:
        """Prevent an existing result directory from silently changing problem or budget."""
        if (saved.problem, saved.budget_tokens) != (expected.problem, expected.budget_tokens):
            raise ValueError("checkpoint problem or budget does not match the requested attempt")

    def _read_checkpoint(self, archive: ZipFile) -> SessionState:
        """Decode and validate saved progress without modifying the archive or runtime."""
        return SessionState.from_json(archive.read(CHECKPOINT_FILENAME).decode(TEXT_ENCODING))

    def _require_new_attempt(self) -> None:
        """Refuse to replace existing session, solution, or runtime data."""
        if (any((self._directory / name).exists() for name in (SESSION_FILENAME, RUNTIME_DIRECTORY))
                or any(self._directory.glob(SOLUTION_GLOB))):
            raise FileExistsError("attempt already contains session, solution, or runtime data")

    def _require_open_state(self) -> SessionState:
        """Reject persistence operations outside a locked, initialized attempt."""
        if self._session_state is None:
            raise RuntimeError("session manager is not open")
        return self._session_state

    def _require_transcript(self, session_id: str | None) -> Path | None:
        """Reject checkpoints whose native conversation files are missing."""
        if session_id is None:
            return None
        sdk_directory = self._directory / RUNTIME_DIRECTORY / SDK_DIRECTORY
        paths = [path for path in sdk_directory.rglob(f"{session_id}{TRANSCRIPT_EXTENSION}")
                 if path.is_file() and not path.is_symlink()]
        if not paths:
            raise FileNotFoundError("native session transcript is missing")
        transcript, = paths
        return transcript

    def _write_checkpoint(self, session_state: SessionState) -> None:
        """Replace the archive only after all state and runtime files are written."""
        atomic_write(self._directory / SESSION_FILENAME, lambda handle: self._write_archive(handle, session_state))

    def _write_archive(self, handle: IO[bytes], session_state: SessionState) -> None:
        """Compress the closed conversation and workspace with their refinement state."""
        with ZipFile(handle, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(CHECKPOINT_FILENAME, session_state.to_json())
            runtime = self._directory / RUNTIME_DIRECTORY
            for path in sorted(runtime.rglob("*")):
                relative = path.relative_to(runtime)
                if relative.is_relative_to(SDK_DEBUG_DIRECTORY):
                    continue
                if path.is_symlink():
                    self._archive_symlink(archive, path, runtime)
                elif stat.S_ISREG(path.lstat().st_mode) or path.is_dir():
                    archive.write(path, relative)
                else:
                    raise ValueError(f"unsupported session file type: {relative}")

    def _archive_symlink(self, archive: ZipFile, path: Path, runtime: Path) -> None:
        """Store link text without reading its target; make internal absolute links portable."""
        target = path.readlink()
        if path.parent == runtime:
            raise ValueError("session workspace and SDK roots must not be symlinks")
        if target.is_absolute() and target.is_relative_to(runtime):
            target = Path(os.path.relpath(target, path.parent))
        entry = ZipInfo(path.relative_to(runtime).as_posix())
        entry.external_attr = path.lstat().st_mode << ZIP_MODE_SHIFT
        archive.writestr(entry, str(target).encode(TEXT_ENCODING))

    def _restore_runtime(self, archive: ZipFile) -> None:
        """Replace only managed runtime files, rejecting unsafe archive entries first."""
        runtime = self._directory / RUNTIME_DIRECTORY
        entries = [entry for entry in archive.infolist() if entry.filename != CHECKPOINT_FILENAME]
        self._validate_archive_entries(entries)
        remove_runtime(self._directory)
        create_runtime(self._directory)
        for entry in entries:
            if not stat.S_ISLNK(entry.external_attr >> ZIP_MODE_SHIFT):
                path = Path(archive.extract(entry, runtime))
                if not entry.is_dir():
                    os.chmod(path, stat.S_IMODE(entry.external_attr >> ZIP_MODE_SHIFT))
        for entry in entries:
            if stat.S_ISLNK(entry.external_attr >> ZIP_MODE_SHIFT):
                path = runtime / entry.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(archive.read(entry).decode(TEXT_ENCODING))
        for entry in sorted(entries, key=lambda item: len(PurePosixPath(item.filename).parts), reverse=True):
            if entry.is_dir():
                os.chmod(runtime / entry.filename, stat.S_IMODE(entry.external_attr >> ZIP_MODE_SHIFT))

    def _validate_archive_entries(self, entries: list[ZipInfo]) -> None:
        """Reject traversal and writes beneath links before replacing any runtime files."""
        links = {PurePosixPath(entry.filename) for entry in entries if stat.S_ISLNK(entry.external_attr >> ZIP_MODE_SHIFT)}
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[INITIAL_COUNT] not in (WORKSPACE_DIRECTORY, SDK_DIRECTORY)
                    or any(parent in links for parent in path.parents)
                    or (path in links and path.parent == PurePosixPath())):
                raise ValueError("unsafe session archive entry")

    def _write_solutions(self, state: RefinementState) -> None:
        """Regenerate every captured budget solution from authoritative checkpoint state."""
        for multiplier, solution in state.solution_checkpoints.items():
            content = (solution or "").encode(TEXT_ENCODING)
            path = self._directory / SOLUTION_FILENAME_TEMPLATE.format(multiplier=multiplier)
            atomic_write(path, lambda handle: handle.write(content))
