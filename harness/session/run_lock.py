"""Process ownership of a seed directory; Docker runs acquire this lock on the host."""

import fcntl
from pathlib import Path
from types import TracebackType
from typing import IO

from harness.session.constants import LOCK_FILENAME
from harness.session.errors import AttemptAlreadyRunning


class RunLock:
    """Hold .lock through a context manager; contention raises AttemptAlreadyRunning and process exit releases ownership."""

    def __init__(self, directory: Path) -> None:
        """Identify the seed without creating files until ownership is requested."""
        self._directory = directory
        self._handle: IO[str] | None = None

    def __enter__(self) -> None:
        """Acquire nonblocking process ownership before any seed files are read or changed."""
        if self._handle is not None:
            raise RuntimeError("run lock is already held")
        self._directory.mkdir(parents=True, exist_ok=True)
        handle = (self._directory / LOCK_FILENAME).open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise AttemptAlreadyRunning("attempt already running") from error
        except BaseException:
            handle.close()
            raise
        self._handle = handle

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        """Release the descriptor without removing the shared lock file."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
