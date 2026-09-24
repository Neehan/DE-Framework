"""Durable file publication and owned runtime cleanup shared by runs and audits."""

import os
import shutil
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO

from harness.session.constants import (
    RUNTIME_DIRECTORY,
    SDK_DIRECTORY,
    TEMP_FILE_SUFFIX,
    WORKSPACE_DIRECTORY,
)


def atomic_write(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Publish a complete file only after flushing its bytes, then sync the parent directory."""
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=TEMP_FILE_SUFFIX, delete=False) as handle:
            temporary = Path(handle.name)
            write(handle.file)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_runtime(directory: Path) -> Path:
    """Create the managed workspace and native SDK roots for one owned attempt."""
    runtime = directory / RUNTIME_DIRECTORY
    for name in (WORKSPACE_DIRECTORY, SDK_DIRECTORY):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    return runtime


def remove_runtime(directory: Path) -> None:
    """Remove only managed runtime, including read-only scratch, without following directory links."""
    remove_directory(directory / RUNTIME_DIRECTORY)


def remove_directory(directory: Path) -> None:
    """Clean an owned workspace after its container stops, without following directory links."""
    if directory.is_symlink():
        raise ValueError("managed directory must not be a symlink")
    if directory.exists():
        _make_directories_writable(directory)
        shutil.rmtree(directory)


def _make_directories_writable(directory: Path) -> None:
    """Permit owned directory cleanup without changing symlinks or their targets."""
    os.chmod(directory, stat.S_IMODE(directory.stat().st_mode) | stat.S_IRWXU)
    for path in directory.iterdir():
        if not path.is_symlink() and path.is_dir():
            _make_directories_writable(path)


@contextmanager
def fresh_runtime(directory: Path) -> Iterator[Path]:
    """Discard scratch from an interrupted audit before starting, and clean it after connection shutdown."""
    remove_runtime(directory)
    try:
        yield create_runtime(directory)
    finally:
        remove_runtime(directory)
