"""Grouped subprocess doubles used by sandbox tests."""

from dataclasses import dataclass
from unittest.mock import MagicMock


@dataclass
class SandboxProcesses:
    """Docker inspect, create, execute, and remove subprocesses in invocation order."""

    inspection: MagicMock
    creation: MagicMock
    execution: MagicMock
    removal: MagicMock
