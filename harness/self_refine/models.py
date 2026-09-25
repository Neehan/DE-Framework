"""Configuration, resumable refinement progress, and session events."""

from dataclasses import dataclass, field
from enum import StrEnum, auto
from math import ceil

from harness.utils.constants import (
    COUNT_INCREMENT,
    INITIAL_COUNT,
    OUTPUT_TOKENS_PER_BLOCK,
    ZERO_TOKENS,
)


class Phase(StrEnum):
    """The instruction to execute next in the shared conversation."""

    SOLVE = auto()
    CRITIQUE = auto()
    REVISE = auto()
    CONTINUE = auto()


class RunStatus(StrEnum):
    """Execution status retained with refinement progress."""

    RUNNING = auto()
    FINISHED = auto()
    EXHAUSTED = auto()
    PAUSED = auto()


@dataclass(frozen=True)
class RefinementConfig:
    """Injected convergence settings; both conditions must hold to finish."""

    min_rounds: int
    min_no_gap_critiques: int

    def __post_init__(self) -> None:
        """Reject negative round floors and nonpositive critique thresholds."""
        if self.min_rounds < INITIAL_COUNT:
            raise ValueError("min_rounds must be nonnegative")
        if self.min_no_gap_critiques <= INITIAL_COUNT:
            raise ValueError("min_no_gap_critiques must be positive")


@dataclass
class RefinementState:
    """One attempt's problem, allowance, and progress for saving or resuming."""

    problem: str
    budget_tokens: int
    pause_at_tokens: int | None
    phase: Phase = field(init=False, default=Phase.SOLVE)
    output_tokens: int = field(init=False, default=ZERO_TOKENS)
    rounds: int = field(init=False, default=INITIAL_COUNT)
    no_gap_critiques: int = field(init=False, default=INITIAL_COUNT)
    warning_sent: bool = field(init=False, default=False)
    solution: str | None = field(init=False, default=None)
    status: RunStatus = field(init=False, default=RunStatus.RUNNING)
    solution_checkpoints: dict[int, str | None] = field(init=False, default_factory=dict)
    checkpoint_count: int = field(init=False)

    def __post_init__(self) -> None:
        """Allocate checkpoint labels through the stopping boundary, including a prefix pause."""
        self.checkpoint_count = ceil(self.active_limit / OUTPUT_TOKENS_PER_BLOCK)

    @property
    def active_limit(self) -> int:
        """Return the current stopping allowance; advertised budget can extend beyond a prefix."""
        return self.budget_tokens if self.pause_at_tokens is None else self.pause_at_tokens

    def validate(self) -> None:
        """Validate the task and accounting before spending any tokens."""
        if not self.problem.strip():
            raise ValueError("problem must not be empty")
        if self.budget_tokens <= ZERO_TOKENS:
            raise ValueError("budget_tokens must be positive")
        if self.output_tokens < ZERO_TOKENS:
            raise ValueError("output_tokens must be nonnegative")
        if self.pause_at_tokens is not None:
            if not ZERO_TOKENS < self.pause_at_tokens < self.budget_tokens:
                raise ValueError("pause_at_tokens must be inside the budget")
        if not INITIAL_COUNT <= self.no_gap_critiques <= self.rounds:
            raise ValueError("refinement counters are inconsistent")
        if self.checkpoint_count <= INITIAL_COUNT:
            raise ValueError("checkpoint_count must be positive")
        expected = set(range(COUNT_INCREMENT, len(self.solution_checkpoints) + COUNT_INCREMENT))
        if set(self.solution_checkpoints) != expected or len(expected) > self.checkpoint_count:
            raise ValueError("solution checkpoints must be contiguous within the allocation")
