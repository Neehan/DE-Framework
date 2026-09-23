"""Selected reference material and the narrow request passed to one audit worker."""

import json
from dataclasses import asdict, dataclass

from audit.constants import CORRECTNESS, OUTLINE_STEPS, STEP_RECOGNITION


@dataclass(frozen=True)
class AuditReference:
    """Keep one fixed reference proof and its three outline steps on the host."""

    solution: str
    steps: list[str]

    def __post_init__(self) -> None:
        """Reject missing reference proofs or outlines before any provider call."""
        if not self.solution.strip() or len(self.steps) != OUTLINE_STEPS or any(not step.strip() for step in self.steps):
            raise ValueError("audit requires a reference solution and exactly three nonempty outline steps")


@dataclass(frozen=True)
class AuditRequest:
    """Give each judge only its own instructions and selected mathematical content."""

    kind: str
    model: str
    problem: str
    reference: str
    solution: str
    steps: list[str] | None

    def __post_init__(self) -> None:
        """Require complete selected input while allowing an empty submitted solution to receive a verdict."""
        if self.kind not in (CORRECTNESS, STEP_RECOGNITION):
            raise ValueError("unknown audit kind")
        if not self.model.strip() or not self.problem.strip() or not self.reference.strip():
            raise ValueError("audit model, problem, and reference must not be empty")
        if self.kind == CORRECTNESS and self.steps is not None:
            raise ValueError("correctness audit must not receive outline steps")
        if self.kind == STEP_RECOGNITION:
            AuditReference(self.reference, self.steps or [])

    def to_json(self) -> str:
        """Encode selected audit input for stdin transport."""
        return json.dumps(asdict(self))


@dataclass(frozen=True)
class SeedAudit:
    """Pass authoritative checkpoint solutions and selected problem identity to the audit runner."""

    model: str
    problem_id: str
    problem: str
    solutions: dict[int, str | None]
