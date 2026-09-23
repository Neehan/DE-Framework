"""Live session state, checkpoint serialization, and execution events."""

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto
from uuid import UUID

from harness.self_refine.models import Phase, RefinementState, RunStatus
from harness.utils.constants import ZERO_TOKENS


@dataclass
class SessionState:
    """Own live refinement progress and its native conversation ID; serialize them at phase boundaries."""

    refinement: RefinementState
    session_id: str | None
    completed_message_ids: set[str] = field(init=False, default_factory=set)
    fork_session: bool = field(init=False, default=False)

    def to_json(self) -> str:
        """Encode the checkpoint without serializing a live SDK client."""
        data = {
            "state": asdict(self.refinement),
            "session_id": self.session_id,
            "completed_message_ids": sorted(self.completed_message_ids),
            "fork_session": self.fork_session,
        }
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "SessionState":
        """Reconstruct state and reject invalid phases, statuses, IDs, or accounting."""
        data = json.loads(text)
        saved = data["state"]
        state = RefinementState(saved["problem"], saved["budget_tokens"], saved["pause_at_tokens"])
        state.phase = Phase(saved["phase"])
        state.output_tokens = saved["output_tokens"]
        state.rounds = saved["rounds"]
        state.no_gap_critiques = saved["no_gap_critiques"]
        state.warning_sent = saved["warning_sent"]
        state.solution = saved["solution"]
        state.checkpoint_count = saved["checkpoint_count"]
        state.solution_checkpoints = {int(key): value for key, value in saved["solution_checkpoints"].items()}
        state.status = RunStatus(saved["status"])
        state.validate()
        session_id = data["session_id"]
        if session_id is not None:
            UUID(session_id)
        session = cls(state, session_id)
        session.completed_message_ids = set(data["completed_message_ids"])
        session.fork_session = data["fork_session"]
        return session


class EventKind(StrEnum):
    """Usage progress or the terminal outcome of a session run."""

    USAGE = auto()
    COMPLETED = auto()
    INTERRUPTED = auto()


class ConnectionStatus(StrEnum):
    """The mutually exclusive lifecycle states of one SDK connection."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    CLOSING = auto()


@dataclass(frozen=True)
class SessionEvent:
    """Cumulative output usage for one AgentSession.run call, with a terminal reply."""

    kind: EventKind
    output_tokens: int
    text: str

    def __post_init__(self) -> None:
        """Reject invalid usage and completed responses without text."""
        if self.output_tokens < ZERO_TOKENS:
            raise ValueError("event output_tokens must be nonnegative")
        if self.kind == EventKind.USAGE and self.text:
            raise ValueError("usage events must not contain response text")
        if self.kind == EventKind.COMPLETED:
            if not self.text.strip():
                raise ValueError("a completed response must contain text")
