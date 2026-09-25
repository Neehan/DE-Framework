"""Validated launcher selections, permitted problem input, and batch outcomes."""

import json
from dataclasses import asdict, dataclass, field

from experiments.registry import EXPERIMENTS
from harness.proxy.models import ProviderConfig
from harness.self_refine.models import RefinementState
from harness.utils.constants import (
    INITIAL_COUNT,
    OUTPUT_TOKENS_PER_BLOCK,
)

from launcher.constants import (
    DATASET_NAMES,
    LITELLM_PREFIX,
    MODEL_DIRECTORY_ESCAPE,
    MODEL_NAME,
    SAFE_COMPONENT,
)


def validate_model(model: str) -> None:
    """Reject unsafe model identifiers before using them in result paths."""
    if not MODEL_NAME.fullmatch(model) or any(part in {"", ".", ".."} for part in model.split("/")):
        raise ValueError("invalid model identifier")


def encode_model_directory(model: str) -> str:
    """Use one reversible path encoding for experiments and framework outputs."""
    validate_model(model)
    return MODEL_DIRECTORY_ESCAPE.sub(lambda match: f"%{ord(match.group()):02X}", model)


@dataclass(frozen=True)
class Problem:
    """Retain selection fields and only the sketch permitted by the chosen experiment."""

    problem_id: str
    statement: str
    domain: str
    sketch: str | None

    def __post_init__(self) -> None:
        """Require safe result paths and nonempty problem content."""
        if not SAFE_COMPONENT.fullmatch(self.problem_id):
            raise ValueError(f"invalid problem id: {self.problem_id!r}")
        if not self.statement.strip() or not self.domain.strip():
            raise ValueError(f"problem {self.problem_id} requires a statement and domain")


@dataclass(frozen=True)
class LaunchConfig:
    """One CLI selection; compute budget and directory suffix from the same multiplier."""

    dataset: str
    experiment: str
    compute_multiplier_k: int
    model: str
    seeds: list[int]
    max_concurrency: int
    problems: list[str] | None
    domain: str | None

    def __post_init__(self) -> None:
        """Reject invalid selections before loading data or launching Docker."""
        if self.dataset not in DATASET_NAMES or self.experiment not in EXPERIMENTS:
            raise ValueError("unsupported dataset or experiment")
        EXPERIMENTS[self.experiment].resolve_multiplier(self.compute_multiplier_k)
        if self.max_concurrency <= INITIAL_COUNT:
            raise ValueError("max concurrency must be positive")
        if not self.seeds or min(self.seeds) < INITIAL_COUNT or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a nonempty list of unique nonnegative integers")
        validate_model(self.model)
        if self.problems is not None and (not self.problems or len(set(self.problems)) != len(self.problems)):
            raise ValueError("problem IDs must be nonempty and unique")

    @property
    def model_directory(self) -> str:
        """Encode separators and uppercase letters distinctly, including on case-insensitive filesystems."""
        return encode_model_directory(self.model)

    @property
    def budget_tokens(self) -> int:
        """Return the total output allowance shared across every refinement phase."""
        return self.compute_multiplier_k * OUTPUT_TOKENS_PER_BLOCK

    @property
    def experiment_directory(self) -> str:
        """Give each compute budget a separate results namespace."""
        return f"{self.experiment}-{self.compute_multiplier_k}x"


@dataclass(frozen=True)
class AuditConfig(LaunchConfig):
    """Select existing solver results and a separately routed judge model."""

    audit_model: str

    def __post_init__(self) -> None:
        """Validate shared selection fields and require an independent judge."""
        super().__post_init__()
        validate_model(self.audit_model)
        if self.audit_model.removeprefix(LITELLM_PREFIX) == self.model.removeprefix(LITELLM_PREFIX):
            raise ValueError("audit model must differ from the solver model")


@dataclass(frozen=True)
class ProviderRoute:
    """Keep the provider's model alias and real credentials exclusively on the host."""

    model: str
    providers: tuple[ProviderConfig, ...]
    use_anthropic_resets: bool


@dataclass(frozen=True)
class RunRequest:
    """Permit only the selected problem, sketch, and execution settings across the container boundary."""

    experiment: str
    model: str
    problem: str
    budget_tokens: int
    sketch: str | None
    pause_at_tokens: int | None

    @property
    def initial_state(self) -> RefinementState:
        """Construct fresh expected state; the session manager restores authoritative saved progress."""
        return RefinementState(self.problem, self.budget_tokens, self.pause_at_tokens)

    def to_json(self) -> str:
        """Serialize the explicitly permitted container input."""
        return json.dumps(asdict(self))


@dataclass
class BatchResult:
    """Count completed, skipped, and failed attempts for the terminal summary and exit status."""

    completed: int = field(init=False, default=INITIAL_COUNT)
    skipped: int = field(init=False, default=INITIAL_COUNT)
    failed: int = field(init=False, default=INITIAL_COUNT)
