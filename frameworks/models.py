"""Input selections, observations, fitted parameters, and predictions for SG and DE."""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from frameworks.utils.constants import TOTAL_BUDGET
from launcher.constants import DATASET_NAMES
from launcher.models import validate_model


@dataclass(frozen=True)
class BaseParameters:
    """Base for fitted parameter dataclasses; subclasses declare their framework's fields."""


@dataclass(frozen=True)
class PriorFit:
    """Record one optimizer start; initialization holds positive parameters before log transformation, and nonfinite NLLs are recorded as None."""

    initialization: list[float]
    success: bool
    nll: float | None
    nll_gap: float | None
    reached_best: bool
    message: str


@dataclass(frozen=True)
class SGParameters(BaseParameters):
    """Store the short success and trial counts used by SG."""

    successes: int
    trials: int


@dataclass(frozen=True)
class BetaParameters(BaseParameters):
    """Store the positive Beta shapes a and b of a learned prior."""

    a: float
    b: float

    def __post_init__(self) -> None:
        """Require finite, positive Beta shapes before fitting or predicting."""
        if not isfinite(self.a) or not isfinite(self.b) or self.a <= 0 or self.b <= 0:
            raise ValueError("Beta shapes must be finite and positive")


@dataclass(frozen=True)
class RSGParameters(SGParameters):
    """Keep observed successes and trials together with prior shapes a and b for prediction and reporting."""

    a: float
    b: float


@dataclass(frozen=True)
class DirichletParameters(BaseParameters):
    """Store execution concentrations for blocks 1..8 and unsolved; zero marks an excluded category."""

    concentrations: list[float]

    def __post_init__(self) -> None:
        """Require finite nonnegative concentrations and at least two supported execution outcomes."""
        if len(self.concentrations) != TOTAL_BUDGET + 1:
            raise ValueError("execution requires one concentration per block and one for unsolved")
        if any(not isfinite(value) or value < 0 for value in self.concentrations):
            raise ValueError("Dirichlet concentrations must be finite and nonnegative")
        if sum(value > 0 for value in self.concentrations) < 2:
            raise ValueError("R-DE requires some first-block and some later or unsolved oracle outcomes")


@dataclass(frozen=True)
class RDEPrior(BaseParameters):
    """Hold the independent Beta discovery prior and Dirichlet execution prior."""

    discovery: BetaParameters
    execution: DirichletParameters

    def __post_init__(self) -> None:
        """Require first-block support so short successes can inform both discovery and execution."""
        if self.execution.concentrations[0] <= 0:
            raise ValueError("R-DE requires some first-block oracle completions")

@dataclass(frozen=True)
class RDEParameters(BaseParameters):
    """Retain the prior and short counts; oracle_counts contains first completions by block, then unsolved."""

    prior: RDEPrior
    successes: int
    trials: int
    oracle_counts: list[int]


@dataclass(frozen=True)
class RDEPosterior:
    """Rows represent problems; Beta shapes and case probabilities use columns for possible execution-failure counts. Later-execution concentrations use columns for completion outcomes."""

    log_evidence: NDArray[np.float64]
    case_probabilities: NDArray[np.float64]
    discovery_a: NDArray[np.float64]
    discovery_b: NDArray[np.float64]
    first_execution_a: NDArray[np.float64]
    first_execution_b: NDArray[np.float64]
    later_execution_concentrations: NDArray[np.float64]


@dataclass(frozen=True)
class DEParameters(BaseParameters):
    """Store discovery probability alpha, cumulative execution probabilities epsilon, and fit diagnostics."""

    alpha: float
    epsilon: list[float]
    at_boundary: bool
    is_unidentified: bool


@dataclass(frozen=True)
class BasePrediction:
    """Hold shared prediction fields; subclasses may add framework-specific outputs."""

    dataset: str
    problem_id: str
    n: int
    k: int
    success_probability: float


@dataclass(frozen=True)
class Observation:
    """Hold problem measurements; equality and hashing use only dataset and problem ID."""

    dataset: str
    problem_id: str
    short_attempts: int = field(compare=False)
    short_successes: int = field(compare=False)
    oracle_attempts: int = field(compare=False)
    # Cumulative successes by checkpoint, from 1x through TOTAL_BUDGET.
    oracle_successes: list[int] = field(compare=False)

    def __post_init__(self) -> None:
        """Require valid counts for both types of input data."""
        if not self.dataset or not self.problem_id:
            raise ValueError("observations require dataset and problem identity")

        counts = [self.short_attempts, self.short_successes, self.oracle_attempts, *self.oracle_successes]
        for count in counts:
            if type(count) is not int or count < 0:
                raise ValueError("observation counts must be nonnegative integers")

        if self.short_attempts < 1 or self.short_successes > self.short_attempts:
            raise ValueError("invalid short-success counts")

        if self.oracle_attempts < 1 or len(self.oracle_successes) != TOTAL_BUDGET:
            raise ValueError("oracle observations require attempts and one success count per checkpoint")
        if self.oracle_successes[-1] > self.oracle_attempts:
            raise ValueError("oracle successes cannot exceed attempts")
        for earlier, later in zip(self.oracle_successes, self.oracle_successes[1:]):
            if later < earlier:
                raise ValueError("oracle successes must be cumulative")


@dataclass(frozen=True)
class InputSelection:
    """Select a model and optional dataset, problem, and domain subsets."""

    model: str
    datasets: list[str]
    problems: list[str] | None
    domain: str | None

    def __post_init__(self) -> None:
        """Reject invalid model names and ambiguous filters before reading input data."""
        validate_model(self.model)

        if not self.datasets:
            raise ValueError("datasets must be supported and unique")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets must be supported and unique")
        for dataset in self.datasets:
            if dataset not in DATASET_NAMES:
                raise ValueError("datasets must be supported and unique")

        if self.problems is None:
            return
        if not self.problems or len(set(self.problems)) != len(self.problems):
            raise ValueError("problem IDs must be nonempty and unique")
