"""Unaided experiment and base for experiments that specialize its prompts."""

from harness.self_refine.models import RefinementConfig
from harness.self_refine.recovery import Recovery
from harness.self_refine.self_refine import SelfRefine
from harness.utils.constants import (
    DEFAULT_COMPUTE_MULTIPLIER_K,
    DEFAULT_MIN_ROUNDS,
    MAX_COMPUTE_MULTIPLIER_K,
    MIN_COMPUTE_MULTIPLIER_K,
)
from harness.utils.prompt_loader import load_prompt

from experiments.constants import SKETCH_PROMPT_FILE, UNAIDED


class Unaided(SelfRefine):
    """Run shared self-refinement with the problem and budget supplied in RefinementState.

    Subclasses declare their name, fixed multiplier, and selected sketch role, then override prompts as needed. Budget variants use the same class; no overrides are required.
    """

    name = UNAIDED
    compute_multiplier: int | None = None
    sketch_role: str | None = None
    min_rounds = DEFAULT_MIN_ROUNDS

    def __init__(self, recovery: Recovery, config: RefinementConfig, sketch: str | None) -> None:
        """Accept only the sketch permitted by this experiment, alongside shared execution settings."""
        super().__init__(recovery, config)
        if self.sketch_role is None and sketch is not None:
            raise ValueError(f"{self.name} does not accept a sketch")
        if self.sketch_role is not None and (sketch is None or not sketch.strip()):
            raise ValueError(f"{self.name} requires a {self.sketch_role} sketch")
        self._sketch = sketch

    def _append_sketch(self, prompt: str) -> str:
        """Separate the selected sketch clearly from the otherwise shared instruction."""
        assert self._sketch is not None
        return f"{prompt}\n\n{load_prompt(SKETCH_PROMPT_FILE, {'sketch': self._sketch})}"

    @classmethod
    def resolve_multiplier(cls, requested: int | None) -> int:
        """Use the experiment default and reject unsupported budget changes."""
        multiplier = requested if requested is not None else cls.compute_multiplier or DEFAULT_COMPUTE_MULTIPLIER_K
        if not MIN_COMPUTE_MULTIPLIER_K <= multiplier <= MAX_COMPUTE_MULTIPLIER_K:
            raise ValueError(f"compute multiplier must be {MIN_COMPUTE_MULTIPLIER_K}–{MAX_COMPUTE_MULTIPLIER_K}")
        if cls.compute_multiplier is not None and multiplier != cls.compute_multiplier:
            raise ValueError(f"{cls.name} requires {cls.compute_multiplier}x compute")
        return multiplier
