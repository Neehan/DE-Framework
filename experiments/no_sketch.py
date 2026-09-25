"""One-block control with an independent experiment identity."""

from experiments.constants import NO_SKETCH, SINGLE_BLOCK_MULTIPLIER
from experiments.unaided import Unaided


class NoSketch(Unaided):
    """Reuse unaided prompts and execution, capped at one block; no overrides required."""

    name = NO_SKETCH
    compute_multiplier = SINGLE_BLOCK_MULTIPLIER
