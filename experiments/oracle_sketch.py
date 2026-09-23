"""One-block oracle-sketch control."""

from experiments.constants import ORACLE_SKETCH, SINGLE_BLOCK_MULTIPLIER
from experiments.oracle_execution import OracleExecution


class OracleSketch(OracleExecution):
    """Keep oracle execution's prompt and cap compute at one block; no overrides required."""

    name = ORACLE_SKETCH
    compute_multiplier = SINGLE_BLOCK_MULTIPLIER
