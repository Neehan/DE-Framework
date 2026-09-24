"""One-block control using the dataset's alternative sketch."""

from experiments.constants import ALT_SKETCH, ALTERNATE_ROLE
from experiments.oracle_sketch import OracleSketch


class AltSketch(OracleSketch):
    """Reuse the oracle-sketch prompt and budget with the alternate role; no overrides required."""

    name = ALT_SKETCH
    sketch_role = ALTERNATE_ROLE
