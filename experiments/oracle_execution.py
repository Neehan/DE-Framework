"""Initial oracle-sketch execution with the shared eight-block refinement loop."""

from harness.self_refine.models import RefinementState

from experiments.constants import (
    ORACLE_EXECUTION,
    ORACLE_EXECUTION_MULTIPLIER,
    REFERENCE_ROLE,
)
from experiments.unaided import Unaided


class OracleExecution(Unaided):
    """Extend only the initial solve instruction with the selected reference sketch; no overrides required."""

    name = ORACLE_EXECUTION
    compute_multiplier = ORACLE_EXECUTION_MULTIPLIER
    sketch_role = REFERENCE_ROLE

    def _get_solve_prompt(self, state: RefinementState) -> str:
        """Preserve the shared problem/budget prompt and append the permitted sketch."""
        return self._append_sketch(super()._get_solve_prompt(state))
