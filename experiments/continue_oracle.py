"""Oracle-sketch intervention on an independent continuation fork."""

from experiments.constants import CONTINUE_ORACLE, REFERENCE_ROLE
from experiments.continue_unaided import ContinueUnaided


class ContinueOracle(ContinueUnaided):
    """Append the reference sketch to the first continuation instruction; no overrides required."""

    name = CONTINUE_ORACLE
    sketch_role = REFERENCE_ROLE

    def _get_continue_prompt(self) -> str:
        """Use the same continuation allowance while revealing the sketch only in this fork."""
        return self._append_sketch(super()._get_continue_prompt())
