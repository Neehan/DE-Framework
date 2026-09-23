"""One-block continuation from a shared unaided prefix."""

from harness.self_refine.models import Phase, RefinementState
from harness.utils.constants import INITIAL_COUNT, OUTPUT_TOKENS_PER_BLOCK
from harness.utils.prompt_loader import load_prompt

from experiments.constants import (
    CONTINUATION_TOTAL_MULTIPLIER,
    CONTINUE,
    CONTINUE_PROMPT_FILE,
)
from experiments.unaided import Unaided


class ContinueUnaided(Unaided):
    """Handle the fork's first continuation phase, then reuse ordinary refinement; no overrides required."""

    name = CONTINUE
    compute_multiplier = CONTINUATION_TOTAL_MULTIPLIER
    min_rounds = INITIAL_COUNT

    def _get_phase_prompt(self, state: RefinementState) -> str:
        """Issue continuation once; supply the problem if rollback left no completed conversation."""
        if state.phase == Phase.CONTINUE:
            prompt = self._get_continue_prompt()
            return f"{state.problem}\n\n{prompt}" if state.output_tokens == INITIAL_COUNT else prompt
        return super()._get_phase_prompt(state)

    def _get_continue_prompt(self) -> str:
        """Give both branches the same additional allowance and ordinary continuation instruction."""
        return load_prompt(CONTINUE_PROMPT_FILE, {"budget_tokens": str(OUTPUT_TOKENS_PER_BLOCK)})
