"""Shared solve, critique, and revise loop with resumable budget control."""

from contextlib import aclosing

from harness.self_refine.constants import (
    CRITIQUE_PROMPT_FILE,
    GAPS_FOUND_VERDICT,
    LAST_LINE_INDEX,
    NO_GAPS_VERDICT,
    REVISE_PROMPT_FILE,
    SOLVE_PROMPT_FILE,
    WARNING_PROMPT_FILE,
)
from harness.self_refine.models import (
    Phase,
    RefinementConfig,
    RefinementState,
    RunStatus,
)
from harness.self_refine.recovery import Recovery
from harness.session.models import EventKind, SessionEvent
from harness.utils.constants import (
    COUNT_INCREMENT,
    INITIAL_COUNT,
    OUTPUT_TOKENS_PER_BLOCK,
    WARNING_TOKENS,
)
from harness.utils.prompt_loader import load_prompt


class SelfRefine:
    """Run one prepared session; subclasses may override _get_solve_prompt for sketches.

    No overrides are required. Recovery coordinates connection lifetime and phase-boundary persistence.
    """

    def __init__(self, recovery: Recovery, config: RefinementConfig) -> None:
        """Inject session ownership and convergence settings."""
        self._recovery = recovery
        self._config = config
        self._running = False

    async def run(self, state: RefinementState) -> RefinementState:
        """Advance the supplied progress until convergence, exhaustion, or pause."""
        if self._running:
            raise RuntimeError("this self-refine controller is already running")
        self._running = True
        try:
            return await self._recovery.run(state, self._run_phases)
        finally:
            self._running = False

    async def _run_phases(self, state: RefinementState) -> RefinementState:
        """Skip completed attempts and checkpoint each phase before starting another."""
        state.validate()
        if state.status != RunStatus.RUNNING:
            return state
        while True:
            state.status = self._get_budget_status(state)
            if state.status == RunStatus.RUNNING:
                await self._warn_if_needed(state)
                state = await self._run_phase(state)
            through_tokens = state.output_tokens if state.status == RunStatus.RUNNING else state.active_limit
            self._capture_solution_checkpoints(state, through_tokens)
            await self._recovery.checkpoint()
            if state.status != RunStatus.RUNNING:
                return state

    async def _run_phase(self, state: RefinementState) -> RefinementState:
        """Consume one phase fully, including termination after an interrupt."""
        starting_tokens = state.output_tokens
        interrupt_requested = False
        terminal: SessionEvent | None = None
        async with aclosing(self._recovery.session.run(self._get_phase_prompt(state))) as events:
            async for event in events:
                if terminal is not None:
                    raise RuntimeError("session emitted an event after termination")
                self._update_token_usage(state, starting_tokens + event.output_tokens)
                if event.kind == EventKind.USAGE:
                    if (
                        not interrupt_requested
                        and self._get_budget_status(state) != RunStatus.RUNNING
                    ):
                        await self._recovery.session.interrupt()
                        interrupt_requested = True
                    await self._warn_if_needed(state)
                else:
                    if event.kind == EventKind.INTERRUPTED and not interrupt_requested:
                        raise RuntimeError("session interrupted without a request")
                    terminal = event
        if terminal is None:
            raise RuntimeError("session ended without a terminal event")
        return await self._resolve_phase_outcome(state, terminal)

    async def _resolve_phase_outcome(self, state: RefinementState, event: SessionEvent) -> RefinementState:
        """Restore an ineligible prefix phase before pausing; retain completed phases within their allowance."""
        if state.pause_at_tokens is not None and (
            state.output_tokens > state.pause_at_tokens
            or (event.kind == EventKind.INTERRUPTED and state.output_tokens >= state.pause_at_tokens)
        ):
            state = await self._recovery.restore()
            state.status = RunStatus.PAUSED
        elif event.kind == EventKind.INTERRUPTED or state.output_tokens > state.budget_tokens:
            state.status = self._get_budget_status(state)
        elif self._apply_phase_result(state, event.text):
            state.status = RunStatus.FINISHED
        else:
            state.status = self._get_budget_status(state)
        return state

    def _update_token_usage(self, state: RefinementState, total_tokens: int) -> None:
        """Record output and freeze crossed budgets, deferring exact boundaries until phase completion."""
        if total_tokens < state.output_tokens:
            raise ValueError("session usage must not decrease within a run")
        state.output_tokens = total_tokens
        self._capture_solution_checkpoints(state, total_tokens - COUNT_INCREMENT)

    def _capture_solution_checkpoints(self, state: RefinementState, through_tokens: int) -> None:
        """Freeze the last complete solution at crossed budgets; the final label uses the branch allowance."""
        for multiplier in range(len(state.solution_checkpoints) + COUNT_INCREMENT, state.checkpoint_count + COUNT_INCREMENT):
            boundary = state.active_limit if multiplier == state.checkpoint_count else multiplier * OUTPUT_TOKENS_PER_BLOCK
            if boundary > through_tokens:
                break
            state.solution_checkpoints[multiplier] = state.solution

    async def _warn_if_needed(self, state: RefinementState) -> None:
        """Queue one notice using the first observed low remaining allowance."""
        if (
            state.warning_sent
            or state.status != RunStatus.RUNNING
            or self._get_budget_status(state) != RunStatus.RUNNING
        ):
            return
        remaining_tokens = state.budget_tokens - state.output_tokens
        if remaining_tokens <= WARNING_TOKENS:
            await self._recovery.session.queue(
                load_prompt(WARNING_PROMPT_FILE, {"remaining_tokens": str(remaining_tokens)})
            )
            state.warning_sent = True

    def _get_budget_status(self, state: RefinementState) -> RunStatus:
        """Return exhaustion, pause, or permission to keep running."""
        if state.output_tokens >= state.budget_tokens:
            return RunStatus.EXHAUSTED
        if state.pause_at_tokens is not None:
            if state.output_tokens >= state.pause_at_tokens:
                return RunStatus.PAUSED
        return RunStatus.RUNNING

    def _get_phase_prompt(self, state: RefinementState) -> str:
        """Choose the shared instruction for the pending operation."""
        if state.phase == Phase.SOLVE:
            return self._get_solve_prompt(state)
        if state.phase == Phase.CRITIQUE:
            return load_prompt(CRITIQUE_PROMPT_FILE, {
                "no_gaps_verdict": NO_GAPS_VERDICT, "gaps_found_verdict": GAPS_FOUND_VERDICT,
            })
        return load_prompt(REVISE_PROMPT_FILE, {})

    def _get_solve_prompt(self, state: RefinementState) -> str:
        """State the task and full budget; sketch arms may extend this message."""
        return load_prompt(SOLVE_PROMPT_FILE, {
            "budget_tokens": str(state.budget_tokens), "problem": state.problem,
        })

    def _apply_phase_result(self, state: RefinementState, text: str) -> bool:
        """Update the solution or critique streak, advance the phase, and report convergence."""
        if state.phase == Phase.CRITIQUE:
            no_gaps = self._critique_has_no_gaps(text)
            state.rounds += COUNT_INCREMENT
            state.no_gap_critiques = (
                state.no_gap_critiques + COUNT_INCREMENT if no_gaps else INITIAL_COUNT
            )
            state.phase = Phase.REVISE
            return (
                state.rounds >= self._config.min_rounds
                and state.no_gap_critiques >= self._config.min_no_gap_critiques
            )
        state.solution = text
        state.phase = Phase.CRITIQUE
        return False

    def _critique_has_no_gaps(self, text: str) -> bool:
        """Count only an explicit final clean verdict; all other critiques require revision."""
        lines = text.strip().splitlines()
        return bool(lines) and lines[LAST_LINE_INDEX].strip() == NO_GAPS_VERDICT
