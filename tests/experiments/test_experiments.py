"""Shared prompts, permitted initial sketches, and unrestricted late continuation."""

from collections.abc import Callable

import pytest
from experiments.alt_sketch import AltSketch
from experiments.constants import SKETCH_PROMPT_FILE
from experiments.continue_oracle import ContinueOracle
from experiments.continue_unaided import ContinueUnaided
from experiments.no_sketch import NoSketch
from experiments.oracle_execution import OracleExecution
from experiments.oracle_sketch import OracleSketch
from experiments.unaided import Unaided
from harness.self_refine.constants import GAPS_FOUND_VERDICT, NO_GAPS_VERDICT
from harness.self_refine.models import Phase, RefinementState, RunStatus
from harness.session.models import EventKind, SessionEvent
from harness.utils.constants import OUTPUT_TOKENS_PER_BLOCK
from harness.utils.prompt_loader import load_prompt

from tests.constants import (
    INITIAL_SOLUTION,
    ONE_ROUND,
    PHASE_TOKENS,
    PREFIX_UNITS,
    PROBLEM,
    REVISED_SOLUTION,
    TOTAL_UNITS,
)
from tests.launcher.constants import REFERENCE_SKETCH
from tests.support.fake_agent_session import FakeAgentSession


@pytest.mark.parametrize("experiment", [Unaided, NoSketch, OracleExecution, OracleSketch, AltSketch])
def test_initial_sketch_only_extends_shared_solve_prompt(
    make_experiment: Callable[[type[Unaided], str | None, list[list[SessionEvent]]], tuple[Unaided, FakeAgentSession]],
    state: RefinementState, experiment: type[Unaided],
) -> None:
    """Only the initial solve receives an explicitly separated sketch; later instructions are identical."""
    sketch = REFERENCE_SKETCH if experiment.sketch_role is not None else None
    selected, _ = make_experiment(experiment, sketch, [])
    control, _ = make_experiment(Unaided, None, [])
    expected = control._get_phase_prompt(state)
    if sketch is not None:
        expected += "\n\n" + load_prompt(SKETCH_PROMPT_FILE, {"sketch": sketch})
    assert selected._get_phase_prompt(state) == expected
    for phase in (Phase.CRITIQUE, Phase.REVISE):
        state.phase = phase
        assert selected._get_phase_prompt(state) == control._get_phase_prompt(state)
        assert REFERENCE_SKETCH not in selected._get_phase_prompt(state)


@pytest.mark.parametrize(("experiment", "sketch"), [
    (Unaided, REFERENCE_SKETCH), (NoSketch, REFERENCE_SKETCH), (ContinueUnaided, REFERENCE_SKETCH),
    (OracleExecution, None), (OracleSketch, None), (AltSketch, None), (ContinueOracle, "  "),
])
def test_experiment_rejects_missing_or_forbidden_sketch(
    make_experiment: Callable[[type[Unaided], str | None, list[list[SessionEvent]]], tuple[Unaided, FakeAgentSession]],
    experiment: type[Unaided], sketch: str | None,
) -> None:
    """Construction fails before model execution if the selected arm's input contract is violated."""
    with pytest.raises(ValueError, match="sketch"):
        make_experiment(experiment, sketch, [])


@pytest.mark.parametrize("experiment", [ContinueUnaided, ContinueOracle])
async def test_continuation_injects_once_then_refines_without_ten_round_floor(
    make_experiment: Callable[[type[Unaided], str | None, list[list[SessionEvent]]], tuple[Unaided, FakeAgentSession]],
    state: RefinementState, experiment: type[Unaided],
) -> None:
    """A previously short prefix may converge after normal critiques; the sketch occurs only in continuation."""
    replies = [INITIAL_SOLUTION, GAPS_FOUND_VERDICT, REVISED_SOLUTION, NO_GAPS_VERDICT, REVISED_SOLUTION, NO_GAPS_VERDICT]
    sketch = REFERENCE_SKETCH if experiment.sketch_role is not None else None
    selected, agent = make_experiment(experiment, sketch, [
        [SessionEvent(EventKind.COMPLETED, PHASE_TOKENS, text)] for text in replies
    ])
    control, _ = make_experiment(ContinueUnaided, None, [])
    state.phase = Phase.CONTINUE
    state.rounds = ONE_ROUND
    state.no_gap_critiques = ONE_ROUND
    expected = control._get_phase_prompt(state)
    if sketch is not None:
        expected += "\n\n" + load_prompt(SKETCH_PROMPT_FILE, {"sketch": sketch})
    result = await selected.run(state)
    assert result.status == RunStatus.FINISHED
    assert result.rounds < Unaided.min_rounds
    assert result.solution == REVISED_SOLUTION
    assert agent.prompts[0] == expected
    assert str(OUTPUT_TOKENS_PER_BLOCK) in agent.prompts[0]
    assert all(REFERENCE_SKETCH not in prompt and "additional output tokens" not in prompt for prompt in agent.prompts[1:])
    assert len(agent.prompts) == len(replies)


@pytest.mark.parametrize("experiment", [ContinueUnaided, ContinueOracle])
async def test_empty_prefix_continuation_receives_problem_and_permitted_sketch(
    make_experiment: Callable[[type[Unaided], str | None, list[list[SessionEvent]]], tuple[Unaided, FakeAgentSession]],
    experiment: type[Unaided],
) -> None:
    """A first-solve rollback leaves no transcript, so its one-block branch must receive the original task explicitly."""
    state = RefinementState(PROBLEM, OUTPUT_TOKENS_PER_BLOCK, None)
    state.phase = Phase.CONTINUE
    state.checkpoint_count = TOTAL_UNITS
    state.solution_checkpoints = {key: None for key in range(ONE_ROUND, PREFIX_UNITS + ONE_ROUND)}
    sketch = REFERENCE_SKETCH if experiment.sketch_role is not None else None
    selected, agent = make_experiment(experiment, sketch, [
        [SessionEvent(EventKind.COMPLETED, OUTPUT_TOKENS_PER_BLOCK, REVISED_SOLUTION)],
    ])
    result = await selected.run(state)
    assert result.status == RunStatus.EXHAUSTED
    assert result.solution_checkpoints[TOTAL_UNITS] == REVISED_SOLUTION
    assert agent.prompts[0].startswith(PROBLEM)
    assert str(OUTPUT_TOKENS_PER_BLOCK) in agent.prompts[0]
    assert (REFERENCE_SKETCH in agent.prompts[0]) == (sketch is not None)
