"""Compose experiment classes with shared in-memory session doubles."""

from collections.abc import Callable

import pytest
from experiments.unaided import Unaided
from harness.self_refine.models import RefinementConfig
from harness.session.models import SessionEvent
from harness.utils.constants import DEFAULT_MIN_NO_GAP_CRITIQUES

from tests.support.fake_agent_session import FakeAgentSession
from tests.support.fake_recovery import FakeRecovery


@pytest.fixture
def make_experiment() -> Callable[[type[Unaided], str | None, list[list[SessionEvent]]], tuple[Unaided, FakeAgentSession]]:
    """Use each experiment's production convergence settings and expose recorded phase prompts."""
    def create(
        experiment: type[Unaided], sketch: str | None, replies: list[list[SessionEvent]],
    ) -> tuple[Unaided, FakeAgentSession]:
        """Inject a scripted executor without opening native sessions or network connections."""
        agent = FakeAgentSession(replies)
        config = RefinementConfig(experiment.min_rounds, DEFAULT_MIN_NO_GAP_CRITIQUES)
        return experiment(FakeRecovery(agent), config, sketch), agent

    return create
