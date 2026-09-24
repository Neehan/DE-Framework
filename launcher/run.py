"""Host entrypoint for solver experiments using shared launcher infrastructure."""

import asyncio
import os
from functools import partial

from experiments.continue_unaided import ContinueUnaided
from experiments.registry import EXPERIMENTS
from harness.sandbox.constants import SUCCESS_EXIT_CODE
from harness.sandbox.docker import Docker

from launcher.cli import parse_arguments, run_cli
from launcher.constants import DATASETS_DIRECTORY, FAILED_EXIT_CODE, RESULTS_DIRECTORY
from launcher.dataset import Dataset
from launcher.docker_environment import docker_environment
from launcher.launcher.continuation_launcher import ContinuationLauncher
from launcher.launcher.run_launcher import RunLauncher
from launcher.models import LaunchConfig
from launcher.provider import resolve_provider


async def _launch_batch(config: LaunchConfig) -> int:
    """Validate selected problems and credentials, then delegate scheduling and environment ownership."""
    experiment = EXPERIMENTS[config.experiment]
    problems = Dataset(DATASETS_DIRECTORY).load(config.dataset, config.problems, config.domain, experiment.sketch_role)
    route = resolve_provider(config.model, os.environ)
    environment = partial(docker_environment, Docker(asyncio.create_subprocess_exec), route, False)
    launcher = ContinuationLauncher if issubclass(experiment, ContinueUnaided) else RunLauncher
    result = await launcher(config, environment, RESULTS_DIRECTORY).run(problems, route.model)
    return FAILED_EXIT_CODE if result.failed else SUCCESS_EXIT_CODE


if __name__ == "__main__":
    run_cli(parse_arguments, _launch_batch)
