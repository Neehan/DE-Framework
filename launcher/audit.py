"""Host entrypoint for correctness followed by step recognition on saved solver results."""

import asyncio
import os
from functools import partial

from harness.sandbox.constants import SUCCESS_EXIT_CODE
from harness.sandbox.docker import Docker

from launcher.cli import parse_audit_arguments, run_cli
from launcher.constants import DATASETS_DIRECTORY, FAILED_EXIT_CODE, RESULTS_DIRECTORY
from launcher.dataset import Dataset
from launcher.docker_environment import docker_environment
from launcher.launcher.audit_launcher import AuditLauncher
from launcher.models import AuditConfig
from launcher.provider import resolve_provider


async def _launch_batch(config: AuditConfig) -> int:
    """Load only local references, route the judge separately, and delegate the fixed audit sequence."""
    dataset = Dataset(DATASETS_DIRECTORY)
    problems = dataset.load(config.dataset, config.problems, config.domain, None)
    references = dataset.load_references(config.dataset, problems)
    route = resolve_provider(config.audit_model, os.environ)
    environment = partial(docker_environment, Docker(asyncio.create_subprocess_exec), route, True)
    result = await AuditLauncher(config, environment, RESULTS_DIRECTORY, references).run(problems, route.model)
    return FAILED_EXIT_CODE if result.failed else SUCCESS_EXIT_CODE


if __name__ == "__main__":
    run_cli(parse_audit_arguments, _launch_batch)
