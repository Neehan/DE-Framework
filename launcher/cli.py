"""Parse the deliberately small experiment-launching command line."""

import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable, Coroutine
from subprocess import CalledProcessError
from typing import Any, TypeVar

from dotenv import load_dotenv
from experiments.registry import EXPERIMENTS
from harness.sandbox.constants import COMMAND_ARGUMENT_OFFSET
from harness.utils.asyncio import run_process
from harness.utils.constants import (
    DEFAULT_COMPUTE_MULTIPLIER_K,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    MAX_COMPUTE_MULTIPLIER_K,
    MIN_COMPUTE_MULTIPLIER_K,
    OUTPUT_TOKENS_PER_BLOCK,
    TEXT_ENCODING,
)

from launcher.constants import DATASET_NAMES, DOTENV_FILE, FAILED_EXIT_CODE, LOG_FORMAT
from launcher.models import AuditConfig, LaunchConfig

Config = TypeVar("Config", bound=LaunchConfig)


def parse_arguments(arguments: list[str]) -> LaunchConfig:
    """Parse the run entrypoint using the shared selection arguments."""
    return _parse_arguments(_create_parser("run"), arguments, LaunchConfig)


def parse_audit_arguments(arguments: list[str]) -> AuditConfig:
    """Add only the judge model to the same solver-result selection arguments."""
    parser = _create_parser("audit")
    parser.add_argument("--audit-model", default=DEFAULT_JUDGE_MODEL, help="Independent judge model; GPT uses litellm/<model>")
    return _parse_arguments(parser, arguments, AuditConfig)


def _create_parser(command: str) -> ArgumentParser:
    """Define shared selectors once for the run and audit entrypoints."""
    parser = ArgumentParser(prog=f"python -m launcher.{command}", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter,
                            description="Run or audit local olympiad experiments in isolated Docker containers.")
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES, help="Local dataset subset")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS, help="Experiment to run")
    parser.add_argument("--model", required=True, help="Claude/Muse model ID or litellm/<model>; GPT uses LiteLLM")
    parser.add_argument("--seeds", required=True, type=int, nargs="+", help="Explicit seed IDs, e.g. 1 2 3")
    parser.add_argument("--compute-multiplier-k", type=int,
                        help=f"Unaided: {MIN_COMPUTE_MULTIPLIER_K}–{MAX_COMPUTE_MULTIPLIER_K}, default {DEFAULT_COMPUTE_MULTIPLIER_K}; other experiments use their fixed budget. Each unit is {OUTPUT_TOKENS_PER_BLOCK:,} output tokens")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY, help="Maximum active attempts")
    parser.add_argument("--problems", nargs="+", help="Explicit problem IDs; omit to select all")
    parser.add_argument("--domain", help="Restrict to one domain")
    return parser


def _parse_arguments(parser: ArgumentParser, arguments: list[str], config_type: type[Config]) -> Config:
    """Validate selected experiment budgets and reject unknown or invalid options before execution."""
    args = parser.parse_args(arguments)
    try:
        args.compute_multiplier_k = EXPERIMENTS[args.experiment].resolve_multiplier(args.compute_multiplier_k)
        return config_type(**vars(args))
    except ValueError as error:
        parser.error(str(error))


def run_cli(parse: Callable[[list[str]], Config], launch: Callable[[Config], Coroutine[Any, Any, int]]) -> None:
    """Share dotenv loading, process cancellation, and setup errors across both entrypoints."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    config = parse(sys.argv[COMMAND_ARGUMENT_OFFSET:])
    load_dotenv(DOTENV_FILE, override=False)
    try:
        code = run_process(launch(config))
    except CalledProcessError as error:
        logging.error("Docker setup failed: %s", error.stderr.decode(TEXT_ENCODING))
        code = FAILED_EXIT_CODE
    except (ValueError, KeyError, OSError) as error:
        logging.error("Launcher setup failed: %s", error)
        code = FAILED_EXIT_CODE
    raise SystemExit(code)
