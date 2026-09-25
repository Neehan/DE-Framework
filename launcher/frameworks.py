"""Fit SG, DE, R-SG, and R-DE locally from all eligible audited seeds."""

import logging
import sys
from argparse import ArgumentParser

from frameworks.models import InputSelection
from frameworks.runner import FrameworkRunner
from frameworks.utils.dataloader import DataLoader
from harness.sandbox.constants import COMMAND_ARGUMENT_OFFSET
from jsonschema import ValidationError
from launcher.constants import DATASET_NAMES, DATASETS_DIRECTORY, FAILED_EXIT_CODE, LOG_FORMAT, RESULTS_DIRECTORY
from launcher.dataset import Dataset


def parse_arguments(arguments: list[str]) -> InputSelection:
    """Select input data; both datasets and all problems are included unless filtered."""
    parser = ArgumentParser(allow_abbrev=False, description="Fit SG, DE, R-SG, and R-DE using all eligible audited seeds.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", dest="datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--problems", nargs="+")
    parser.add_argument("--domain")
    try:
        return InputSelection(**vars(parser.parse_args(arguments)))
    except ValueError as error:
        parser.error(str(error))


def main() -> None:
    """Load selected observations and fit all frameworks without invoking a provider."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    selection = parse_arguments(sys.argv[COMMAND_ARGUMENT_OFFSET:])
    try:
        data_loader = DataLoader(Dataset(DATASETS_DIRECTORY), RESULTS_DIRECTORY)
        runner = FrameworkRunner(data_loader, RESULTS_DIRECTORY)
        for path in runner.run(selection):
            logging.info("Saved %s", path)
    except (ValueError, OSError, RuntimeError, ArithmeticError, ValidationError) as error:
        logging.error("Framework fitting failed: %s", error)
        raise SystemExit(FAILED_EXIT_CODE) from error


if __name__ == "__main__":
    main()
