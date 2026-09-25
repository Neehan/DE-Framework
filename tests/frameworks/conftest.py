"""Reusable local input files for framework tests; no real results are accessed."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from frameworks.runner import FrameworkRunner
from frameworks.models import InputSelection
from frameworks.utils.constants import TOTAL_BUDGET
from frameworks.utils.dataloader import DataLoader
from launcher.dataset import Dataset
from launcher.frameworks import parse_arguments
from launcher.models import encode_model_directory

from tests.frameworks.constants import DATASET, SHORT_SEEDS, MODEL, ORACLE_SEEDS, ROWS
from tests.frameworks.helpers import save_bank


@pytest.fixture
def selection() -> InputSelection:
    """Limit most tests to one dataset; seed selection is intentionally absent."""
    return parse_arguments(["--dataset", DATASET, "--model", MODEL])


@pytest.fixture
def banks(tmp_path: Path) -> Path:
    """Include more than three oracle seeds and later proof regressions in both datasets."""
    dataset = tmp_path / "datasets"
    dataset.mkdir()
    for name in [DATASET, "imoproofbench"]:
        inputs = [replace(row, dataset=name, problem_id=row.problem_id if name == DATASET else f"imo_{row.problem_id}") for row in ROWS]
        problems = [{"problem_id": row.problem_id, "statement": row.problem_id, "domain": "algebra"} for row in inputs]
        (dataset / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in problems))
        root = tmp_path / "results" / name / encode_model_directory(MODEL)
        for experiment, horizon in [["unaided", 1], ["oracle-execution", TOTAL_BUDGET]]:
            records = []
            for row in inputs:
                completions = [
                    next((k for k, count in enumerate(row.oracle_successes, 1) if seed <= count), TOTAL_BUDGET + 1)
                    for seed in ORACLE_SEEDS
                ]
                seeds = SHORT_SEEDS if experiment == "unaided" else ORACLE_SEEDS
                for seed in seeds:
                    for k in range(1, horizon + 1):
                        passing = seed <= row.short_successes if experiment == "unaided" else k == completions[seed - 1]
                        records.append({"dataset": name, "solver_model": MODEL, "experiment": experiment,
                                        "compute_multiplier_k": horizon, "problem_id": row.problem_id, "seed": seed,
                                        "checkpoint_multiplier_k": k, "correctness": {"score": 7 if passing else 0, "note": "test"}})
            save_bank(root / f"{experiment}-{horizon}x/audit.jsonl", records)
    return tmp_path


@pytest.fixture
def data_loader(banks: Path) -> DataLoader:
    """Inject disposable dataset and result locations."""
    return DataLoader(Dataset(banks / "datasets"), banks / "results")


@pytest.fixture
def runner(data_loader: DataLoader, banks: Path) -> FrameworkRunner:
    """Use production fitting and output writing with synthetic observations."""
    return FrameworkRunner(data_loader, banks / "results")
