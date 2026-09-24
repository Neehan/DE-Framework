"""Load all eligible short and oracle observations, warning once about insufficient seeds."""

import json
import logging
from pathlib import Path

from audit.constants import COMPILED_AUDIT_FILENAME
from experiments.constants import ORACLE_EXECUTION, UNAIDED
from harness.utils.constants import TEXT_ENCODING
from jsonschema import Draft202012Validator
from launcher.dataset import Dataset
from launcher.models import Problem, encode_model_directory

from frameworks.models import InputSelection, Observation
from frameworks.utils.constants import MIN_SHORT_SEEDS, MIN_ORACLE_SEEDS, MIN_AUDIT_SCORE, SOURCE_AUDIT_SCHEMA, TOTAL_BUDGET


class DataLoader:
    """load returns observations for eligible selected problems using every complete audited seed."""

    def __init__(self, datasets: Dataset, results: Path) -> None:
        """Inject local dataset access and the results directory."""
        self._datasets = datasets
        self._results = results

    def load(self, selection: InputSelection) -> list[Observation]:
        """Skip insufficiently sampled problems and report their IDs in one warning across both datasets."""
        observations = []
        skipped = []
        for dataset, problems in self._select_problems(selection).items():
            directory = self._results / dataset / encode_model_directory(selection.model)
            ids = {problem.problem_id for problem in problems}
            short = self._read_runs(directory / f"{UNAIDED}-1x", ids, 1, selection.model)
            oracle = self._read_runs(directory / f"{ORACLE_EXECUTION}-{TOTAL_BUDGET}x", ids, TOTAL_BUDGET, selection.model)
            for problem in problems:
                short_runs, oracle_runs = short[problem.problem_id], oracle[problem.problem_id]
                if len(short_runs) < MIN_SHORT_SEEDS or len(oracle_runs) < MIN_ORACLE_SEEDS:
                    skipped.append(f"{dataset}/{problem.problem_id} (unaided={len(short_runs)}, oracle={len(oracle_runs)})")
                    continue
                observations.append(Observation(
                    dataset=dataset,
                    problem_id=problem.problem_id,
                    short_attempts=len(short_runs),
                    short_successes=sum(run[1] for run in short_runs),
                    oracle_attempts=len(oracle_runs),
                    oracle_successes=self._cumulative_successes(oracle_runs),
                ))
        if skipped:
            logging.getLogger(__name__).warning(
                "Skipped problems with insufficient audited seeds (need at least %d unaided-1x and %d complete oracle runs): %s",
                MIN_SHORT_SEEDS, MIN_ORACLE_SEEDS, ", ".join(skipped),
            )
        return observations

    def _select_problems(self, selection: InputSelection) -> dict[str, list[Problem]]:
        """Apply filters across the selected datasets so an ID need not occur in every dataset."""
        problems = {name: self._datasets.load(name, None, None, None) for name in selection.datasets}
        if selection.domain is not None:
            if not any(problem.domain == selection.domain for group in problems.values() for problem in group):
                raise ValueError(f"unknown domain: {selection.domain}")
            problems = {name: [problem for problem in group if problem.domain == selection.domain]
                        for name, group in problems.items()}
        if selection.problems is not None:
            unknown = set(selection.problems) - {problem.problem_id for group in problems.values() for problem in group}
            if unknown:
                raise ValueError(f"unknown problem IDs after domain filter: {sorted(unknown)}")
            problems = {name: [problem for problem in group if problem.problem_id in selection.problems]
                        for name, group in problems.items()}
        return {name: group for name, group in problems.items() if group}

    def _read_runs(self, directory: Path, problems: set[str], horizon: int, model: str) -> dict[str, list[dict[int, bool]]]:
        """Read every seed, rejecting corrupt rows and excluding seeds without all required checkpoints."""
        path = directory / COMPILED_AUDIT_FILENAME
        complete: dict[str, list[dict[int, bool]]] = {problem: [] for problem in problems}
        if not path.exists():
            return complete
        validator = Draft202012Validator(SOURCE_AUDIT_SCHEMA)
        runs: dict[str, dict[int, dict[int, bool]]] = {}
        with path.open(encoding=TEXT_ENCODING) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["problem_id"] not in problems:
                    continue
                validator.validate(row)
                if (row["dataset"] != directory.parent.parent.name or row["solver_model"] != model
                        or f"{row['experiment']}-{row['compute_multiplier_k']}x" != directory.name):
                    raise ValueError(f"audit identity does not match {path}")
                checkpoint = row["checkpoint_multiplier_k"]
                run = runs.setdefault(row["problem_id"], {}).setdefault(row["seed"], {})
                if checkpoint in run or checkpoint > horizon:
                    raise ValueError(f"duplicate or unexpected audit checkpoint in {path}")
                run[checkpoint] = row["correctness"]["score"] >= MIN_AUDIT_SCORE
        expected = set(range(1, horizon + 1))
        for problem, problem_runs in runs.items():
            for checkpoints in problem_runs.values():
                if checkpoints.keys() == expected:
                    complete[problem].append(checkpoints)
        return complete

    def _cumulative_successes(self, runs: list[dict[int, bool]]) -> list[int]:
        """Count runs solved by each checkpoint, retaining success even if a later revision fails."""
        counts = [0] * TOTAL_BUDGET
        for run in runs:
            solved = False
            for checkpoint in range(1, TOTAL_BUDGET + 1):
                solved = solved or run[checkpoint]
                counts[checkpoint - 1] += solved
        return counts
