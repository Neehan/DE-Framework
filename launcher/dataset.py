"""Host-only selection from local JSONL datasets."""

import json
from pathlib import Path
from typing import Any

from audit.models import AuditReference
from experiments.constants import REFERENCE_ROLE
from harness.utils.constants import COUNT_INCREMENT, TEXT_ENCODING

from launcher.constants import DATASET_NAMES
from launcher.models import Problem


class Dataset:
    """Load permitted problem fields and validate ID/domain filters before any attempt starts; no overrides required."""

    def __init__(self, directory: Path) -> None:
        """Inject the local dataset directory for production or tests."""
        self._directory = directory

    def load(self, name: str, problem_ids: list[str] | None, domain: str | None, sketch_role: str | None) -> list[Problem]:
        """Read one local subset and reject unknown selections or empty results."""
        if name not in DATASET_NAMES:
            raise ValueError(f"unknown dataset: {name}")
        problems = self._read_problems(self._directory / f"{name}.jsonl", sketch_role)
        if domain is not None:
            if domain not in {problem.domain for problem in problems}:
                raise ValueError(f"unknown domain: {domain}")
            problems = [problem for problem in problems if problem.domain == domain]
        if problem_ids is not None:
            by_id = {problem.problem_id: problem for problem in problems}
            unknown = set(problem_ids) - by_id.keys()
            if unknown:
                raise ValueError(f"unknown problem IDs after domain filter: {sorted(unknown)}")
            problems = [by_id[problem_id] for problem_id in problem_ids]
        if not problems:
            raise ValueError("dataset selection contains no problems")
        for problem in problems:
            if sketch_role is not None and (problem.sketch is None or not problem.sketch.strip()):
                raise ValueError(f"problem {problem.problem_id} requires a {sketch_role} sketch")
        return problems

    def _read_problems(self, path: Path, sketch_role: str | None) -> list[Problem]:
        """Project permitted fields on the host, excluding full solutions and steps."""
        problems = {}
        for row in self._read_rows(path):
            problem = Problem(row["problem_id"], row["statement"], row["domain"], self._select_sketch(row, sketch_role))
            if problem.problem_id in problems:
                raise ValueError(f"duplicate problem ID: {problem.problem_id}")
            problems[problem.problem_id] = problem
        return list(problems.values())

    def load_references(self, name: str, problems: list[Problem]) -> dict[str, AuditReference]:
        """Select the fixed reference proof and outline for both audits, regardless of the solver's sketch arm."""
        if name not in DATASET_NAMES:
            raise ValueError(f"unknown dataset: {name}")
        selected = {problem.problem_id: problem for problem in problems}
        references = {}
        for row in self._read_rows(self._directory / f"{name}.jsonl"):
            problem_id = row["problem_id"]
            if problem_id not in selected:
                continue
            if problem_id in references or row["statement"] != selected[problem_id].statement:
                raise ValueError(f"duplicate or mismatched reference: {problem_id}")
            reference, = [solution for solution in row["solutions"] if solution["role"] == REFERENCE_ROLE]
            references[problem_id] = AuditReference(reference["solution"], reference["steps"])
        if references.keys() != selected.keys():
            raise ValueError("missing selected audit references")
        return references

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        """Read only local JSONL; dataset access stays in the host launcher."""
        with path.open(encoding=TEXT_ENCODING) as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _select_sketch(self, row: dict[str, Any], role: str | None) -> str | None:
        """Select by dataset role without silently substituting another solution's sketch."""
        if role is None:
            return None
        sketches = [solution["sketch"] for solution in row["solutions"] if solution["role"] == role]
        if len(sketches) > COUNT_INCREMENT:
            raise ValueError(f"problem {row['problem_id']} has multiple {role} sketches")
        return sketches.pop() if sketches else None
