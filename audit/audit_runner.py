"""Judge budget checkpoints and carry saved verdicts across identical submissions."""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from harness.sandbox.sandbox import Sandbox
from harness.session.run_lock import RunLock
from harness.utils.storage import remove_directory

from audit.constants import (
    AUDIT_FIELDS,
    AUDIT_WORKSPACE_DIRECTORY,
    EMPTY_AUDIT_RECORD,
    STEP_RECOGNITION,
)
from audit.models import AuditReference, AuditRequest, SeedAudit
from audit.registry import AUDITS
from audit.storage import is_checkpoint_audited, read_audit, write_audit


class AuditRunner:
    """Prepare or run every checkpoint under the caller's seed lock; no overrides required.

    Inject references and judge identity once. Each missing stage gets its own sandbox.
    """

    def __init__(self, judge_model: str, references: dict[str, AuditReference]) -> None:
        """Retain host-only references and the identity recorded with new judgments."""
        self._judge_model = judge_model
        self._references = references

    def prepare(self, directory: Path, solutions: dict[int, str | None]) -> bool:
        """Publish empty or reused verdicts and report whether any judge work remains."""
        record = self._prepare_record(directory, solutions)
        if all(is_checkpoint_audited(record["checkpoints"][f"{key}x"]) for key in solutions):
            remove_directory(directory / AUDIT_WORKSPACE_DIRECTORY)
            return False
        return True

    async def run(self, directory: Path, seed: SeedAudit, sandbox_factory: Callable[[], Sandbox]) -> None:
        """Persist each completed stage before judging another, including nonconsecutive duplicate solutions."""
        record = self._prepare_record(directory, seed.solutions)
        for multiplier, solution in sorted(seed.solutions.items()):
            checkpoint = record["checkpoints"][f"{multiplier}x"]
            for kind, (verdict_field, model_field) in AUDIT_FIELDS.items():
                if verdict_field in checkpoint:
                    continue
                reference = self._references[seed.problem_id]
                request = AuditRequest(kind, seed.model, seed.problem, reference.solution, solution or "",
                                       reference.steps if kind == STEP_RECOGNITION else None)
                target = self._stage_directory(directory, multiplier, kind)
                await sandbox_factory().run(target, request.to_json(), RunLock)
                checkpoint[verdict_field] = AUDITS[kind].read_result(target)
                checkpoint[model_field] = self._judge_model
                self._reuse_saved_verdicts(record, seed.solutions)
                write_audit(directory, record)
                remove_directory(target)
        remove_directory(directory / AUDIT_WORKSPACE_DIRECTORY)

    def _prepare_record(self, directory: Path, solutions: dict[int, str | None]) -> dict[str, Any]:
        """Validate checkpoint labels and publish only newly available deterministic or reused stages."""
        record = read_audit(directory)
        if not solutions or set(record["checkpoints"]) - {f"{key}x" for key in solutions}:
            raise ValueError("audit checkpoints do not match saved solution checkpoints")
        previous = deepcopy(record)
        self._reuse_saved_verdicts(record, solutions)
        if record != previous:
            write_audit(directory, record)
        return record

    def _reuse_saved_verdicts(self, record: dict[str, Any], solutions: dict[int, str | None]) -> None:
        """Index exact solution text in memory and fill missing stages with their original judge identities."""
        checkpoints = record["checkpoints"]
        saved: dict[str, dict[str, Any]] = {}
        for multiplier, solution in sorted(solutions.items()):
            checkpoint = checkpoints.setdefault(f"{multiplier}x", {})
            if not (solution or "").strip():
                if checkpoint and checkpoint != EMPTY_AUDIT_RECORD:
                    raise ValueError("empty solution has a judge verdict")
                checkpoint.update(deepcopy(EMPTY_AUDIT_RECORD))
            cached = saved.setdefault(solution or "", {})
            self._copy_missing_verdicts(cached, checkpoint)
        for multiplier, solution in solutions.items():
            checkpoint = checkpoints[f"{multiplier}x"]
            self._copy_missing_verdicts(checkpoint, saved[solution or ""])

    def _copy_missing_verdicts(self, destination: dict[str, Any], source: dict[str, Any]) -> None:
        """Copy each missing verdict together with its original judge identity, preserving existing results."""
        for field, model_field in AUDIT_FIELDS.values():
            if field not in destination and field in source:
                destination.update({field: source[field], model_field: source[model_field]})

    def _stage_directory(self, directory: Path, multiplier: int, kind: str) -> Path:
        """Reject links at every disposable directory level before a sandbox resolves its mount."""
        workspace = directory / AUDIT_WORKSPACE_DIRECTORY
        checkpoint = workspace / f"{multiplier}x"
        target = checkpoint / kind
        if any(path.is_symlink() for path in (workspace, checkpoint, target)):
            raise ValueError("audit workspace must not be a symlink")
        return target
