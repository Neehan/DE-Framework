"""Exact-text reuse and recovery of partially audited budget checkpoints."""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from audit.audit_runner import AuditRunner
from audit.constants import (
    AUDIT_WORKSPACE_DIRECTORY,
    CORRECTNESS,
    RESULT_FILENAME,
    STEP_RECOGNITION,
)
from audit.models import AuditReference, AuditRequest, SeedAudit
from audit.storage import read_audit
from harness.sandbox.sandbox import Sandbox

from tests.audit.constants import (
    AUDIT_MODEL,
    CORRECTNESS_RESULT,
    REFERENCE,
    STEP_RESULT,
    STEPS,
    SUBMISSION,
)
from tests.audit.helpers import save_audit
from tests.constants import INITIAL_SOLUTION, PROBLEM


async def test_reuse_nonconsecutive_solutions_and_resume_partial_stages(tmp_path: Path) -> None:
    """Reuse A/B/A by exact text, retain old judge identities, and retry only the failed stage; whitespace changes get judged."""
    old_judge, first_judge, next_judge = AUDIT_MODEL, f"first/{AUDIT_MODEL}", f"next/{AUDIT_MODEL}"
    seed = SeedAudit(AUDIT_MODEL, "p1", PROBLEM, {1: SUBMISSION, 2: INITIAL_SOLUTION, 3: SUBMISSION, 4: SUBMISSION + "\n"})
    save_audit(tmp_path, {"3x": {"correctness": CORRECTNESS_RESULT, "correctness_audit_model": old_judge}})
    references = {seed.problem_id: AuditReference(REFERENCE, STEPS)}
    runner = AuditRunner(first_judge, references)
    seen: list[tuple[str, str]] = []
    sandbox = create_autospec(Sandbox, instance=True)

    async def execute(directory: Path, content: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Fail the first recognition of B after A has already been published at both labels."""
        request = AuditRequest(**json.loads(content))
        key = request.solution, request.kind
        if key == (INITIAL_SOLUTION, STEP_RECOGNITION) and key not in seen:
            seen.append(key)
            raise ConnectionError("recognition disconnected")
        seen.append(key)
        with ownership(directory):
            verdict = CORRECTNESS_RESULT if request.kind == CORRECTNESS else STEP_RESULT
            (directory / RESULT_FILENAME).write_text(json.dumps(verdict))

    sandbox.run.side_effect = execute
    assert runner.prepare(tmp_path, seed.solutions)
    with pytest.raises(ConnectionError):
        await runner.run(tmp_path, seed, lambda: sandbox)
    saved = read_audit(tmp_path)["checkpoints"]
    assert saved["1x"] == saved["3x"]
    assert saved["1x"]["correctness_audit_model"] == old_judge
    assert saved["1x"]["step_audit_model"] == first_judge
    assert "step_recognition" not in saved["2x"]
    resumed = AuditRunner(next_judge, references)
    await resumed.run(tmp_path, seed, lambda: sandbox)
    saved = read_audit(tmp_path)["checkpoints"]
    assert saved["2x"]["correctness_audit_model"] == first_judge
    assert saved["2x"]["step_audit_model"] == saved["4x"]["step_audit_model"] == next_judge
    assert not resumed.prepare(tmp_path, seed.solutions)
    assert not (tmp_path / AUDIT_WORKSPACE_DIRECTORY).exists()
    assert seen == [(SUBMISSION, STEP_RECOGNITION), (INITIAL_SOLUTION, CORRECTNESS),
                    (INITIAL_SOLUTION, STEP_RECOGNITION), (INITIAL_SOLUTION, STEP_RECOGNITION),
                    (SUBMISSION + "\n", CORRECTNESS), (SUBMISSION + "\n", STEP_RECOGNITION)]
