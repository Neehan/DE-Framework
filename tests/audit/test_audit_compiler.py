"""Whole-experiment exports preserve verdicts, resumable progress, and prior valid output."""

import fcntl
import json
from pathlib import Path

import pytest
from audit.audit_compiler import AuditCompiler
from audit.constants import COMPILED_AUDIT_FILENAME, EMPTY_AUDIT_RECORD
from harness.session.constants import LOCK_FILENAME

from tests.audit.constants import (
    AUDIT_MODEL,
    COMPILER_SEEDS,
    COMPLETE_AUDIT,
    CORRECTNESS_RESULT,
)
from tests.audit.helpers import save_audit


def test_compilation_collects_complete_pairs_and_sorts_numeric_seeds(experiment_directory: Path) -> None:
    """Keep all problems and seeds, exact judge identities, and empty skips; omit partial stages and other experiments."""
    for seed in reversed(COMPILER_SEEDS):
        save_audit(experiment_directory / "p1" / f"seed_{seed}", {"1x": COMPLETE_AUDIT, "4x": COMPLETE_AUDIT})
    save_audit(experiment_directory / "p2/seed_1", {"1x": EMPTY_AUDIT_RECORD})
    save_audit(experiment_directory / "p3/seed_1", {"1x": COMPLETE_AUDIT, "2x": {"correctness": CORRECTNESS_RESULT, "correctness_audit_model": AUDIT_MODEL}})
    save_audit(experiment_directory.parent / "unaided-1x/p1/seed_1", {"1x": COMPLETE_AUDIT, "4x": COMPLETE_AUDIT})
    compiler = AuditCompiler(experiment_directory)
    destination = compiler.compile()
    original = destination.read_bytes()
    records = [json.loads(line) for line in destination.read_text().splitlines()]
    identity = {"dataset": "aobench", "solver_model": "litellm/gpt-test", "experiment": "unaided", "compute_multiplier_k": 4}
    assert records == [
        *[{**identity, "problem_id": "p1", "seed": seed, "checkpoint_multiplier_k": multiplier, **COMPLETE_AUDIT}
          for seed in COMPILER_SEEDS for multiplier in (1, 4)],
        {**identity, "problem_id": "p2", "seed": 1, "checkpoint_multiplier_k": 1, **EMPTY_AUDIT_RECORD},
        {**identity, "problem_id": "p3", "seed": 1, "checkpoint_multiplier_k": 1, **COMPLETE_AUDIT},
    ]
    assert compiler.compile().read_bytes() == original


@pytest.mark.parametrize("malformed", ["json", "verdict", "seed"])
def test_invalid_source_preserves_previous_export(experiment_directory: Path, malformed: str) -> None:
    """Invalid saved data fails loudly instead of replacing the existing export with silently missing rows."""
    source = save_audit(experiment_directory / "p1/seed_1", {"1x": COMPLETE_AUDIT, "4x": COMPLETE_AUDIT})
    compiler = AuditCompiler(experiment_directory)
    destination = compiler.compile()
    original = destination.read_bytes()
    if malformed == "json":
        source.write_text("{")
    elif malformed == "verdict":
        save_audit(source.parent, {"correctness": {"score": "invalid"}})
    else:
        source.parent.rename(source.parent.with_name("seed_01"))
    with pytest.raises(ValueError):
        compiler.compile()
    assert destination.read_bytes() == original


def test_collection_and_publication_share_experiment_lock(experiment_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A concurrent compiler cannot publish a newer snapshot between this compiler's read and write."""
    compiler = AuditCompiler(experiment_directory)
    destination = experiment_directory / COMPILED_AUDIT_FILENAME

    def collect_under_lock() -> list[dict[str, object]]:
        """Probe from a distinct descriptor while checking that publication has not happened yet."""
        with (experiment_directory / LOCK_FILENAME).open("a+") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not destination.exists()
        return []

    monkeypatch.setattr(compiler, "_collect_records", collect_under_lock)
    assert compiler.compile().read_bytes() == b""
    with (experiment_directory / LOCK_FILENAME).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
