"""Compile completed per-seed verdicts into one experiment's atomic JSONL export."""

import fcntl
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from harness.session.constants import LOCK_FILENAME
from harness.utils.constants import COUNT_INCREMENT, INITIAL_COUNT, TEXT_ENCODING
from harness.utils.storage import atomic_write

from audit.constants import AUDIT_FILENAME, COMPILED_AUDIT_FILENAME
from audit.storage import is_checkpoint_audited, read_audit


class AuditCompiler:
    """Compile all completed audits beneath an injected experiment directory; no overrides required.

    Per-seed records remain authoritative. The experiment lock serializes collection and publication.
    """

    def __init__(self, directory: Path) -> None:
        """Accept results/<dataset>/<encoded-model>/<experiment>-<k>x without CLI selection filters."""
        self._directory = directory

    def compile(self) -> Path:
        """Validate and publish all completed pairs, preserving the previous export on any read failure."""
        self._directory.mkdir(parents=True, exist_ok=True)
        destination = self._directory / COMPILED_AUDIT_FILENAME
        with (self._directory / LOCK_FILENAME).open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            records = self._collect_records()
            content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode(TEXT_ENCODING)
            atomic_write(destination, lambda handle: handle.write(content))
        return destination

    def _collect_records(self) -> list[dict[str, Any]]:
        """Attach experiment identity once and sort complete rows by problem and numeric seed."""
        experiment, multiplier = self._directory.name.rsplit("-", COUNT_INCREMENT)
        identity = {
            "dataset": self._directory.parent.parent.name,
            "solver_model": unquote(self._directory.parent.name),
            "experiment": experiment,
            "compute_multiplier_k": int(multiplier.removesuffix("x")),
        }
        records = []
        for path in self._directory.glob(f"*/seed_*/{AUDIT_FILENAME}"):
            record = read_audit(path.parent)
            seed = int(path.parent.name.removeprefix("seed_"))
            if seed < INITIAL_COUNT or path.parent.name != f"seed_{seed}":
                raise ValueError(f"invalid audit seed directory: {path.parent}")
            for label, checkpoint in record["checkpoints"].items():
                if is_checkpoint_audited(checkpoint):
                    records.append({**identity, "problem_id": path.parent.parent.name, "seed": seed,
                                    "checkpoint_multiplier_k": int(label.removesuffix("x")), **checkpoint})
        return sorted(records, key=lambda record: (record["problem_id"], record["seed"], record["checkpoint_multiplier_k"]))
