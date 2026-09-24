"""Read and validate the shared per-seed audit record."""

import json
from pathlib import Path
from typing import Any

from harness.utils.constants import TEXT_ENCODING
from harness.utils.storage import atomic_write
from jsonschema import Draft202012Validator, ValidationError

from audit.constants import AUDIT_FIELDS, AUDIT_FILENAME, AUDIT_RECORD_SCHEMA


def read_audit(directory: Path) -> dict[str, Any]:
    """Return saved progress or an empty record; malformed verdicts fail with their source path."""
    path = directory / AUDIT_FILENAME
    if not path.exists():
        return {"checkpoints": {}}
    try:
        record = json.loads(path.read_text(encoding=TEXT_ENCODING))
        Draft202012Validator(AUDIT_RECORD_SCHEMA).validate(record)
    except (ValueError, ValidationError) as error:
        raise ValueError(f"invalid audit record: {path}") from error
    return record


def is_checkpoint_audited(record: dict[str, Any]) -> bool:
    """Require both stages in an already validated record, including deterministic empty-solution skips."""
    return all(field in record for field, _ in AUDIT_FIELDS.values())


def write_audit(directory: Path, record: dict[str, Any]) -> None:
    """Atomically publish validated checkpoint verdicts after each stage or reuse."""
    Draft202012Validator(AUDIT_RECORD_SCHEMA).validate(record)
    content = json.dumps(record, ensure_ascii=False).encode(TEXT_ENCODING)
    atomic_write(directory / AUDIT_FILENAME, lambda handle: handle.write(content))
