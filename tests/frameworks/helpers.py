"""Write disposable compiled audit banks for framework tests."""

import json
from pathlib import Path
from typing import Any


def save_bank(path: Path, records: list[dict[str, Any]]) -> None:
    """Write controlled inputs without invoking provider or Docker infrastructure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
