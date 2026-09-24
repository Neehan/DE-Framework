"""Read Markdown prompts and substitute named fields without interpreting mathematical braces."""

from collections.abc import Mapping
from pathlib import Path

from harness.utils.constants import PROMPT_PLACEHOLDER, TEXT_ENCODING


def load_prompt(path: Path, values: Mapping[str, str]) -> str:
    """Render template fields once, preserving inserted text and rejecting missing values."""
    template = path.read_text(encoding=TEXT_ENCODING).strip()
    missing = set(PROMPT_PLACEHOLDER.findall(template)) - values.keys()
    if missing:
        raise ValueError(f"Unfilled placeholders in {path.name}: {sorted(missing)}")
    return PROMPT_PLACEHOLDER.sub(lambda match: values[match.group("name")], template)
