"""Template rendering preserves mathematical input and fails on missing substitutions."""

from pathlib import Path

import pytest
from harness.self_refine.constants import SOLVE_PROMPT_FILE
from harness.utils.constants import TEXT_ENCODING
from harness.utils.prompt_loader import load_prompt

from tests.constants import BUDGET_TOKENS, PROMPT_TEST_PROBLEM, PROMPT_TEST_TEMPLATE


def test_substitution_preserves_math_and_does_not_render_inserted_text(tmp_path: Path) -> None:
    """Template braces and placeholder-like problem text survive a single rendering pass."""
    path = tmp_path / SOLVE_PROMPT_FILE.name
    path.write_text(PROMPT_TEST_TEMPLATE, encoding=TEXT_ENCODING)
    prompt = load_prompt(path, {
        "problem": PROMPT_TEST_PROBLEM, "budget_tokens": str(BUDGET_TOKENS),
    })
    assert prompt == r"Simplify \frac{x}{y}. Problem: " + PROMPT_TEST_PROBLEM + f". Budget: {BUDGET_TOKENS}."


def test_missing_template_value_fails_from_another_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Locate real templates independently of cwd and reject a missing problem immediately."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unfilled placeholders in solve.md:.*problem"):
        load_prompt(SOLVE_PROMPT_FILE, {"budget_tokens": str(BUDGET_TOKENS)})
