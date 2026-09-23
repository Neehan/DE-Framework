"""Experiment identities, fixed compute allocations, and sketch roles."""

from harness.utils.constants import OUTPUT_TOKENS_PER_BLOCK, PROMPTS_DIRECTORY

UNAIDED = "unaided"
NO_SKETCH = "no-sketch"
ORACLE_EXECUTION = "oracle-execution"
ORACLE_SKETCH = "oracle-sketch"
ALT_SKETCH = "alt-sketch"
CONTINUE = "continue-at-3x"
CONTINUE_ORACLE = "continue-oracle-at-3x"
REFERENCE_ROLE = "reference"
ALTERNATE_ROLE = "alternate"
SINGLE_BLOCK_MULTIPLIER = 1
ORACLE_EXECUTION_MULTIPLIER = 8
CONTINUATION_TOTAL_MULTIPLIER = 4
PREFIX_MULTIPLIER = 3
PREFIX_TOKENS = PREFIX_MULTIPLIER * OUTPUT_TOKENS_PER_BLOCK
PREFIX_DIRECTORY = f"unaided-prefix-{PREFIX_MULTIPLIER}x"
SKETCH_PROMPT_FILE = PROMPTS_DIRECTORY / "sketch.md"
CONTINUE_PROMPT_FILE = PROMPTS_DIRECTORY / "continue.md"
