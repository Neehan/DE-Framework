"""Grade mathematical correctness using the 0/5/6/7 rubric."""

from audit.base_audit import BaseAudit
from audit.constants import CORRECTNESS, CORRECTNESS_PROMPT_FILE, CORRECTNESS_SCHEMA


class CorrectnessAudit(BaseAudit):
    """Supply the correctness prompt and score contract; no overrides required."""

    name = CORRECTNESS
    prompt_file = CORRECTNESS_PROMPT_FILE
    schema = CORRECTNESS_SCHEMA
