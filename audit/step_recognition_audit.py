"""Recognize the frozen outline's three steps independently of correctness."""

from harness.utils.constants import COUNT_INCREMENT

from audit.base_audit import BaseAudit
from audit.constants import (
    STEP_RECOGNITION,
    STEP_RECOGNITION_PROMPT_FILE,
    STEP_RECOGNITION_SCHEMA,
)
from audit.models import AuditRequest


class StepRecognitionAudit(BaseAudit):
    """Add the reference outline to the recognition prompt; no overrides required."""

    name = STEP_RECOGNITION
    prompt_file = STEP_RECOGNITION_PROMPT_FILE
    schema = STEP_RECOGNITION_SCHEMA

    def _get_prompt_values(self, request: AuditRequest) -> dict[str, str]:
        """Render the three fixed steps without including a correctness judgment."""
        assert request.steps is not None
        outline = "\n".join(f"{number}. {step}" for number, step in enumerate(request.steps, start=COUNT_INCREMENT))
        return {**super()._get_prompt_values(request), "outline": outline}
