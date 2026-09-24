"""The fixed correctness-then-recognition audit sequence."""

from audit.correctness_audit import CorrectnessAudit
from audit.step_recognition_audit import StepRecognitionAudit

AUDITS = {audit.name: audit for audit in (CorrectnessAudit, StepRecognitionAudit)}
