"""Shared audit inputs and verdicts for component and native-worker tests."""

CORRECTNESS_RESULT = {"score": 0, "note": "The submitted proof leaves its main claim unjustified."}
STEP_RESULT = {"steps": [{"present": True, "reason": "The ingredient is explicitly recognized."} for _ in range(3)]}
REFERENCE = "A complete reference proof, provided only to the audit."
STEPS = ["Introduce the construction.", "Establish its invariant.", "Derive the conclusion."]
SUBMISSION = "## Final Solution\nThe main claim holds, but its proof is missing."
AUDIT_MODEL = "gpt-audit-test"
COMPLETE_AUDIT = {
    "correctness": CORRECTNESS_RESULT, "correctness_audit_model": AUDIT_MODEL,
    "step_recognition": STEP_RESULT, "step_audit_model": f"litellm/{AUDIT_MODEL}",
}
COMPILER_EXPERIMENT = "aobench/litellm%2Fgpt-test/unaided-4x"
COMPILER_SEEDS = (2, 10)
