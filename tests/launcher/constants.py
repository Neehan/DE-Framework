"""Explicit launcher inputs and forbidden reference sentinels."""

CLI_ARGUMENTS = ["--dataset", "aobench", "--experiment", "unaided", "--model", "claude-test", "--seeds", "1", "3", "24"]
REFERENCE_SENTINEL = "reference solution must stay on host"
REFERENCE_SKETCH = "Use the selected reference construction."
ALTERNATE_SKETCH = "Use the selected alternate construction."
PROBLEM_ROWS = [
    {"problem_id": "p1", "statement": "Prove A.", "domain": "algebra", "solutions": [
        {"role": "reference", "solution": REFERENCE_SENTINEL, "sketch": REFERENCE_SKETCH, "steps": [REFERENCE_SENTINEL]},
        {"role": "alternate", "solution": REFERENCE_SENTINEL, "sketch": ALTERNATE_SKETCH, "steps": [REFERENCE_SENTINEL]},
    ]},
    {"problem_id": "p2", "statement": "Prove B.", "domain": "combinatorics", "solutions": []},
]
TEST_SEEDS = [1, 2, 3]
TWO_WORKERS = 2
TEST_ROUTE_KEY = "only-the-host-gets-this-key"
TEST_ROUTE_URL = "http://127.0.0.1:4000"
