"""Fixed inputs for deterministic refinement tests."""

PROBLEM = "Prove the test claim."
DATASET_REPO_URL = "https://huggingface.co/datasets/anonymous/benchmark"
PROMPT_TEST_TEMPLATE = r"Simplify \frac{x}{y}. Problem: {{problem}}. Budget: {{budget_tokens}}."
PROMPT_TEST_PROBLEM = r"Explain \{x\} and {{budget_tokens}} literally"
INITIAL_SOLUTION = "Initial complete solution."
REVISED_SOLUTION = "Revised complete solution."
PARTIAL_SOLUTION = "Unfinished revision..."
PHASE_TOKENS = 1_000
BUDGET_TOKENS = 100_000
WARNING_CROSSING_TOKENS = 82_650
LATER_USAGE_TOKENS = 90_000
EXPECTED_WARNING_REMAINDER = 17_350
UNIT_TOKENS = 200_000
TOTAL_UNITS = 4
PREFIX_UNITS = 3
TWO_PHASES = 2
ONE_ROUND = 1
TWO_ROUNDS = 2
THREE_CLEAN_CRITIQUES = 3
FIVE_ROUNDS = 5
ASYNC_TEST_TIMEOUT_SECONDS = 2
EVENT_LOOP_YIELD_SECONDS = 0
DISCONNECT_MESSAGE = "injected stream disconnect"
NATIVE_SESSION_ID = "12345678-1234-5678-1234-567812345678"
SDK_SESSION_ID = "test-session"
SDK_MESSAGE_ID = "test-message"
SDK_MODEL = "test-model"
SDK_CLI_PATH = "/usr/local/bin/claude"
SDK_INITIAL_TOKENS = 3
SDK_FIRST_TOKENS = 40
SDK_SECOND_TOKENS = 60
SDK_FINAL_TOKENS = 120
SDK_BUFFER_OVERFLOW_MESSAGES = 150
SDK_WARNING = "You have 17000 output tokens left."
SDK_PROJECT_DIRECTORY = "projects/test-project"
TEST_ATTEMPT_DIRECTORY = "results/test-dataset/test-model/unaided/test-problem/seed_1"
WORKSPACE_FILENAME = "work.txt"
FAILED_PHASE_FILENAME = "unfinished.txt"
LOCK_HELD_EXIT_CODE = 73
EXPECTED_RECOVERY_DELAYS = (2.0, 4.0, 8.0, 16.0, 30.0, 30.0)
SYNTHETIC_SUCCESS = "No response requested."
TRANSIENT_PROVIDER_ERROR = "status code: 503"
TRANSIENT_DIAGNOSTICS = (
    "Stream ended without receiving any events", "Stream idle timeout - no chunks received",
    "Peer closed connection", "Incomplete chunked read", "RemoteProtocolError",
    "Connection reset", "Connection terminated", "Disconnect/reset before headers",
    "Service unavailable", "Bad gateway", "Gateway timeout",
    "API Error: 502", "API Error: 503", "API Error: 504",
    "Status code: 502", "Status code: 503", "Status code: 504",
)
FATAL_PROCESS_EXIT_CODE = 1
KILLED_PROCESS_EXIT_CODE = -9
SANDBOX_IMAGE = "harness:test"
SANDBOX_NETWORK = "harness-provider-only"
SANDBOX_CONTAINER_ID = "test-container"
SANDBOX_PROVIDER_ENV = {"ANTHROPIC_AUTH_TOKEN": "test-secret", "ANTHROPIC_BASE_URL": "http://provider:4000"}
SANDBOX_PROXY_ENV = {"ANTHROPIC_API_KEY": "disposable-run-token", "ANTHROPIC_BASE_URL": "http://host.docker.internal:4000"}
SANDBOX_COMMAND = ("python", "-V")
PROVIDER_TEST_ADDRESSES = ("192.0.2.10", "2001:db8::10")
PROVIDER_TEST_HOST = "provider"
PROVIDER_TEST_PORT = 4000
FIREWALL_TEST_COMMAND = ["iptables", "-P", "OUTPUT", "DROP"]
SANDBOX_REQUEST = '{"problem": "selected problem only"}'
SANDBOX_FAILURE = b"injected container failure"
LOCK_PROBE = f"""import fcntl
import sys
with open(sys.argv[1], 'a+') as handle:
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit({LOCK_HELD_EXIT_CODE})
"""
LOCK_HOLDER = """import fcntl
import sys
with open(sys.argv[1], 'a+') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('locked', flush=True)
    sys.stdin.read()
"""
FORK_CRASH_PROBE = """import os
import sys
from pathlib import Path
from harness.self_refine.models import RefinementState
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from tests.constants import BUDGET_TOKENS, FATAL_PROCESS_EXIT_CODE, PROBLEM

source = SessionManager(Path(sys.argv[1]), RunLock)
source.open(RefinementState(PROBLEM, BUDGET_TOKENS, None))
target = SessionManager(Path(sys.argv[2]), RunLock)
if sys.argv[3] == 'before_archive':
    os.replace = lambda *args: os._exit(FATAL_PROCESS_EXIT_CODE)
else:
    target._write_solutions = lambda state: os._exit(FATAL_PROCESS_EXIT_CODE)
source.fork(target, BUDGET_TOKENS)
"""

READ_ONLY_DIRECTORY_MODE = 0o555
OWNER_DIRECTORY_MODE = 0o700

FIRST_SOLUTION_FILENAME = "solution_1x.md"
