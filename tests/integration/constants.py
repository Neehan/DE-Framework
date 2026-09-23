"""Settings for opt-in Docker checks using an existing harness image and local providers."""

from pathlib import Path

from harness.sandbox.constants import CONTAINER_RUN_DIRECTORY
from harness.utils.constants import RECOVERY_MAX_RETRIES

INTEGRATION_IMAGE_ENV = "HARNESS_TEST_IMAGE"
IMPLEMENTATION_PARENT_INDEX = 2
IMPLEMENTATION_DIRECTORY = Path(__file__).resolve().parents[IMPLEMENTATION_PARENT_INDEX]
DOCKER_TIMEOUT_SECONDS = 180
CLI_TIMEOUT_SECONDS = 120
SOCKET_TIMEOUT_SECONDS = 0.5
ENDPOINT_START_TIMEOUT_SECONDS = 10
ENDPOINT_POLL_SECONDS = 0.1
PROVIDER_PORT = 4000
BLOCKED_PORT = 4001
PUBLIC_HTTPS_ENDPOINT = ("1.1.1.1", 443)
DOCKER_DNS_ENDPOINT = ("127.0.0.11", 53)
DNS_RESPONSE_BYTES = 512
DNS_QUERY = b"\x04\xd2\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x08provider\x00\x00\x01\x00\x01"
CAPABILITY_FIELDS = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
HEXADECIMAL_BASE = 16
LOOPBACK_ADDRESS = "127.0.0.1"
ANY_ADDRESS = "0.0.0.0"
EPHEMERAL_PORT = 0
HTTP_OK = 200
TEST_UPSTREAM_KEY = "test-upstream-only-secret"
PROVIDER_MODEL = "claude-opus-4-6"
LAUNCHER_MODELS = (PROVIDER_MODEL, "litellm/gpt-local-test", "muse-local-test")
INPUT_TOKENS = 100
OUTPUT_TOKENS = 40
BLOCK_INDEX = 0
PROVIDER_RESPONSE = "## Final Solution\nComplete proof from the local test provider."
BRANCH_NAMES = ("control", "oracle")
NETWORK_REPORT_FILENAME = "network-check.json"
OWN_RUN_FILE = "own.txt"
OWN_RUN_CONTENT = "owned scratch content"
SIBLING_SOLUTION = Path("other-run/solution_1x.md")
LOCAL_DATASET = Path("datasets/reference.jsonl")
ABSOLUTE_ESCAPE_LINK = "absolute-link"
RELATIVE_ESCAPE_LINK = "relative-link"
DOCKER_SOCKET = Path("/var/run/docker.sock")
NETWORK_MODULE = "tests.integration.network_check"
NATIVE_SESSION_MODULE = "tests.integration.native_session_check"
PYTHONPATH = "/app:/test-support"
PROVIDER_SERVER_CODE = f"""from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from threading import Event, Thread
for port in ({PROVIDER_PORT}, {BLOCKED_PORT}):
    server = ThreadingHTTPServer(('{ANY_ADDRESS}', port), SimpleHTTPRequestHandler)
    Thread(target=server.serve_forever, daemon=True).start()
print('ready', flush=True)
Event().wait()
"""

TOOL_PROBLEM = "Exercise the native tools for the proxy compatibility check, then give a final solution."
TOOL_FILE = f"{CONTAINER_RUN_DIRECTORY}/tool-check.txt"
TOOL_IMAGE = f"{CONTAINER_RUN_DIRECTORY}/tool-check.png"
TOOL_PDF = f"{CONTAINER_RUN_DIRECTORY}/tool-check.pdf"
TOOL_PDF_BASE64 = 'JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCA3MiA3Ml0gL0NvbnRlbnRzIDQgMCBSID4+CmVuZG9iago0IDAgb2JqCjw8IC9MZW5ndGggMCA+PgpzdHJlYW0KCmVuZHN0cmVhbQplbmRvYmoKeHJlZgowIDUKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDA5IDAwMDAwIG4gCjAwMDAwMDAwNTggMDAwMDAgbiAKMDAwMDAwMDExNSAwMDAwMCBuIAowMDAwMDAwMjAwIDAwMDAwIG4gCnRyYWlsZXIKPDwgL1NpemUgNSAvUm9vdCAxIDAgUiA+PgpzdGFydHhyZWYKMjQ5CiUlRU9GCg=='
TOOL_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aG1cAAAAASUVORK5CYII="
TOOL_IMAGE_ID = "toolu_read_image"
TOOL_ERROR_ID = "toolu_command_error"
DENIED_TOOL_IDS = ("toolu_blocked_curl", "toolu_blocked_wget")
TOOL_STEPS = {
    "toolu_write_text": ("Write", {"file_path": TOOL_FILE, "content": "before proxy edit\n"}),
    "toolu_read_text": ("Read", {"file_path": TOOL_FILE}),
    "toolu_edit_text": ("Edit", {"file_path": TOOL_FILE, "old_string": "before proxy edit", "new_string": "after proxy edit"}),
    "toolu_glob": ("Glob", {"pattern": "tool-check.txt", "path": CONTAINER_RUN_DIRECTORY}),
    "toolu_grep": ("Grep", {"pattern": "after proxy edit", "path": TOOL_FILE, "output_mode": "content"}),
    "toolu_check_text": ("Bash", {"command": f"cat {TOOL_FILE}", "description": "Check the edited scratch file"}),
    "toolu_blocked_curl": ("Bash", {"command": "curl --version", "description": "Check the mandatory command denial"}),
    "toolu_blocked_wget": ("Bash", {"command": "wget --version", "description": "Check the mandatory command denial"}),
    "toolu_create_image": ("Bash", {"command": f"python -c \"import base64; from pathlib import Path; Path('{TOOL_IMAGE}').write_bytes(base64.b64decode('{TOOL_IMAGE_BASE64}'))\"", "description": "Create a local test image"}),
    TOOL_IMAGE_ID: ("Read", {"file_path": TOOL_IMAGE}),
    "toolu_create_pdf": ("Bash", {"command": f"python -c \"import base64; from pathlib import Path; Path('{TOOL_PDF}').write_bytes(base64.b64decode('{TOOL_PDF_BASE64}'))\"", "description": "Create a local test PDF"}),
    "toolu_read_pdf": ("Read", {"file_path": TOOL_PDF}),
    TOOL_ERROR_ID: ("Bash", {"command": "exit 7", "description": "Check that a failed command returns a tool error"}),
}

LIMITED_KEYS = tuple(f"limited-test-key-{index}" for index in range(RECOVERY_MAX_RETRIES + 1))
STREAM_RATE_LIMIT_MODEL = "claude-stream-rate-limit-test"
RATE_LIMIT_STREAM_ERROR = 'event: error\ndata: {"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}\n\n'
SPEND_LIMIT_MODEL = "claude-spend-limit-test"
STREAM_SPEND_LIMIT_MODEL = "claude-stream-spend-limit-test"
SPEND_LIMIT_STREAM_ERROR = 'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error","message":"Provider spend limit reached"}}\n\n'

PREFIX_OVERRUN_RESPONSE = "Discard this over-budget prefix reasoning before either continuation."
