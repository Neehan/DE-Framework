"""Codex gateway names, private storage, and pinned LiteLLM compatibility settings."""

from pathlib import Path

IMAGE = "harness:codex"
CONTAINER_PREFIX = "harness-codex"
AUTH_VOLUME_PREFIX = "harness-codex-auth"
ACCOUNT_LABEL = "harness.codex.account"
HOST = "127.0.0.1"
BIND_HOST = "0.0.0.0"
BASE_PORT = 4200
PORT = 4000
MAX_PORT = 65535
MIN_ACCOUNT = 1
MODEL_ALIAS = "gpt-*"
PYTHON = "/app/.venv/bin/python"
AUTH_DIRECTORY = Path("/var/lib/litellm-chatgpt")
AUTH_FILENAME = "auth.json"
CONFIG_DIRECTORY = "/app"
CONFIG_FILENAME = "gateway.json"
PRIVATE_FILE_MODE = 0o600
STARTUP_TIMEOUT_SECONDS = 120
HEALTH_TIMEOUT_SECONDS = 3
HEALTH_POLL_SECONDS = 1
REQUIRED_AUTH_FIELDS = ("access_token", "refresh_token", "id_token", "account_id")
AUTH_SCHEMA = {
    "type": "object", "required": ["tokens"],
    "properties": {"tokens": {
        "type": "object", "required": list(REQUIRED_AUTH_FIELDS),
        "properties": {name: {"type": "string", "minLength": 1} for name in REQUIRED_AUTH_FIELDS},
    }},
}
IDENTITY_COMMAND = (
    "import json; from pathlib import Path; "
    f"print(json.loads(Path('{AUTH_DIRECTORY / AUTH_FILENAME}').read_text())['account_id'])"
)
PATCH_TARGET = Path("/app/.venv/lib")
PATCH_GLOB = "python*/site-packages/litellm/litellm_core_utils/prompt_templates/factory.py"
PATCH_ORIGINAL = '                    next_m["content"] = m["content"] + " " + next_m["content"]'
# This replacement applies only to the pinned third-party LiteLLM source.
PATCH_REPLACEMENT = '''                    if isinstance(m["content"], list) or isinstance(next_m["content"], list):
                        system_content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": str(m["content"])}]
                        next_content = next_m["content"] if isinstance(next_m["content"], list) else [{"type": "text", "text": str(next_m["content"])}]
                        next_m["content"] = system_content + next_content
                    else:
                        next_m["content"] = str(m["content"]) + " " + str(next_m["content"])'''
