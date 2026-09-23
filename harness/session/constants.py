"""SDK protocol settings used by agent sessions."""

import re
from pathlib import Path

from harness.utils.constants import SELF_REFINE_PROMPTS_DIRECTORY

REPLAY_USER_MESSAGES = "replay-user-messages"
SESSION_NAME_ARG = "name"
DEFAULT_SESSION_ID = "default"
SUCCESS_SUBTYPE = "success"
RATE_LIMIT_REJECTED = "rejected"
API_ERROR_PREFIX = "api error:"
DEFAULT_SYSTEM_PROMPT_FILE = SELF_REFINE_PROMPTS_DIRECTORY / "shared_self_refine.md"
AUTO_COMPACT_ARG = "autocompact"
MAX_OUTPUT_TOKENS_ENV = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
CLI_PATH_ENV = "HARNESS_CLI_PATH"
API_TIMEOUT_ENV = "API_TIMEOUT_MS"
STREAM_IDLE_TIMEOUT_ENV = "CLAUDE_STREAM_IDLE_TIMEOUT_MS"
STREAM_WATCHDOG_ENV = "CLAUDE_ENABLE_STREAM_WATCHDOG"
DISABLE_NONSTREAMING_FALLBACK_ENV = "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"
MAX_API_RETRIES_ENV = "CLAUDE_CODE_MAX_RETRIES"
ENV_ENABLED = "1"
STRUCTURED_OUTPUT_TOOL = "StructuredOutput"
LOCAL_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")
DISALLOWED_TOOLS = frozenset({
    "WebSearch", "WebFetch", "Task", "Agent", "Workflow", "Skill", "SlashCommand",
    "Artifact", "SendMessage", "SendUserFile", "TaskCreate", "TaskGet", "TaskList",
    "TaskUpdate", "TaskOutput", "TaskStop", "CronCreate", "CronDelete", "CronList",
    "ScheduleWakeup", "Monitor", "PushNotification", "RemoteTrigger", "EnterWorktree",
    "ExitWorktree", "EnterPlanMode", "ExitPlanMode", "EndConversation", "ToolSearch",
    "AskUserQuestion", "NotebookEdit", "PowerShell",
})
BASH_DENIALS = frozenset({
    "Bash(curl:*)", "Bash(wget:*)", "Bash(claude:*)", "Bash(git:*)",
    "Bash(ssh:*)", "Bash(scp:*)", "Bash(sftp:*)", "Bash(ftp:*)", "Bash(rsync:*)",
    "Bash(nc:*)", "Bash(ncat:*)", "Bash(telnet:*)", "Bash(pip install:*)",
    "Bash(pip3 install:*)", "Bash(pip download:*)", "Bash(npm install:*)",
    "Bash(npm i:*)", "Bash(apt:*)", "Bash(apt-get:*)", "Bash(brew:*)", "Bash(conda install:*)",
})
SETTING_SOURCES_ARG = "setting-sources"
MANAGED_SETTINGS_FILE = Path("/etc/claude-code/managed-settings.json")
TOOL_POLICY_ARGS = frozenset({"disallowedTools", "disallowed-tools"})
TRANSIENT_ERROR_MARKERS = (
    "rate_limit", "server_error", "overloaded_error", "connection reset", "connection closed",
    "connection refused", "econnreset", "econnrefused", "socket hang up",
    "stream disconnected", "timed out", "timeout", "status code: 429",
    "status code: 500", "status code: 502", "status code: 503", "status code: 504",
    "stream ended without receiving any events", "peer closed connection",
    "incomplete chunked read", "remoteprotocolerror", "connection terminated",
    "disconnect/reset before headers", "service unavailable", "bad gateway",
    "api error: 502", "api error: 503", "api error: 504",
)

SESSION_FILENAME = "session.zip"
SOLUTION_FILENAME_TEMPLATE = "solution_{multiplier}x.md"
SOLUTION_GLOB = "solution*.md"
LOCK_FILENAME = ".lock"
CHECKPOINT_FILENAME = "checkpoint.json"
RUNTIME_DIRECTORY = ".runtime"
WORKSPACE_DIRECTORY = "workspace"
SDK_DIRECTORY = "sdk"
SDK_DEBUG_DIRECTORY = Path(SDK_DIRECTORY) / "debug"
CLAUDE_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"
TRANSCRIPT_EXTENSION = ".jsonl"
TEMP_FILE_SUFFIX = ".tmp"
ZIP_MODE_SHIFT = 16

RATE_LIMIT_DIAGNOSTIC = re.compile(r"\b429\b|rate[_ -]limit", re.IGNORECASE)
SPEND_LIMIT_MARKERS = ("spend limit", "usage limit reached", "credit balance is too low")
SPEND_LIMIT_MESSAGE = "Provider spend limit reached"
