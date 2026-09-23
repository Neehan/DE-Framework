"""Launcher paths, supported inputs, and host-side provider configuration."""

import re
from pathlib import Path

IMPLEMENTATION_DIRECTORY = Path(__file__).resolve().parent.parent
DATASETS_DIRECTORY = IMPLEMENTATION_DIRECTORY / "datasets"
RESULTS_DIRECTORY = IMPLEMENTATION_DIRECTORY / "results"
DOTENV_FILE = IMPLEMENTATION_DIRECTORY / ".env"
DATASET_NAMES = ("aobench", "imoproofbench")
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]*\Z")
DOCKER_IMAGE = "harness:run"
RUN_DOCKERFILE = IMPLEMENTATION_DIRECTORY / "docker/Dockerfile.run"
NETWORK_PREFIX = "harness-launcher-"
WORKER_MODULE = "launcher.worker"
FAILED_EXIT_CODE = 1
LOG_FORMAT = "%(levelname)s %(message)s"
ANTHROPIC_URL = "https://api.anthropic.com"
META_URL = "https://api.meta.ai"
LITELLM_PREFIX = "litellm/"
CLAUDE_PREFIX = "claude-"
GPT_PREFIX = "gpt-"
LITELLM_URL_ENV = "LITELLM_BASE_URL"
LITELLM_KEY_ENV = "LITELLM_API_KEY"
META_URL_ENV = "META_BASE_URL"
META_KEY_ENV = "META_API_KEY"

CREDENTIAL_ENV_PATTERN = r"{name}(?:_(?P<suffix>\d+))?"

MODEL_DIRECTORY_ESCAPE = re.compile(r"[/A-Z]")
PREFIX_LOCK_POLL_SECONDS = 1.0

AUDIT_DOCKER_IMAGE = "harness:audit"
AUDIT_DOCKERFILE = IMPLEMENTATION_DIRECTORY / "docker/Dockerfile.audit"
AUDIT_WORKER_MODULE = "audit.worker"
