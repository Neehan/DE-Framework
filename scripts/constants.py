"""Host setup paths, dataset source, and auth-file environment names."""

import re

AUTH_FILE_PREFIX = "CODEX_AUTH_FILE"
AUTH_FILE_PATTERN = re.compile(rf"{AUTH_FILE_PREFIX}_(?P<account>[1-9]\d*)\Z")
DATASET_REPO_URL_ENV = "HF_DATASET_REPO_URL"
DATASET_FILE_URL = "{repository}/resolve/main/{name}.jsonl"
DOWNLOAD_TIMEOUT_SECONDS = 60
