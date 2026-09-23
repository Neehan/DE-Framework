"""Deterministic inputs for request filtering and local HTTP forwarding tests."""

LOOPBACK = "127.0.0.1"
TEST_KEY = "upstream-secret-never-in-container"
REQUEST_BODY = {"model": "local-test", "max_tokens": 100, "messages": [{"role": "user", "content": "Prove the claim."}]}
STREAM_CHUNKS = (b"data: first\n\n", b"data: second\n\n")

SECOND_KEY = "second-upstream-credential"
POOL_NOW = 1_000.0
RESET_DELAY = 60.0
LIMIT_RESPONSES = {
    "worker-reported-limit": (200, {}),
    "limited-reset": (429, {"anthropic-ratelimit-unified-reset": str(POOL_NOW + RESET_DELAY)}),
    "limited-delay": (429, {"retry-after": str(RESET_DELAY)}),
    "limited-missing": (429, {}),
    "limited-invalid": (429, {"anthropic-ratelimit-unified-reset": "invalid"}),
    "limited-past": (429, {"anthropic-ratelimit-unified-reset": str(POOL_NOW)}),
    "limited-rejected": (200, {"anthropic-ratelimit-unified-status": "rejected", "anthropic-ratelimit-unified-reset": str(POOL_NOW + RESET_DELAY)}),
    "invalid-credential": (401, {}),
    "unavailable": (503, {}),
}
