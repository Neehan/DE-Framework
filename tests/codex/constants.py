"""Synthetic account identities and credentials; never valid provider authentication."""

AUTH_TOKENS = {"access_token": "fake-access", "refresh_token": "fake-refresh", "id_token": "fake-id", "account_id": "fake-account"}
GATEWAY_KEY = "sk-test-gateway"
IMAGE_ID = "sha256:gateway-test"
ACCOUNT_NUMBER = 1
SECOND_ACCOUNT_NUMBER = 2
CONFIG_ERROR_CODE = 1
CONFIG_ERROR = b"copy failed"
CODEX_TEST_IMAGE_ENV = "HARNESS_CODEX_TEST_IMAGE"
TOOL_FILENAME = "gateway-check.txt"
TOOL_CONTENT = "gateway tool execution verified\n"
TOOL_CALL_ID = "call_gateway_check"
TEST_MODEL = "gpt-5.6-sol"
FUTURE_EXPIRY = 4_000_000_000
RESPONSES_CREATED = 1_700_000_000
RESPONSE_ID = "resp_gateway_test"
