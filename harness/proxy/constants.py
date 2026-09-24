"""Allowed inference protocol and host-side proxy settings."""

from harness.session.constants import LOCAL_TOOLS, STRUCTURED_OUTPUT_TOOL
from harness.utils.constants import MAX_OUTPUT_TOKENS_PER_RESPONSE

PROXY_BIND_HOST = "0.0.0.0"
PROXY_DOCKER_HOST = "host.docker.internal"
PROXY_EPHEMERAL_PORT = 0
PROXY_PORT_INDEX = 1
PROXY_TOKEN_BYTES = 32
PROVIDER_CREDENTIAL_COUNT = 1
PROXY_MAX_REQUEST_BYTES = 32 * 1024 * 1024
PROXY_CONNECT_TIMEOUT_SECONDS = 30
PROXY_SHUTDOWN_TIMEOUT_SECONDS = 5
PROXY_CHUNK_BYTES = 64 * 1024
INFERENCE_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens"})
INFERENCE_QUERY = "beta=true"
API_KEY_ENV = "ANTHROPIC_API_KEY"
AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
AUTH_ENV_HEADERS = {API_KEY_ENV: "x-api-key", AUTH_TOKEN_ENV: "authorization", OAUTH_TOKEN_ENV: "authorization"}
ANTHROPIC_VERSION = "2023-06-01"
OAUTH_BETA = "oauth-2025-04-20"
ALLOWED_BETAS = frozenset({
    OAUTH_BETA, "claude-code-20250219", "interleaved-thinking-2025-05-14",
    "context-1m-2025-08-07", "context-management-2025-06-27", "prompt-caching-scope-2026-01-05",
    "output-128k-2025-02-19", "effort-2025-11-24", "adaptive-thinking-2026-01-28",
    "thinking-token-count-2026-05-13", "structured-outputs-2025-12-15",
    "extended-cache-ttl-2025-04-11", "mid-conversation-system-2026-04-07",
    "task-budgets-2026-03-13",
})

# Locally supplied content and tool calls are allowed; remote references and hosted tools have no schema.
REQUEST_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["model", "messages"],
    "properties": {
        "model": {"type": "string", "minLength": 1},
        "messages": {"type": "array", "items": {"$ref": "#/$defs/message"}},
        "system": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"$ref": "#/$defs/text"}}]},
        "max_tokens": {"type": "integer", "minimum": 1, "maximum": MAX_OUTPUT_TOKENS_PER_RESPONSE},
        "stream": {"type": "boolean"},
        "temperature": {"type": "number"}, "top_p": {"type": "number"}, "top_k": {"type": "integer"},
        "stop_sequences": {"type": "array", "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"$ref": "#/$defs/tool"}},
        "tool_choice": {"type": "object", "additionalProperties": False, "required": ["type"], "properties": {
            "type": {"enum": ["auto", "any", "tool", "none"]}, "name": {"enum": [*LOCAL_TOOLS, STRUCTURED_OUTPUT_TOOL]},
            "disable_parallel_tool_use": {"type": "boolean"},
        }},
        "thinking": {"type": "object", "additionalProperties": False, "required": ["type"], "properties": {
            "type": {"enum": ["enabled", "disabled", "adaptive"]}, "budget_tokens": {"type": "integer"},
        }},
        "output_config": {"type": "object", "additionalProperties": False, "properties": {
            "effort": {"enum": ["low", "medium", "high", "max"]},
            "task_budget": {"type": "object", "additionalProperties": False, "required": ["type", "total"], "properties": {
                "type": {"const": "tokens"}, "total": {"type": "integer", "minimum": 1},
            }},
            "format": {"type": "object", "additionalProperties": False, "required": ["type", "schema"], "properties": {
                "type": {"const": "json_schema"}, "schema": {"type": "object"},
            }},
        }},
        "metadata": {"type": "object", "additionalProperties": False, "properties": {
            "user_id": {"type": "string"},
        }},
        "cache_control": {"$ref": "#/$defs/cache"},
        "context_management": {"type": "object", "additionalProperties": False, "properties": {
            "edits": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["type"], "properties": {
                    "type": {"const": "clear_thinking_20251015"}, "keep": {"const": "all"},
                }}},
        }},
    },
    "$defs": {
        "cache": {"type": "object", "additionalProperties": False, "required": ["type"], "properties": {
            "type": {"const": "ephemeral"}, "ttl": {"enum": ["5m", "1h"]},
        }},
        "text": {"type": "object", "additionalProperties": False, "required": ["type", "text"], "properties": {
            "type": {"const": "text"}, "text": {"type": "string"}, "cache_control": {"$ref": "#/$defs/cache"},
        }},
        "image": {"type": "object", "additionalProperties": False, "required": ["type", "source"], "properties": {
            "type": {"const": "image"}, "cache_control": {"$ref": "#/$defs/cache"},
            "source": {"type": "object", "additionalProperties": False, "required": ["type", "media_type", "data"], "properties": {
                "type": {"const": "base64"}, "media_type": {"enum": ["image/png", "image/jpeg", "image/gif", "image/webp"]},
                "data": {"type": "string"},
            }},
        }},
        "document": {"type": "object", "additionalProperties": False, "required": ["type", "source"], "properties": {
            "type": {"const": "document"}, "cache_control": {"$ref": "#/$defs/cache"},
            "source": {"type": "object", "additionalProperties": False, "required": ["type", "media_type", "data"], "properties": {
                "type": {"const": "base64"}, "media_type": {"const": "application/pdf"}, "data": {"type": "string"},
            }},
        }},
        "local_content": {"anyOf": [{"$ref": "#/$defs/text"}, {"$ref": "#/$defs/image"}, {"$ref": "#/$defs/document"}]},
        "message": {"type": "object", "additionalProperties": False, "required": ["role", "content"], "properties": {
            "role": {"enum": ["user", "assistant"]},
            "content": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"$ref": "#/$defs/block"}}]},
        }},
        "tool": {"type": "object", "additionalProperties": False, "required": ["name", "input_schema"], "properties": {
            "type": {"const": "custom"}, "name": {"enum": [*LOCAL_TOOLS, STRUCTURED_OUTPUT_TOOL]}, "description": {"type": "string"},
            "input_schema": {"type": "object"}, "cache_control": {"$ref": "#/$defs/cache"},
            "strict": {"type": "boolean"},
        }},
        "block": {"anyOf": [
            {"$ref": "#/$defs/local_content"},
            {"type": "object", "additionalProperties": False, "required": ["type", "id", "name", "input"], "properties": {
                "type": {"const": "tool_use"}, "id": {"type": "string"}, "name": {"enum": [*LOCAL_TOOLS, STRUCTURED_OUTPUT_TOOL]},
                "input": {"type": "object"}, "cache_control": {"$ref": "#/$defs/cache"},
            }},
            {"type": "object", "additionalProperties": False, "required": ["type", "tool_use_id"], "properties": {
                "type": {"const": "tool_result"}, "tool_use_id": {"type": "string"}, "is_error": {"type": "boolean"},
                "content": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"$ref": "#/$defs/local_content"}}]},
                "cache_control": {"$ref": "#/$defs/cache"},
            }},
            {"type": "object", "additionalProperties": False, "required": ["type", "thinking", "signature"], "properties": {
                "type": {"const": "thinking"}, "thinking": {"type": "string"}, "signature": {"type": "string"},
            }},
            {"type": "object", "additionalProperties": False, "required": ["type", "data"], "properties": {
                "type": {"const": "redacted_thinking"}, "data": {"type": "string"},
            }},
        ]},
    },
}

CREDENTIAL_COOLDOWN_SECONDS = 300.0
ANTHROPIC_RESET_HEADER = "anthropic-ratelimit-unified-reset"
ANTHROPIC_LIMIT_STATUS_HEADER = "anthropic-ratelimit-unified-status"
ANTHROPIC_LIMIT_REJECTED = "rejected"
RETRY_AFTER_HEADER = "retry-after"
