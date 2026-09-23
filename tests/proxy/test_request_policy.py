"""Accepted native conversation content and genuinely different provider-side lookup escape routes."""

import json
from copy import deepcopy
from typing import Any

import pytest
from harness.proxy.request_policy import RequestPolicy
from jsonschema import ValidationError

from tests.constants import BUDGET_TOKENS
from tests.proxy.constants import REQUEST_BODY


@pytest.mark.parametrize("addition", [
    {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
    {"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]},
    {"tools": [{"type": "code_execution_20260120", "name": "code_execution"}]},
    {"tools": [{"type": "future_hosted_tool", "name": "Bash", "input_schema": {}}]},
    {"tools": [{"name": "WebSearch", "input_schema": {}}]},
    {"mcp_servers": [{"type": "url", "url": "https://forbidden.example"}]},
    {"container": {"skills": [{"type": "anthropic", "skill_id": "pdf"}]}},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://forbidden.example"}}]}]},
    {"messages": [{"role": "user", "content": [{"type": "document", "source": {"type": "file", "file_id": "other-run"}}]}]},
    {"messages": [{"role": "user", "content": [{"type": "document", "source": {"type": "url", "url": "https://forbidden.example"}}]}]},
    {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image", "source": {"type": "url", "url": "https://forbidden.example"}}]}]}]},
    {"output_config": {"future_lookup": True}},
    {"output_config": {"task_budget": {"type": "tokens", "total": BUDGET_TOKENS, "future_lookup": True}}},
    {"future_capability": True},
])
def test_provider_capabilities_and_remote_references_are_rejected(addition: dict[str, Any]) -> None:
    """Unknown fields fail at their structured location, including nested tool-result content."""
    with pytest.raises(ValidationError):
        RequestPolicy().parse(json.dumps({**REQUEST_BODY, **addition}).encode())


def test_local_tools_thinking_and_ordinary_web_words_are_allowed() -> None:
    """Filter capabilities rather than phrases or URLs appearing as ordinary problem/tool text."""
    body = deepcopy(REQUEST_BODY)
    body["tools"] = [{"name": "Bash", "input_schema": {"type": "object"}}]
    body["messages"] = [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "reasoning", "signature": "signature"},
            {"type": "tool_use", "id": "t", "name": "Bash", "input": {"command": "echo web_search https://example.com"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "web_search https://example.com"}]},
    ]
    assert RequestPolicy().parse(json.dumps(body).encode()) == body


def test_locally_generated_images_and_structured_output_remain_usable() -> None:
    """Native file tools may return inline images, but cannot ask the provider to fetch a URL or file ID."""
    body = {**REQUEST_BODY, "messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "read", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "cG5n"}},
        ]},
    ]}], "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}},
                          "task_budget": {"type": "tokens", "total": BUDGET_TOKENS}}}
    assert RequestPolicy().parse(json.dumps(body).encode()) == body


@pytest.mark.parametrize("raw", [b'{"model":NaN}', b'{"model":Infinity}', b'not json'])
def test_malformed_and_nonstandard_json_is_rejected(raw: bytes) -> None:
    """Never forward a body interpreted differently by a stricter upstream JSON parser."""
    with pytest.raises(ValueError):
        RequestPolicy().parse(raw)


def test_unknown_beta_is_rejected() -> None:
    """New provider features require an explicit review before they become available to runs."""
    with pytest.raises(ValueError, match="unapproved"):
        RequestPolicy().validate_betas("future-search-feature")
