"""Read structured conversation blocks captured from the native CLI."""

from collections.abc import Iterator
from typing import Any


def iter_message_blocks(messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield blocks from structured messages; plain-text messages contain no tool calls or results."""
    for message in messages:
        match message["content"]:
            case str():
                continue
            case list(blocks):
                yield from blocks
            case _:
                raise AssertionError("expected text or content blocks in native message")
