"""Repair system-content merging and preserve required Codex streaming in pinned LiteLLM."""

import py_compile

from harness.utils.constants import COUNT_INCREMENT, TEXT_ENCODING

from codex.constants import (
    PATCH_GLOB, PATCH_ORIGINAL, PATCH_REPLACEMENT, PATCH_TARGET,
    STREAMING_PATCH_GLOB, STREAMING_PATCH_ORIGINAL, STREAMING_PATCH_REPLACEMENT,
)


def _patch_source(pattern: str, original: str, replacement: str) -> None:
    """Require the exact affected source before changing it; unexpected versions fail the build."""
    target, = PATCH_TARGET.glob(pattern)
    source = target.read_text(encoding=TEXT_ENCODING)
    if source.count(original) != COUNT_INCREMENT:
        raise RuntimeError("LiteLLM source differs from the verified compatibility patch")
    target.write_text(source.replace(original, replacement), encoding=TEXT_ENCODING)
    py_compile.compile(str(target), doraise=True)


if __name__ == "__main__":
    _patch_source(PATCH_GLOB, PATCH_ORIGINAL, PATCH_REPLACEMENT)
    _patch_source(STREAMING_PATCH_GLOB, STREAMING_PATCH_ORIGINAL, STREAMING_PATCH_REPLACEMENT)
