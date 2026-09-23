"""Repair the reproduced list-valued system prompt bug in the pinned LiteLLM image."""

import py_compile

from harness.utils.constants import COUNT_INCREMENT, TEXT_ENCODING

from codex.constants import PATCH_GLOB, PATCH_ORIGINAL, PATCH_REPLACEMENT, PATCH_TARGET


def patch_system_content() -> None:
    """Require the exact affected source before changing it; unexpected versions fail the build."""
    target, = PATCH_TARGET.glob(PATCH_GLOB)
    source = target.read_text(encoding=TEXT_ENCODING)
    if source.count(PATCH_ORIGINAL) != COUNT_INCREMENT:
        raise RuntimeError("LiteLLM source differs from the verified compatibility patch")
    target.write_text(source.replace(PATCH_ORIGINAL, PATCH_REPLACEMENT), encoding=TEXT_ENCODING)
    py_compile.compile(str(target), doraise=True)


if __name__ == "__main__":
    patch_system_content()
