"""Import a Codex login into an exclusively created, private LiteLLM credential file."""

import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from harness.utils.constants import TEXT_ENCODING
from jsonschema import ValidationError, validate

from codex.constants import (
    AUTH_DIRECTORY,
    AUTH_FILENAME,
    AUTH_SCHEMA,
    PRIVATE_FILE_MODE,
    REQUIRED_AUTH_FIELDS,
)


def read_credentials(source: Path) -> bytes:
    """Validate an explicit login file and project only the fields LiteLLM requires."""
    payload = json.loads(source.read_text(encoding=TEXT_ENCODING))
    try:
        validate(payload, AUTH_SCHEMA)
    except ValidationError:
        raise ValueError("Codex auth requires nonempty access_token, refresh_token, id_token, and account_id") from None
    tokens = cast(dict[str, str], payload["tokens"])
    return json.dumps({name: tokens[name] for name in REQUIRED_AUTH_FIELDS}).encode(TEXT_ENCODING)


def write_credentials(destination: Path, content: bytes) -> None:
    """Import a new identity or keep its existing refreshed credentials; never replace another account or a link."""
    if destination.is_symlink():
        raise FileExistsError("credential destination must not be a symlink")
    if destination.exists():
        existing = json.loads(destination.read_text(encoding=TEXT_ENCODING))
        if existing["account_id"] != json.loads(content)["account_id"]:
            raise FileExistsError("credential destination belongs to another account")
        return
    with NamedTemporaryFile(dir=destination.parent) as handle:
        os.fchmod(handle.fileno(), PRIVATE_FILE_MODE)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        os.link(handle.name, destination)


def import_credentials() -> None:
    """Receive the host's projected credentials over stdin inside the private volume helper."""
    write_credentials(AUTH_DIRECTORY / AUTH_FILENAME, sys.stdin.buffer.read())


if __name__ == "__main__":
    import_credentials()
