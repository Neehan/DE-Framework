"""Credential projection and atomic, non-destructive account import."""

import json
from pathlib import Path
from stat import S_IMODE

import pytest
from codex.auth import read_credentials, write_credentials
from codex.constants import PRIVATE_FILE_MODE, REQUIRED_AUTH_FIELDS

from tests.codex.constants import AUTH_TOKENS


def test_import_projects_private_tokens_and_refuses_replacement(tmp_path: Path) -> None:
    """An existing account and the user's source file survive repeat imports unchanged."""
    source, destination = tmp_path / "source.json", tmp_path / "auth.json"
    source.write_text(json.dumps({"tokens": AUTH_TOKENS, "irrelevant": "discard"}))
    original = source.read_bytes()
    content = read_credentials(source)
    write_credentials(destination, content)
    assert json.loads(destination.read_bytes()) == AUTH_TOKENS
    assert S_IMODE(destination.stat().st_mode) == PRIVATE_FILE_MODE
    write_credentials(destination, json.dumps({**AUTH_TOKENS, "access_token": "old-token"}).encode())
    with pytest.raises(FileExistsError, match="another account"):
        write_credentials(destination, json.dumps({**AUTH_TOKENS, "account_id": "different-account"}).encode())
    assert destination.read_bytes() == content and source.read_bytes() == original
    assert set(tmp_path.iterdir()) == {source, destination}


def test_import_refuses_link_even_when_identity_matches(tmp_path: Path) -> None:
    """A matching identity cannot authorize writing through a linked credential destination."""
    existing = tmp_path / "existing.json"
    content = json.dumps(AUTH_TOKENS).encode()
    existing.write_bytes(content)
    destination = tmp_path / "auth.json"
    destination.symlink_to(existing)
    with pytest.raises(FileExistsError, match="symlink"):
        write_credentials(destination, content)
    assert existing.read_bytes() == content


@pytest.mark.parametrize("field", REQUIRED_AUTH_FIELDS)
def test_invalid_auth_never_leaks_tokens(tmp_path: Path, field: str) -> None:
    """Report missing fields without jsonschema's usual secret-bearing instance dump."""
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps({"tokens": {**AUTH_TOKENS, field: ""}}))
    with pytest.raises(ValueError) as caught:
        read_credentials(source)
    assert not any(token in str(caught.value) for token in AUTH_TOKENS.values())
