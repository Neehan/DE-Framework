"""Host dotenv loading preserves exported credentials and stays outside worker setup."""

import os
from pathlib import Path

import pytest
from harness.sandbox.constants import SUCCESS_EXIT_CODE
from launcher import cli
from launcher.constants import LITELLM_KEY_ENV, LITELLM_URL_ENV
from launcher.models import LaunchConfig

from tests.launcher.constants import CLI_ARGUMENTS, TEST_ROUTE_KEY, TEST_ROUTE_URL


def test_repository_dotenv_loads_without_overriding_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a disposable .env and injected execution, never reading repository credentials or contacting Docker."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{LITELLM_URL_ENV}={TEST_ROUTE_URL}\n{LITELLM_KEY_ENV}=ignored-file-key\n")
    monkeypatch.setattr(cli, "DOTENV_FILE", dotenv)
    monkeypatch.setattr(cli.sys, "argv", ["launcher", *CLI_ARGUMENTS])
    monkeypatch.delenv(LITELLM_URL_ENV, raising=False)
    monkeypatch.setenv(LITELLM_KEY_ENV, TEST_ROUTE_KEY)

    async def launch(config: LaunchConfig) -> int:
        """Verify the actual process boundary sees file defaults and exported credential precedence."""
        assert os.environ[LITELLM_URL_ENV] == TEST_ROUTE_URL
        assert os.environ[LITELLM_KEY_ENV] == TEST_ROUTE_KEY
        return SUCCESS_EXIT_CODE

    with pytest.raises(SystemExit) as exit_status:
        cli.run_cli(cli.parse_arguments, launch)
    assert exit_status.value.code == SUCCESS_EXIT_CODE
