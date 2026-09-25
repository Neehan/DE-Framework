"""Provider diagnostic coverage independent of SDK startup and streaming fixtures."""

import pytest
from harness.session.errors import is_transient_provider_error

from tests.constants import TRANSIENT_DIAGNOSTICS, TRANSIENT_PROVIDER_ERROR


@pytest.mark.parametrize("diagnostic", (TRANSIENT_PROVIDER_ERROR, *TRANSIENT_DIAGNOSTICS))
def test_transient_provider_diagnostics(diagnostic: str) -> None:
    """Recognize every supported transient diagnostic, regardless of capitalization."""
    assert is_transient_provider_error(diagnostic.upper())


@pytest.mark.parametrize("diagnostic", ("Invalid API key", "bad settings", ""))
def test_permanent_or_empty_diagnostics(diagnostic: str) -> None:
    """Do not classify permanent or unexplained failures as recoverable."""
    assert not is_transient_provider_error(diagnostic)
