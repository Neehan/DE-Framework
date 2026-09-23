"""CLI arrays, strict local selection, and single-provider routing."""

from dataclasses import asdict

import pytest
from experiments.constants import ALTERNATE_ROLE, REFERENCE_ROLE
from harness.proxy.constants import API_KEY_ENV, AUTH_TOKEN_ENV, OAUTH_TOKEN_ENV
from harness.utils.constants import DEFAULT_MAX_CONCURRENCY, OUTPUT_TOKENS_PER_BLOCK
from launcher.cli import parse_arguments
from launcher.dataset import Dataset
from launcher.models import LaunchConfig
from launcher.provider import resolve_provider

from tests.launcher.constants import (
    ALTERNATE_SKETCH,
    CLI_ARGUMENTS,
    REFERENCE_SENTINEL,
    REFERENCE_SKETCH,
    TEST_ROUTE_KEY,
    TEST_ROUTE_URL,
)


@pytest.mark.parametrize(("experiment", "multiplier"), [
    ("unaided", 1), ("no-sketch", 1), ("oracle-execution", 8), ("oracle-sketch", 1),
    ("alt-sketch", 1), ("continue-at-3x", 4), ("continue-oracle-at-3x", 4),
])
def test_experiment_budget_defaults_and_fixed_budget_validation(experiment: str, multiplier: int) -> None:
    """Named arms supply their fixed allowance and reject unsupported CLI overrides before execution."""
    config = parse_arguments([*CLI_ARGUMENTS, "--experiment", experiment])
    assert config.compute_multiplier_k == multiplier
    assert config.budget_tokens == multiplier * OUTPUT_TOKENS_PER_BLOCK
    assert config.experiment_directory == f"{experiment}-{multiplier}x"
    if experiment != "unaided":
        with pytest.raises(SystemExit):
            parse_arguments([*CLI_ARGUMENTS, "--experiment", experiment, "--compute-multiplier-k", "2"])


@pytest.mark.parametrize(("role", "expected", "excluded"), [
    (REFERENCE_ROLE, REFERENCE_SKETCH, ALTERNATE_SKETCH),
    (ALTERNATE_ROLE, ALTERNATE_SKETCH, REFERENCE_SKETCH),
])
def test_dataset_projects_only_selected_sketch_and_validates_selected_rows(
    dataset: Dataset, role: str, expected: str, excluded: str,
) -> None:
    """An unselected missing sketch is harmless; selected missing sketches fail without exposing proofs or steps."""
    selected = dataset.load("aobench", ["p1"], None, role)
    assert selected[0].sketch == expected
    assert excluded not in repr(selected)
    assert REFERENCE_SENTINEL not in repr(selected)
    with pytest.raises(ValueError, match=f"p2 requires a {role} sketch"):
        dataset.load("aobench", ["p2"], None, role)


def test_seed_arrays_and_compute_budget_have_one_source(launch_config: LaunchConfig) -> None:
    """Explicit seed IDs have no predefined roster; one multiplier determines both budget and output label."""
    assert launch_config.seeds == [1, 3, 24]
    assert launch_config.max_concurrency == DEFAULT_MAX_CONCURRENCY
    assert launch_config.budget_tokens == OUTPUT_TOKENS_PER_BLOCK
    expanded = parse_arguments([*CLI_ARGUMENTS, "--compute-multiplier-k", "8"])
    assert expanded.budget_tokens == OUTPUT_TOKENS_PER_BLOCK * expanded.compute_multiplier_k
    assert expanded.experiment_directory == "unaided-8x"


@pytest.mark.parametrize("addition", [
    ["--seeds", "1-24"], ["--seeds", "1,2"], ["--seeds", "1", "1"], ["--seeds", "-1"],
    ["--compute-multiplier-k", "0"], ["--compute-multiplier-k", "9"], ["--max-concurrency", "0"],
    ["--experiment", "unknown"], ["--dataset", "unknown"], ["--model", "../escape"],
])
def test_invalid_cli_is_rejected(addition: list[str]) -> None:
    """Fail before Docker for unsupported experiments, invalid settings, and ambiguous seed syntax."""
    with pytest.raises(SystemExit):
        parse_arguments([*CLI_ARGUMENTS, *addition])


def test_dataset_retains_only_permitted_fields_and_applies_filters(dataset: Dataset) -> None:
    """References never survive projection into the problem objects used by the scheduler."""
    selected = dataset.load("aobench", ["p1"], "algebra", None)
    assert [problem.problem_id for problem in selected] == ["p1"]
    assert all(problem.sketch is None for problem in selected)
    assert set(asdict(selected.pop())) == {"problem_id", "statement", "domain", "sketch"}
    assert REFERENCE_SENTINEL not in repr(dataset.load("aobench", None, None, None))


@pytest.mark.parametrize(("ids", "domain"), [(["missing"], None), (None, "unknown"), (["p2"], "algebra")])
def test_invalid_problem_selection_fails(dataset: Dataset, ids: list[str] | None, domain: str | None) -> None:
    """Unknown IDs and incompatible domain selections cannot silently become partial runs."""
    with pytest.raises(ValueError):
        dataset.load("aobench", ids, domain, None)


@pytest.mark.parametrize(("model", "environment", "auth", "alias"), [
    ("claude-test", {API_KEY_ENV: TEST_ROUTE_KEY}, API_KEY_ENV, "claude-test"),
    ("claude-test", {OAUTH_TOKEN_ENV: TEST_ROUTE_KEY}, OAUTH_TOKEN_ENV, "claude-test"),
    ("litellm/gpt-test", {"LITELLM_BASE_URL": TEST_ROUTE_URL, "LITELLM_API_KEY": TEST_ROUTE_KEY}, AUTH_TOKEN_ENV, "gpt-test"),
    ("gpt-test", {"LITELLM_BASE_URL": TEST_ROUTE_URL, "LITELLM_API_KEY": TEST_ROUTE_KEY}, AUTH_TOKEN_ENV, "gpt-test"),
    ("litellm/claude-test", {"LITELLM_BASE_URL": TEST_ROUTE_URL, "LITELLM_API_KEY": TEST_ROUTE_KEY}, AUTH_TOKEN_ENV, "claude-test"),
    ("muse-test", {"META_BASE_URL": TEST_ROUTE_URL, "META_API_KEY": TEST_ROUTE_KEY}, AUTH_TOKEN_ENV, "muse-test"),
])
def test_provider_routes_keep_one_credential_host_side(model: str, environment: dict[str, str], auth: str, alias: str) -> None:
    """Strip only the routing prefix and pass a minimal credential environment to the host proxy."""
    route = resolve_provider(model, environment)
    assert route.model == alias
    assert route.use_anthropic_resets == model.startswith("claude-")
    assert route.providers[0].auth_variable == auth
    assert route.providers[0].credential == TEST_ROUTE_KEY
    assert TEST_ROUTE_KEY not in repr(route)


@pytest.mark.parametrize("environment", [{}, {API_KEY_ENV: TEST_ROUTE_KEY, OAUTH_TOKEN_ENV: TEST_ROUTE_KEY}])
def test_missing_or_multiple_credentials_fail(environment: dict[str, str]) -> None:
    """Reject missing credentials and ambiguous authentication families."""
    with pytest.raises(ValueError):
        resolve_provider("claude-test", environment)


@pytest.mark.parametrize(("model", "name", "settings"), [
    ("claude-test", API_KEY_ENV, {}),
    ("claude-test", OAUTH_TOKEN_ENV, {}),
    ("muse-test", "META_API_KEY", {}),
    ("litellm/gpt-test", "LITELLM_API_KEY", {"LITELLM_BASE_URL": TEST_ROUTE_URL}),
])
def test_numbered_credentials_are_ordered_deduplicated_and_host_only(
    model: str, name: str, settings: dict[str, str],
) -> None:
    """Numeric suffix ordering and duplicate removal apply to every supported provider route."""
    route = resolve_provider(model, {
        **settings, name: "first", f"{name}_10": "tenth", f"{name}_2": "second",
        f"{name}_3": "first", f"{name}_4": "  ", f"{name}_EXPIRES_AT": "not-a-credential",
    })
    assert [provider.credential for provider in route.providers] == ["first", "second", "tenth"]
    assert "tenth" not in repr(route)


def test_codex_endpoints_share_a_key_without_collapsing_accounts() -> None:
    """Numbered gateways preserve endpoint identity and reject ambiguous key pairing."""
    environment = {"LITELLM_BASE_URL_2": "http://127.0.0.1:4202", "LITELLM_BASE_URL_1": "http://127.0.0.1:4201",
                   "LITELLM_API_KEY": TEST_ROUTE_KEY}
    route = resolve_provider("gpt-test", environment)
    assert [provider.url for provider in route.providers] == ["http://127.0.0.1:4201", "http://127.0.0.1:4202"]
    assert all(provider.credential == TEST_ROUTE_KEY for provider in route.providers)
    assert not route.use_anthropic_resets
    with pytest.raises(ValueError, match="shared gateway key"):
        resolve_provider("gpt-test", {**environment, "LITELLM_API_KEY_2": "ambiguous-key"})
