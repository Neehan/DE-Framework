"""Resolve a provider endpoint and numbered host-only credentials for the selected model."""

import re
from collections.abc import Mapping

from harness.proxy.constants import (
    AUTH_ENV_HEADERS,
    AUTH_TOKEN_ENV,
    PROVIDER_CREDENTIAL_COUNT,
)
from harness.proxy.models import ProviderConfig
from harness.sandbox.constants import PROVIDER_BASE_URL_ENV
from harness.utils.constants import INITIAL_COUNT, MUSE_PREFIX

from launcher.constants import (
    ANTHROPIC_URL,
    CLAUDE_PREFIX,
    CREDENTIAL_ENV_PATTERN,
    GPT_PREFIX,
    LITELLM_KEY_ENV,
    LITELLM_PREFIX,
    LITELLM_URL_ENV,
    META_KEY_ENV,
    META_URL,
    META_URL_ENV,
)
from launcher.models import ProviderRoute


def resolve_provider(model: str, env: Mapping[str, str]) -> ProviderRoute:
    """Route GPT through LiteLLM and collect one authentication family's numbered credentials."""
    provider_model = model.removeprefix(LITELLM_PREFIX)
    if model.startswith((LITELLM_PREFIX, GPT_PREFIX)):
        return _resolve_litellm(provider_model, env)
    elif model.startswith(MUSE_PREFIX):
        url, names, auth = env.get(META_URL_ENV, META_URL), (META_KEY_ENV,), AUTH_TOKEN_ENV
    elif model.startswith(CLAUDE_PREFIX):
        url, names, auth = env.get(PROVIDER_BASE_URL_ENV, ANTHROPIC_URL), tuple(AUTH_ENV_HEADERS), None
    else:
        raise ValueError("model must use claude-, muse-, gpt-, or the litellm/ route")
    if not provider_model:
        raise ValueError("provider model must not be empty")
    populated = []
    for name in names:
        credentials = _read_numbered_values(env, name)
        if credentials:
            populated.append((name, credentials))
    if len(populated) != PROVIDER_CREDENTIAL_COUNT:
        raise ValueError("provider requires credentials from exactly one authentication family")
    name, credentials = populated.pop()
    providers = tuple(ProviderConfig(url.rstrip("/"), auth or name, value) for value in credentials)
    return ProviderRoute(provider_model, providers, model.startswith(CLAUDE_PREFIX))


def _resolve_litellm(model: str, env: Mapping[str, str]) -> ProviderRoute:
    """Pair multiple gateways with one shared key, or multiple keys with one gateway."""
    urls = _read_numbered_values(env, LITELLM_URL_ENV)
    keys = _read_numbered_values(env, LITELLM_KEY_ENV)
    if not model or not urls or not keys:
        raise ValueError("LiteLLM requires a model, endpoint, and credential")
    if len(urls) > PROVIDER_CREDENTIAL_COUNT and len(keys) > PROVIDER_CREDENTIAL_COUNT:
        raise ValueError("multiple LiteLLM endpoints require one shared gateway key")
    providers = tuple(ProviderConfig(url.rstrip("/"), AUTH_TOKEN_ENV, key) for url in urls for key in keys)
    return ProviderRoute(model, providers, False)


def _read_numbered_values(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    """Read numbered endpoint or credential values in numeric order and remove duplicates."""
    pattern = re.compile(CREDENTIAL_ENV_PATTERN.format(name=re.escape(name)))
    entries = []
    for variable, value in env.items():
        match = pattern.fullmatch(variable)
        if match is not None and value.strip():
            index = int(match.group("suffix")) if match.group("suffix") is not None else INITIAL_COUNT
            entries.append((index, variable, value.strip()))
    return tuple(dict.fromkeys(value for _, _, value in sorted(entries)))
