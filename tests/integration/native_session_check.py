"""Run real native-session checkpoint and fork assertions inside a network-isolated test container."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from experiments.unaided import Unaided
from harness.self_refine.models import RefinementConfig, RefinementState, RunStatus
from harness.self_refine.recovery import Recovery
from harness.session.connection_manager import ConnectionManager
from harness.session.constants import (
    CLAUDE_CONFIG_DIR_ENV,
    DISALLOWED_TOOLS,
    SESSION_FILENAME,
    SETTING_SOURCES_ARG,
)
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import INITIAL_COUNT

from tests.constants import BUDGET_TOKENS, ONE_ROUND, PROBLEM
from tests.integration.constants import (
    BRANCH_NAMES,
    CLI_TIMEOUT_SECONDS,
    PROVIDER_MODEL,
)
from tests.integration.local_provider import LocalProvider
from tests.integration.provider_server import serve_provider


async def run_refinement(manager: SessionManager, state: RefinementState, options: ClaudeAgentOptions) -> RefinementState:
    """Exercise Unaided and the production session components with the actual installed SDK and CLI."""
    connection = ConnectionManager(options, ClaudeSDKClient)
    return await Unaided(Recovery(manager, connection, asyncio.sleep), RefinementConfig(ONE_ROUND, ONE_ROUND), None).run(state)


async def check_branch(source: SessionManager, directory: Path, options: ClaudeAgentOptions) -> str:
    """Fork saved native history, continue it, and verify fork intent clears after checkpointing."""
    target = SessionManager(directory, RunLock)
    forked = source.fork(target, None)
    try:
        target.open(forked)
        continued = target.continue_run(BUDGET_TOKENS)
    finally:
        target.close()
    result = await run_refinement(target, continued, options)
    assert result.status == RunStatus.FINISHED
    try:
        target.open(result)
        assert not target.session_state.fork_session
        session_id = target.session_state.session_id
        assert session_id is not None
        return session_id
    finally:
        target.close()


async def check_forks(directory: Path, options: ClaudeAgentOptions) -> None:
    """Require independent native branch IDs and an unchanged source checkpoint."""
    source_path = directory / "source"
    source = SessionManager(source_path, RunLock)
    state = await run_refinement(source, RefinementState(PROBLEM, BUDGET_TOKENS, None), options)
    assert state.status == RunStatus.FINISHED
    original = (source_path / SESSION_FILENAME).read_bytes()
    try:
        source.open(state)
        source_id = source.session_state.session_id
        branches = [await check_branch(source, directory / name, options) for name in BRANCH_NAMES]
        assert len(set(branches)) == len(BRANCH_NAMES)
        assert source_id not in branches
        assert (source_path / SESSION_FILENAME).read_bytes() == original
    finally:
        source.close()


async def check_managed_policy(directory: Path, options: ClaudeAgentOptions) -> None:
    """Request blocked tools directly from the SDK without ConnectionManager's explicit denials."""
    policy_options = replace(
        options, cwd=directory, tools=["Bash", *sorted(DISALLOWED_TOOLS)],
        setting_sources=[], extra_args={SETTING_SOURCES_ARG: ""},
        env={**options.env, CLAUDE_CONFIG_DIR_ENV: str(directory)},
    )
    async with ClaudeSDKClient(options=policy_options) as client:
        await client.query(PROBLEM)
        async for _message in client.receive_response():
            pass


def main() -> None:
    """Serve only a local fake provider and retain SDK debug output to exercise debug-link checkpointing."""
    with serve_provider(LocalProvider) as (url, requests):
        options = ClaudeAgentOptions(model=PROVIDER_MODEL, env={
            "ANTHROPIC_BASE_URL": url,
            "ANTHROPIC_API_KEY": "local-test-only", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DEBUG_SDK": "1",
        })
        with TemporaryDirectory() as directory:
            asyncio.run(asyncio.wait_for(check_forks(Path(directory), options), CLI_TIMEOUT_SECONDS))
        with TemporaryDirectory() as directory:
            asyncio.run(asyncio.wait_for(check_managed_policy(Path(directory), options), CLI_TIMEOUT_SECONDS))
        offered_tools = {tool["name"] for tool in requests[-ONE_ROUND]["tools"]}
        assert "Bash" in offered_tools
        assert not offered_tools.intersection(DISALLOWED_TOOLS)
        assert "expert mathematician" in json.dumps(requests[INITIAL_COUNT]["system"])
        print("Native checkpoints, independent forks, shared system prompt, and managed tool policy passed.")


if __name__ == "__main__":
    main()
