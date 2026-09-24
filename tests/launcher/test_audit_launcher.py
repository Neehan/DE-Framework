"""Independent audit stages, immutable solver inputs, and duplicate-launch ownership."""

import json
from asyncio import Event, create_task, timeout
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, create_autospec

import pytest
from audit.constants import (
    AUDIT_FILENAME,
    AUDIT_WORKSPACE_DIRECTORY,
    COMPILED_AUDIT_FILENAME,
    CORRECTNESS,
    EMPTY_AUDIT_RECORD,
    RESULT_FILENAME,
    STEP_RECOGNITION,
)
from audit.models import AuditReference, AuditRequest
from audit.registry import AUDITS
from harness.sandbox.sandbox import Sandbox
from harness.self_refine.models import RefinementState, RunStatus
from harness.session.constants import SESSION_FILENAME
from harness.session.run_lock import RunLock
from harness.session.session_manager import SessionManager
from harness.utils.constants import DEFAULT_JUDGE_MODEL
from launcher.cli import parse_arguments, parse_audit_arguments
from launcher.dataset import Dataset
from launcher.launcher.audit_launcher import AuditLauncher
from launcher.models import AuditConfig, Problem
from launcher.provider import resolve_provider

from tests.audit.constants import (
    AUDIT_MODEL,
    COMPLETE_AUDIT,
    CORRECTNESS_RESULT,
    REFERENCE,
    STEP_RESULT,
    STEPS,
    SUBMISSION,
)
from tests.audit.helpers import save_audit
from tests.constants import (
    ASYNC_TEST_TIMEOUT_SECONDS,
    FIRST_SOLUTION_FILENAME,
    ONE_ROUND,
    PROBLEM,
)
from tests.launcher.constants import (
    CLI_ARGUMENTS,
    PROBLEM_ROWS,
    TEST_ROUTE_KEY,
    TEST_ROUTE_URL,
)


@pytest.fixture
def audit_config() -> AuditConfig:
    """Use production audit defaults with a single selected seed."""
    return parse_audit_arguments([*CLI_ARGUMENTS, "--seeds", str(ONE_ROUND)])


@pytest.fixture
def saved_solver(audit_config: AuditConfig, tmp_path: Path) -> tuple[Problem, Path]:
    """Keep the authoritative solution different from its export to detect wrong input selection."""
    problem = Problem("p1", PROBLEM, "algebra", None)
    source = tmp_path / audit_config.dataset / audit_config.model_directory / audit_config.experiment_directory / problem.problem_id / "seed_1"
    state = RefinementState(PROBLEM, audit_config.budget_tokens, None)
    manager = SessionManager(source, RunLock)
    manager.open(state)
    state.status, state.solution = RunStatus.FINISHED, SUBMISSION
    state.solution_checkpoints = {ONE_ROUND: SUBMISSION}
    manager.checkpoint()
    manager.close()
    (source / FIRST_SOLUTION_FILENAME).write_text("stale export must not be audited")
    return problem, source


async def test_stage_failure_resumes_recognition_without_regrading(
    audit_config: AuditConfig, saved_solver: tuple[Problem, Path], tmp_path: Path,
) -> None:
    """Persist correctness even when recognition fails; retry only the missing stage with no verdict leakage."""
    problem, source = saved_solver
    original = (source / SESSION_FILENAME).read_bytes()
    seen = []
    sandbox = create_autospec(Sandbox, instance=True)

    async def execute(directory: Path, content: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Validate stage-specific inputs and inject the first recognition failure."""
        request = AuditRequest(**json.loads(content))
        seen.append(request.kind)
        assert request.solution == SUBMISSION and request.reference == REFERENCE
        assert request.model == AUDIT_MODEL and CORRECTNESS_RESULT["note"] not in content
        assert request.steps == (STEPS if request.kind == STEP_RECOGNITION else None)
        if seen == [CORRECTNESS, STEP_RECOGNITION]:
            raise ConnectionError("recognition failed")
        with ownership(directory):
            verdict = CORRECTNESS_RESULT if request.kind == CORRECTNESS else STEP_RESULT
            (directory / RESULT_FILENAME).write_text(json.dumps(verdict))

    sandbox.run.side_effect = execute
    launcher = AuditLauncher(audit_config, lambda: nullcontext(lambda: sandbox), tmp_path, {problem.problem_id: AuditReference(REFERENCE, STEPS)})
    first = await launcher.run([problem], AUDIT_MODEL)
    compiled = source.parent.parent / COMPILED_AUDIT_FILENAME
    assert compiled.read_bytes() == b""
    partial = json.loads((source / AUDIT_FILENAME).read_text())["checkpoints"]["1x"]
    assert partial == {"correctness": CORRECTNESS_RESULT, "correctness_audit_model": audit_config.audit_model}
    next_config = replace(audit_config, audit_model=f"litellm/{AUDIT_MODEL}")
    launcher = AuditLauncher(next_config, lambda: nullcontext(lambda: sandbox), tmp_path, {problem.problem_id: AuditReference(REFERENCE, STEPS)})
    resumed = await launcher.run([problem], AUDIT_MODEL)
    skipped = await launcher.run([problem], AUDIT_MODEL)
    assert first.failed == resumed.completed == skipped.skipped == ONE_ROUND
    assert seen == [CORRECTNESS, STEP_RECOGNITION, STEP_RECOGNITION]
    assert (source / SESSION_FILENAME).read_bytes() == original
    assert json.loads((source / AUDIT_FILENAME).read_text())["checkpoints"]["1x"] == {
        **partial, "step_recognition": STEP_RESULT, "step_audit_model": next_config.audit_model,
    }
    assert not (source / AUDIT_WORKSPACE_DIRECTORY).exists()
    assert json.loads(compiled.read_text())["step_audit_model"] == next_config.audit_model


async def test_duplicate_audit_skips_while_other_owner_runs(
    audit_config: AuditConfig, saved_solver: tuple[Problem, Path], tmp_path: Path,
) -> None:
    """The sequence lock excludes overlapping launchers without blocking independent scheduling."""
    problem, source = saved_solver
    entered, release = Event(), Event()
    sandbox = create_autospec(Sandbox, instance=True)

    async def execute(directory: Path, content: str, ownership: Callable[[Path], AbstractContextManager[object]]) -> None:
        """Hold the first stage until the duplicate launcher has checked ownership."""
        with ownership(directory):
            entered.set()
            await release.wait()
            result = CORRECTNESS_RESULT if directory.name == CORRECTNESS else STEP_RESULT
            (directory / RESULT_FILENAME).write_text(json.dumps(result))

    sandbox.run.side_effect = execute
    launcher = AuditLauncher(audit_config, lambda: nullcontext(lambda: sandbox), tmp_path, {problem.problem_id: AuditReference(REFERENCE, STEPS)})
    async with timeout(ASYNC_TEST_TIMEOUT_SECONDS):
        first = create_task(launcher.run([problem], AUDIT_MODEL))
        await entered.wait()
        repeated = await launcher.run([problem], AUDIT_MODEL)
        release.set()
        result = await first
    assert repeated.skipped == result.completed == ONE_ROUND
    assert sandbox.run.await_count == len(AUDITS)


@pytest.mark.parametrize("status", [RunStatus.RUNNING, RunStatus.PAUSED])
async def test_incomplete_solver_is_not_audited(audit_config: AuditConfig, tmp_path: Path, status: RunStatus) -> None:
    """Only final solver artifacts enter the judging pipeline; incomplete work needs no Docker."""
    problem = Problem("p1", PROBLEM, "algebra", None)
    source = tmp_path / audit_config.dataset / audit_config.model_directory / audit_config.experiment_directory / problem.problem_id / "seed_1"
    manager = SessionManager(source, RunLock)
    state = manager.open(RefinementState(PROBLEM, audit_config.budget_tokens, None))
    state.status = status
    manager.checkpoint()
    manager.close()
    environment = Mock(side_effect=AssertionError("incomplete solver must not start Docker"))
    result = await AuditLauncher(audit_config, environment, tmp_path, {}).run([problem], AUDIT_MODEL)
    assert result.skipped == ONE_ROUND and not result.failed
    environment.assert_not_called()


@pytest.mark.parametrize("solution", [None, "", "  \n"])
async def test_empty_solution_skips_both_judges_and_stays_completed(
    audit_config: AuditConfig, saved_solver: tuple[Problem, Path], tmp_path: Path, solution: str | None,
) -> None:
    """No within-budget submission deterministically scores zero and leaves recognition unobserved."""
    problem, source = saved_solver
    manager = SessionManager(source, RunLock)
    state = manager.open(RefinementState(PROBLEM, audit_config.budget_tokens, None))
    state.status, state.solution = RunStatus.EXHAUSTED, solution
    state.solution_checkpoints = {ONE_ROUND: solution}
    manager.checkpoint()
    manager.close()
    archive = (source / SESSION_FILENAME).read_bytes()
    environment = Mock(side_effect=AssertionError("empty submission must not start Docker"))
    launcher = AuditLauncher(audit_config, environment, tmp_path, {})
    for _ in range(len(AUDITS)):
        result = await launcher.run([problem], AUDIT_MODEL)
        assert result.skipped == ONE_ROUND and not result.failed
    assert json.loads((source / AUDIT_FILENAME).read_text()) == {"checkpoints": {"1x": EMPTY_AUDIT_RECORD}}
    assert (source / SESSION_FILENAME).read_bytes() == archive
    environment.assert_not_called()


async def test_corrupt_audit_is_not_silently_skipped_or_overwritten(
    audit_config: AuditConfig, saved_solver: tuple[Problem, Path], tmp_path: Path,
) -> None:
    """Existing invalid results fail visibly and remain available for diagnosis."""
    problem, source = saved_solver
    invalid = {"correctness": STEP_RESULT, "correctness_audit_model": audit_config.audit_model}
    (source / AUDIT_FILENAME).write_text(json.dumps(invalid))
    environment = Mock(side_effect=AssertionError("corrupt audit must not start Docker"))
    with pytest.raises(ValueError, match="invalid audit record"):
        await AuditLauncher(audit_config, environment, tmp_path, {problem.problem_id: AuditReference(REFERENCE, STEPS)}).run([problem], AUDIT_MODEL)
    environment.assert_not_called()
    assert json.loads((source / AUDIT_FILENAME).read_text()) == invalid


@pytest.mark.parametrize("nested", [False, True])
async def test_audit_workspace_cannot_redirect_mount_to_other_runs(
    audit_config: AuditConfig, saved_solver: tuple[Problem, Path], tmp_path: Path, nested: bool,
) -> None:
    """Reject solver-created links before the sandbox resolves its bind mount or touches another run."""
    problem, source = saved_solver
    outside = tmp_path / "other-seed"
    outside.mkdir()
    sentinel = outside / FIRST_SOLUTION_FILENAME
    sentinel.write_text(SUBMISSION)
    workspace = source / AUDIT_WORKSPACE_DIRECTORY
    if nested:
        workspace.mkdir()
        workspace = workspace / "1x"
    workspace.symlink_to(outside, target_is_directory=True)
    sandbox = create_autospec(Sandbox, instance=True)
    launcher = AuditLauncher(audit_config, lambda: nullcontext(lambda: sandbox), tmp_path, {problem.problem_id: AuditReference(REFERENCE, STEPS)})
    result = await launcher.run([problem], AUDIT_MODEL)
    assert result.failed == ONE_ROUND
    sandbox.run.assert_not_awaited()
    assert sentinel.read_text() == SUBMISSION


async def test_completed_audit_cleans_scratch_left_after_publication(
    audit_config: AuditConfig, saved_solver: tuple[Problem, Path], tmp_path: Path,
) -> None:
    """A crash after the last atomic save leaves only disposable files, never another judging task."""
    problem, source = saved_solver
    record = {"correctness": CORRECTNESS_RESULT, "correctness_audit_model": audit_config.audit_model,
              "step_recognition": STEP_RESULT, "step_audit_model": audit_config.audit_model}
    record = {"checkpoints": {"1x": record}}
    (source / AUDIT_FILENAME).write_text(json.dumps(record))
    save_audit(source.parent.parent / "unselected-problem/seed_24", {"1x": COMPLETE_AUDIT})
    scratch = source / AUDIT_WORKSPACE_DIRECTORY / STEP_RECOGNITION
    scratch.mkdir(parents=True)
    (scratch / RESULT_FILENAME).write_text(json.dumps(STEP_RESULT))
    environment = Mock(side_effect=AssertionError("completed audit must not start Docker"))
    result = await AuditLauncher(audit_config, environment, tmp_path, {problem.problem_id: AuditReference(REFERENCE, STEPS)}).run([problem], AUDIT_MODEL)
    assert result.skipped == ONE_ROUND
    assert not (source / AUDIT_WORKSPACE_DIRECTORY).exists()
    assert json.loads((source / AUDIT_FILENAME).read_text()) == record
    compiled = source.parent.parent / COMPILED_AUDIT_FILENAME
    assert [json.loads(line)["problem_id"] for line in compiled.read_text().splitlines()] == [problem.problem_id, "unselected-problem"]


def test_shared_cli_default_judge_and_provider_route() -> None:
    """The audit selector preserves solver result paths while routing the centrally configured judge through LiteLLM."""
    run, audit = parse_arguments(CLI_ARGUMENTS), parse_audit_arguments(CLI_ARGUMENTS)
    assert audit.audit_model == DEFAULT_JUDGE_MODEL == "litellm/gpt-5.6-sol"
    assert audit.model_directory == run.model_directory and audit.experiment_directory == run.experiment_directory
    route = resolve_provider(audit.audit_model, {"LITELLM_BASE_URL": TEST_ROUTE_URL, "LITELLM_API_KEY": TEST_ROUTE_KEY})
    assert route.model == "gpt-5.6-sol"
    with pytest.raises(SystemExit):
        parse_arguments([*CLI_ARGUMENTS, "--audit-model", DEFAULT_JUDGE_MODEL])
    with pytest.raises(SystemExit):
        parse_audit_arguments([*CLI_ARGUMENTS, "--audit-model", f"litellm/{run.model}"])


@pytest.mark.parametrize("valid", [True, False])
def test_local_audit_references_require_exactly_three_steps(tmp_path: Path, valid: bool) -> None:
    """Reference projection uses the fixed reference role and rejects malformed selected outlines before judging."""
    row = json.loads(json.dumps(PROBLEM_ROWS[0]))
    row["solutions"][0]["steps"] = STEPS if valid else []
    (tmp_path / "aobench.jsonl").write_text(json.dumps(row))
    dataset = Dataset(tmp_path)
    problems = dataset.load("aobench", None, None, None)
    if valid:
        reference = dataset.load_references("aobench", problems)["p1"]
        assert reference.steps == STEPS
        assert reference.solution == row["solutions"][0]["solution"]
    else:
        with pytest.raises(ValueError, match="three nonempty"):
            dataset.load_references("aobench", problems)
