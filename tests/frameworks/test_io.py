"""Verify all-seed loading, minimum sample counts, simple filters, and saved outputs."""

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from frameworks.discovery_execution.discovery_execution import DiscoveryExecution
from frameworks.discovery_execution.regularized_discovery_execution import RegularizedDiscoveryExecution
from frameworks.geometric.regularized_simple_geometric import RegularizedSimpleGeometric
from frameworks.runner import FrameworkRunner
from frameworks.models import BasePrediction, InputSelection
from frameworks.utils.dataloader import DataLoader
from harness.session.constants import LOCK_FILENAME
from jsonschema import ValidationError
from launcher.frameworks import parse_arguments
from launcher.models import encode_model_directory

from tests.frameworks.constants import DATASET, EXPECTED_ALLOCATIONS, EXTRA_SEED, MODEL, ROWS
from tests.frameworks.helpers import save_bank


def test_loader_uses_all_seeds_and_first_success(banks: Path, data_loader: DataLoader, selection: InputSelection) -> None:
    """Noncontiguous extra seeds count too; later proof regressions never undo the first success."""
    assert [asdict(row) for row in data_loader.load(selection)] == [asdict(row) for row in ROWS]
    root = banks / "results" / DATASET / encode_model_directory(MODEL)
    for experiment in ["unaided-1x", "oracle-execution-8x"]:
        path = root / experiment / "audit.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        extra = [{**row, "seed": EXTRA_SEED} for row in records if row["problem_id"] == "p1" and row["seed"] == 1]
        save_bank(path, records + extra)
    row = data_loader.load(selection)[0]
    assert row.short_attempts == 25 and row.short_successes == ROWS[0].short_successes + 1
    assert row.oracle_attempts == 5
    assert row.oracle_successes == [count + 1 for count in ROWS[0].oracle_successes]


def test_seed_minimums_and_incomplete_oracle_runs_produce_one_warning(
    banks: Path, data_loader: DataLoader, selection: InputSelection, caplog: pytest.LogCaptureFixture,
) -> None:
    """23 short or two complete oracle seeds cause skips; exactly three complete oracle seeds remain eligible."""
    root = banks / "results" / DATASET / encode_model_directory(MODEL)
    short = root / "unaided-1x/audit.jsonl"
    records = [json.loads(line) for line in short.read_text().splitlines()]
    save_bank(short, [row for row in records if not (row["problem_id"] == "p1" and row["seed"] == 24)])
    oracle = root / "oracle-execution-8x/audit.jsonl"
    records = [json.loads(line) for line in oracle.read_text().splitlines()]
    save_bank(oracle, [row for row in records if not (
        (row["problem_id"] == "p2" and row["seed"] >= 3)
        or (row["problem_id"] == "p3" and row["seed"] == 4 and row["checkpoint_multiplier_k"] == 8)
    )])
    with caplog.at_level(logging.WARNING):
        observations = data_loader.load(selection)
    assert [row.problem_id for row in observations] == ["p3"]
    assert observations[0].short_attempts == 24 and observations[0].oracle_attempts == 3
    assert len(caplog.records) == 1
    warning = caplog.records[0].getMessage()
    assert "aobench/p1" in warning and "aobench/p2" in warning and "p3" not in warning
    assert "unaided=23" in warning and "oracle=2" in warning


def test_missing_banks_skip_all_with_one_warning_and_no_outputs(
    banks: Path, runner: FrameworkRunner, caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty audit export behaves like insufficient seeds, including when both datasets are selected."""
    for dataset in [DATASET, "imoproofbench"]:
        path = banks / "results" / dataset / encode_model_directory(MODEL) / "oracle-execution-8x/audit.jsonl"
        path.rename(path.with_suffix(".unused"))
    with caplog.at_level(logging.WARNING):
        assert runner.run(parse_arguments(["--model", MODEL])) == []
    assert len(caplog.records) == 1
    assert "aobench/p1" in caplog.text and "imoproofbench/imo_p1" in caplog.text
    assert not list((banks / "results").glob("*/*/frameworks"))


@pytest.mark.parametrize("corruption", ["duplicate", "identity", "score"])
def test_corrupt_audits_fail_instead_of_becoming_skipped_data(
    banks: Path, data_loader: DataLoader, selection: InputSelection, corruption: str,
) -> None:
    """Only insufficient coverage is skippable; invalid measurements remain errors."""
    path = banks / "results" / DATASET / encode_model_directory(MODEL) / "oracle-execution-8x/audit.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "identity":
        rows[0]["solver_model"] = "another-model"
    else:
        rows[0]["correctness"]["score"] = True
    save_bank(path, rows)
    with pytest.raises((ValueError, ValidationError)):
        data_loader.load(selection)


def test_default_datasets_filters_and_output_layout(banks: Path, runner: FrameworkRunner) -> None:
    """Default selection saves all four frameworks by dataset and reruns replace selected outputs."""
    selection = parse_arguments(["--model", MODEL])
    assert selection.datasets == [DATASET, "imoproofbench"]
    directories = runner.run(selection)
    assert len(directories) == 8 and {directory.name for directory in directories} == {"sg", "de", "r-sg", "r-de"}
    for directory in directories:
        assert directory.parent.name == "frameworks"
        assert {path.name for path in directory.iterdir()} == {"fit.json", "predictions.jsonl"}
        fitted = json.loads((directory / "fit.json").read_text())
        predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        assert fitted["model"] == MODEL and fitted["dataset"] == directory.parents[2].name
        expected_ids = {row.problem_id if fitted["dataset"] == DATASET else f"imo_{row.problem_id}" for row in ROWS}
        assert set(fitted["parameters"]) == expected_ids
        assert all(row["dataset"] == fitted["dataset"] for row in predictions)
        assert set(fitted["parameters"]) == {row["problem_id"] for row in predictions}
        parameters = next(iter(fitted["parameters"].values()))
        if directory.name == "sg":
            assert parameters == {"successes": 12, "trials": 24}
        elif directory.name == "de":
            assert set(parameters) == {"alpha", "epsilon", "at_boundary", "is_unidentified"}
            assert parameters["epsilon"][0] == pytest.approx(13 / 28)
        elif directory.name == "r-sg":
            assert parameters == {"successes": 12, "trials": 24, "a": fitted["prior"]["a"], "b": fitted["prior"]["b"]}
        else:
            assert parameters == {"prior": fitted["prior"], "successes": 12, "trials": 24,
                                  "oracle_counts": [1, 1, 0, 1, 0, 0, 0, 0, 1]}
        assert set(predictions[0]) == {"dataset", "problem_id", "n", "k", "success_probability"}
        assert len(predictions) == len(ROWS) * len(EXPECTED_ALLOCATIONS)
    before = {path: path.read_bytes() for directory in directories for path in directory.iterdir()}
    updated = runner.run(replace(selection, problems=["p1"]))
    assert len(updated) == 4
    for directory in updated:
        fitted = json.loads((directory / "fit.json").read_text())
        predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        assert set(fitted["parameters"]) == {"p1"}
        assert len(predictions) == len(EXPECTED_ALLOCATIONS) and {row["problem_id"] for row in predictions} == {"p1"}
    assert all(path.read_bytes() == content for path, content in before.items() if path.parent not in updated)


@pytest.mark.parametrize("failure", ["prediction", "serialization"])
@pytest.mark.parametrize("existing", [False, True])
def test_de_failure_preserves_saved_sg_without_partial_de_output(
    banks: Path, runner: FrameworkRunner, selection: InputSelection, monkeypatch: pytest.MonkeyPatch, failure: str, existing: bool,
) -> None:
    """Prediction or serialization failures leave saved SG and any previous DE files intact."""
    directory = banks / "results" / DATASET / encode_model_directory(MODEL) / "frameworks"
    if existing:
        runner.run(selection)
    previous_de = {path: path.read_bytes() for path in (directory / "de").glob("*")}

    def fail_de_prediction(framework: DiscoveryExecution) -> list[BasePrediction]:
        """Fail before saving or supply a nonfinite value that JSON serialization rejects."""
        if failure == "prediction":
            raise ArithmeticError("injected prediction failure")
        return [BasePrediction(DATASET, ROWS[0].problem_id, 1, 1, float("nan"))]

    monkeypatch.setattr(DiscoveryExecution, "predict", fail_de_prediction)
    with pytest.raises((ArithmeticError, ValueError)):
        runner.run(selection)

    expected_directories = {"sg", "de", "r-sg", "r-de", LOCK_FILENAME} if existing else {"sg", LOCK_FILENAME}
    assert {path.name for path in directory.iterdir()} == expected_directories
    assert {path: path.read_bytes() for path in (directory / "de").glob("*")} == previous_de
    fitted = json.loads((directory / "sg/fit.json").read_text())
    predictions = [json.loads(line) for line in (directory / "sg/predictions.jsonl").read_text().splitlines()]
    assert set(fitted["parameters"]) == {row.problem_id for row in ROWS}
    assert len(predictions) == len(ROWS) * len(EXPECTED_ALLOCATIONS)


def test_regularized_prior_uses_both_selected_datasets(banks: Path, runner: FrameworkRunner) -> None:
    """Different dataset outcomes must influence the same prior before results are partitioned for saving."""
    model_directory = encode_model_directory(MODEL)
    path = banks / "results/imoproofbench" / model_directory / "unaided-1x/audit.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["correctness"]["score"] = 7
    save_bank(path, rows)
    runner.run(parse_arguments(["--model", MODEL]))

    for expected in [RegularizedSimpleGeometric(), RegularizedDiscoveryExecution()]:
        expected.fit(ROWS + [replace(row, dataset="imoproofbench", short_successes=row.short_attempts) for row in ROWS])
        assert expected.prior is not None
        for dataset in [DATASET, "imoproofbench"]:
            output = banks / "results" / dataset / model_directory / "frameworks" / expected.name / "fit.json"
            fitted = json.loads(output.read_text())
            assert fitted["prior"] == asdict(expected.prior)
            assert fitted["optimization"] == expected.get_parameters(dataset)["optimization"]


def test_problem_filter_applies_across_datasets(data_loader: DataLoader) -> None:
    """A problem ID present in only one dataset is valid without specifying a dataset filter."""
    selection = parse_arguments(["--model", MODEL, "--problems", "imo_p1", "--domain", "algebra"])
    observations = data_loader.load(selection)
    assert [[row.dataset, row.problem_id] for row in observations] == [["imoproofbench", "imo_p1"]]
    with pytest.raises(ValueError, match="unknown problem"):
        data_loader.load(replace(selection, problems=["missing"]))
    with pytest.raises(ValueError, match="unknown domain"):
        data_loader.load(replace(selection, domain="missing"))


@pytest.mark.parametrize("arguments", [
    ["--framework", "sg"], ["--fresh-seeds", "1"], ["--oracle-seeds", "1"], ["--n", "2"],
    ["--dataset", "unknown"], ["--dataset", DATASET, DATASET], ["--problems", "p1", "p1"],
])
def test_cli_rejects_removed_or_invalid_options(arguments: list[str]) -> None:
    """Only model and input filters remain configurable."""
    with pytest.raises(SystemExit):
        parse_arguments(["--model", MODEL, *arguments])
