"""Fit allocation frameworks on shared input data and save results separately by dataset."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from harness.session.run_lock import RunLock
from harness.utils.constants import TEXT_ENCODING
from launcher.models import encode_model_directory

from frameworks.discovery_execution.discovery_execution import DiscoveryExecution
from frameworks.discovery_execution.regularized_discovery_execution import RegularizedDiscoveryExecution
from frameworks.geometric.simple_geometric import SimpleGeometric
from frameworks.geometric.regularized_simple_geometric import RegularizedSimpleGeometric
from frameworks.models import BasePrediction, InputSelection
from frameworks.utils.constants import FIT_FILENAME, FRAMEWORKS_DIRECTORY, JSON_INDENT, MIN_AUDIT_SCORE, PREDICTIONS_FILENAME
from frameworks.utils.dataloader import DataLoader


class FrameworkRunner:
    """run fits all four frameworks on the same selected problems and saves each completed framework; no overrides are required."""

    def __init__(self, data_loader: DataLoader, results: Path) -> None:
        """Inject the input loader and output root."""
        self._data_loader = data_loader
        self._results = results

    def run(self, selection: InputSelection) -> list[Path]:
        """Fit across selected datasets so regularized frameworks share one prior, then save by dataset."""
        observations = self._data_loader.load(selection)
        if not observations:
            return []

        directories = []
        for framework in [SimpleGeometric(), DiscoveryExecution(), RegularizedSimpleGeometric(), RegularizedDiscoveryExecution()]:
            framework.fit(observations)
            predictions_by_dataset: dict[str, list[BasePrediction]] = {}
            for prediction in framework.predict():
                predictions_by_dataset.setdefault(prediction.dataset, []).append(prediction)

            for dataset, predictions in predictions_by_dataset.items():
                directory = self._get_output_directory(selection, dataset, framework.name)
                fit_record = {
                    "framework": framework.name,
                    "model": selection.model,
                    "dataset": dataset,
                    "passing_score": MIN_AUDIT_SCORE,
                    **framework.get_parameters(dataset),
                }
                self._save(directory, fit_record, predictions)
                directories.append(directory)
        return directories

    def _get_output_directory(self, selection: InputSelection, dataset: str, name: str) -> Path:
        """Keep framework outputs separate from experiment outputs."""
        return self._results / dataset / encode_model_directory(selection.model) / FRAMEWORKS_DIRECTORY / name

    def _save(self, directory: Path, fitted: dict[str, object], predictions: list[BasePrediction]) -> None:
        """Prepare both files under a lock, then atomically replace each existing output file."""
        with RunLock(directory.parent):
            with TemporaryDirectory(dir=directory.parent) as temporary:
                staged = Path(temporary)

                fit_json = json.dumps(fitted, indent=JSON_INDENT, allow_nan=False)
                (staged / FIT_FILENAME).write_text(fit_json + "\n", encoding=TEXT_ENCODING)

                with (staged / PREDICTIONS_FILENAME).open("w", encoding=TEXT_ENCODING) as output:
                    for prediction in predictions:
                        record = json.dumps(asdict(prediction), allow_nan=False)
                        output.write(record + "\n")

                # Replace outputs only after both files have serialized successfully.
                directory.mkdir(exist_ok=True)
                for filename in [FIT_FILENAME, PREDICTIONS_FILENAME]:
                    os.replace(staged / filename, directory / filename)
