"""Explicit experiment selection without dynamic discovery or duplicate metadata."""

from experiments.alt_sketch import AltSketch
from experiments.continue_oracle import ContinueOracle
from experiments.continue_unaided import ContinueUnaided
from experiments.no_sketch import NoSketch
from experiments.oracle_execution import OracleExecution
from experiments.oracle_sketch import OracleSketch
from experiments.unaided import Unaided

EXPERIMENTS = {experiment.name: experiment for experiment in (
    Unaided, OracleExecution, NoSketch, OracleSketch, AltSketch, ContinueUnaided, ContinueOracle,
)}
