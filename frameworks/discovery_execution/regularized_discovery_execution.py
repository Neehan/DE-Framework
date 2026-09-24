"""Empirical-Bayes DE with joint Beta/Dirichlet fitting and exact posterior allocation predictions."""

from functools import partial
from itertools import product
from math import comb

import numpy as np
from numpy.typing import NDArray
from scipy.special import poch

from frameworks.base.base_regularized import BaseRegularized
from frameworks.models import Observation, RDEParameters, RDEPosterior, RDEPrior
from frameworks.utils.constants import RDE, RDE_INITIALIZATIONS
from frameworks.utils.probabilities import (
    build_discovery_execution_prior,
    calculate_discovery_execution_posterior,
    discovery_execution_nll,
)


class RegularizedDiscoveryExecution(BaseRegularized[RDEParameters, RDEPrior]):
    """Use inherited fit/predict to learn joint priors and integrate DE success; private helpers compute exact moments."""

    name = RDE

    def _fit_prior(self, observations: list[Observation]) -> RDEPrior:
        """Fit discovery and execution shapes jointly over observed oracle completion categories."""
        successes = np.array([obs.short_successes for obs in observations], dtype=float)
        trials = np.array([obs.short_attempts for obs in observations], dtype=float)
        oracle_counts = np.array([self._get_oracle_counts(obs) for obs in observations], dtype=float)
        total_oracle_counts = oracle_counts.sum(axis=0)
        has_oracle_observations = total_oracle_counts > 0
        if not has_oracle_observations[0] or has_oracle_observations.sum() < 2:
            raise ValueError("R-DE requires some first-block and some later or unsolved oracle outcomes")

        execution_proportions = total_oracle_counts[has_oracle_observations] / total_oracle_counts.sum()
        initializations = []
        for mean, discovery_concentration, execution_concentration in RDE_INITIALIZATIONS:
            initializations.append([
                mean * discovery_concentration, (1 - mean) * discovery_concentration,
                *(float(value) for value in execution_concentration * execution_proportions),
            ])
        objective = partial(discovery_execution_nll, successes=successes, trials=trials, oracle_counts=oracle_counts)
        parameters = self._optimize_prior(objective, initializations, use_gradient=False)
        return build_discovery_execution_prior(parameters, has_oracle_observations)

    def _fit_parameters(self, observation: Observation) -> RDEParameters:
        """Keep the prior and both types of counts so saved predictions can be reconstructed."""
        if self.prior is None:
            raise ValueError("R-DE requires a fitted prior")
        return RDEParameters(
            prior=self.prior, successes=observation.short_successes, trials=observation.short_attempts,
            oracle_counts=self._get_oracle_counts(observation),
        )

    def _get_oracle_counts(self, observation: Observation) -> list[int]:
        """Convert cumulative successes to first-completion counts, including unsolved after the final block."""
        cumulative = [0, *observation.oracle_successes, observation.oracle_attempts]
        return [later - earlier for earlier, later in zip(cumulative, cumulative[1:])]

    def _success_probability(self, params: RDEParameters, n: int, k: int) -> float:
        """Integrate any-of-n success using exact moments of one trajectory's k-block success probability."""
        posterior = calculate_discovery_execution_posterior(
            params.prior, np.array([params.successes], dtype=float), np.array([params.trials], dtype=float),
            np.array([params.oracle_counts], dtype=float),
        )
        success_by_case = self._expected_allocation_success(posterior, n, k)
        return float(np.sum(posterior.case_probabilities * success_by_case))

    def _expected_allocation_success(self, posterior: RDEPosterior, n: int, k: int) -> NDArray[np.float64]:
        """Calculate E[1-(1-s_k)^n] within each possible explanation of the observed short failures."""
        success_by_case = np.zeros_like(posterior.case_probabilities)
        for power in range(1, n + 1):
            # Expand 1-(1-s_k)^n: each term needs the posterior expectation of one power of s_k.
            expected_power = self._expected_success_power(posterior, k, power)
            coefficient = (-1) ** (power + 1) * comb(n, power)
            success_by_case += coefficient * expected_power
        return success_by_case

    def _expected_success_power(self, posterior: RDEPosterior, k: int, power: int) -> NDArray[np.float64]:
        """Calculate E[s_k^power] by enumerating successful discovery/execution paths and integrating their products."""
        # A path discovers after some failures, then completes execution within the remaining blocks.
        # Execution blocks are zero-indexed: block 0 means completion in the first execution block.
        success_paths = []
        for discovery_failures in range(k):
            for execution_block in range(k - discovery_failures):
                success_paths.append([discovery_failures, execution_block])

        # Raising the sum of path probabilities to a power produces products of these paths.
        expected_power = np.zeros_like(posterior.case_probabilities)
        for paths in product(success_paths, repeat=power):
            discovery_failures = sum(failures for failures, _ in paths)
            execution_blocks = [block for _, block in paths]
            discovery_moment = self._beta_moment(
                posterior.discovery_a, posterior.discovery_b, power, discovery_failures,
            )
            execution_moment = self._execution_moment(posterior, execution_blocks)
            expected_power += discovery_moment * execution_moment
        return expected_power

    def _execution_moment(self, posterior: RDEPosterior, blocks: list[int]) -> NDArray[np.float64]:
        """Integrate products of first-block probability and conditional later-category probabilities."""
        first_count = blocks.count(0)
        later_count = len(blocks) - first_count
        first_moment = self._beta_moment(
            posterior.first_execution_a, posterior.first_execution_b, first_count, later_count,
        )
        later_counts = np.bincount(blocks, minlength=posterior.later_execution_concentrations.shape[1] + 1)[1:]
        # Dirichlet mixed moments are products of rising factorials divided by the total's rising factorial.
        numerator = np.prod(poch(posterior.later_execution_concentrations, later_counts), axis=1)
        denominator = poch(posterior.later_execution_concentrations.sum(axis=1), later_count)
        return first_moment * (numerator / denominator)[:, None]

    def _beta_moment(
        self, a: NDArray[np.float64], b: NDArray[np.float64], successes: int, failures: int,
    ) -> NDArray[np.float64]:
        """Return E[q^successes*(1-q)^failures] under Beta(a,b), using stable rising-factorial ratios."""
        return poch(a, successes) * poch(b, failures) / poch(a + b, successes + failures)
