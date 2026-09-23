"""Framework probability calculations, posterior distributions, and prior-fitting likelihoods."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, digamma, gammaln, logsumexp

from frameworks.models import BetaParameters, DirichletParameters, RDEPosterior, RDEPrior
from frameworks.utils.constants import TOTAL_BUDGET


def beta_binomial_nll(
    log_shapes: NDArray[np.float64], successes: NDArray[np.float64], attempts: NDArray[np.float64],
) -> list[float | NDArray[np.float64]]:
    """Return marginal NLL up to a data-only constant and its gradient with respect to log(a), log(b)."""
    a, b = np.exp(log_shapes)
    failures = attempts - successes
    # Binomial coefficients do not depend on a or b, so they can be omitted when fitting.
    value = -np.sum(betaln(a + successes, b + failures) - betaln(a, b))
    total_change = digamma(a + b + attempts) - digamma(a + b)
    gradient = -np.array([
        a * np.sum(digamma(a + successes) - digamma(a) - total_change),
        b * np.sum(digamma(b + failures) - digamma(b) - total_change),
    ])
    return [float(value), gradient]


def discovery_execution_nll(
    log_parameters: NDArray[np.float64], successes: NDArray[np.float64], trials: NDArray[np.float64],
    oracle_counts: NDArray[np.float64],
) -> float:
    """Return joint marginal NLL, omitting binomial and multinomial coefficients constant in the prior."""
    has_oracle_observations = oracle_counts.sum(axis=0) > 0
    prior = build_discovery_execution_prior(np.exp(log_parameters), has_oracle_observations)
    posterior = calculate_discovery_execution_posterior(prior, successes, trials, oracle_counts)
    return -float(posterior.log_evidence.sum())


def build_discovery_execution_prior(
    parameters: NDArray[np.float64], has_oracle_observations: NDArray[np.bool_],
) -> RDEPrior:
    """Read positive optimizer values as a, b, then concentrations for observed outcomes; excluded outcomes stay zero."""
    a, b, *execution_concentrations = parameters
    concentrations = np.zeros(TOTAL_BUDGET + 1)
    concentrations[has_oracle_observations] = execution_concentrations
    return RDEPrior(
        discovery=BetaParameters(a=float(a), b=float(b)),
        execution=DirichletParameters([float(value) for value in concentrations]),
    )


def calculate_discovery_execution_posterior(
    prior: RDEPrior, successes: NDArray[np.float64], trials: NDArray[np.float64],
    oracle_counts: NDArray[np.float64],
) -> RDEPosterior:
    """Update oracle counts, consider each cause of short failures, and calculate the probability of each case."""
    a, b = prior.discovery.a, prior.discovery.b
    concentrations = np.array(prior.execution.concentrations)
    has_oracle_observations = concentrations > 0

    # Update execution with oracle observations: first-block success versus all later or unsolved outcomes.
    oracle_concentrations = concentrations + oracle_counts
    first_block_concentration = oracle_concentrations[:, :1]
    later_execution_concentrations = oracle_concentrations[:, 1:]
    later_concentration = later_execution_concentrations.sum(axis=1, keepdims=True)
    oracle_trials = oracle_counts.sum(axis=1)
    total_concentration = concentrations.sum()
    oracle_log_evidence = gammaln(total_concentration) - gammaln(total_concentration + oracle_trials)
    oracle_log_evidence += (gammaln(oracle_concentrations[:, has_oracle_observations])
                           - gammaln(concentrations[has_oracle_observations])).sum(axis=1)

    # Each case assigns some failed short attempts to execution after discovery; the rest failed discovery.
    short_successes = successes[:, None]
    short_failures = (trials - successes)[:, None]
    execution_failures = np.arange(int(short_failures.max()) + 1)[None, :]
    discovery_failures = np.maximum(short_failures - execution_failures, 0)
    # Shorter rows have impossible cases in the shared columns; -inf gives those cases zero probability.
    log_combinations = np.where(
        execution_failures <= short_failures,
        gammaln(short_failures + 1) - gammaln(execution_failures + 1) - gammaln(discovery_failures + 1), -np.inf,
    )
    discovery_a = a + short_successes + execution_failures
    discovery_b = b + discovery_failures
    first_execution_a = first_block_concentration + short_successes
    first_execution_b = later_concentration + execution_failures

    # Integrate each case's discovery/execution likelihood, then normalize the case probabilities.
    log_case_weights = log_combinations + betaln(discovery_a, discovery_b) - betaln(a, b)
    log_case_weights += betaln(first_execution_a, first_execution_b) - betaln(first_block_concentration, later_concentration)
    short_log_evidence = np.asarray(logsumexp(log_case_weights, axis=1), dtype=float)
    return RDEPosterior(
        log_evidence=oracle_log_evidence + short_log_evidence,
        case_probabilities=np.exp(log_case_weights - short_log_evidence[:, None]),
        discovery_a=discovery_a, discovery_b=discovery_b,
        first_execution_a=first_execution_a, first_execution_b=first_execution_b,
        later_execution_concentrations=later_execution_concentrations,
    )
