"""Validity tests: all mechanisms return perfect matchings."""

import time

import numpy as np
import pytest

from private_matching.instances import GENERATORS, generate_instance
from private_matching.mechanisms import MECHANISMS

MECH_NAMES = list(MECHANISMS.keys())
FAMILIES = ["uniform", "two_gaussian", "ring", "lattice"]
NS = [3, 5, 10]
EPSILONS = [1e-6, 0.01, 1.0, 100.0]


def _is_permutation(sigma: np.ndarray, n: int) -> bool:
    return len(sigma) == n and sorted(sigma.tolist()) == list(range(n))


@pytest.mark.parametrize("mech_name", MECH_NAMES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_mechanism_valid_permutation(mech_name, family, n, epsilon):
    rng = np.random.default_rng(0)
    inst = generate_instance(family, n, rng)
    mech = MECHANISMS[mech_name]()
    start = time.perf_counter()
    sigma = mech.match(inst, epsilon, np.random.default_rng(1))
    elapsed = time.perf_counter() - start
    assert _is_permutation(sigma, n), f"{mech_name} returned invalid sigma: {sigma}"
    assert elapsed < 30.0, f"{mech_name} timed out on n={n}"


def test_all_mechanisms_registered():
    assert set(MECHANISMS.keys()) == {"local", "auction", "dual_sinkhorn"}


def test_all_generators_registered():
    assert set(GENERATORS.keys()) >= {"uniform", "two_gaussian", "ring", "lattice"}
