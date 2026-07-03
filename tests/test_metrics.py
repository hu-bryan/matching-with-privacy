"""Tests for utility metrics."""

import numpy as np
import pytest

from private_matching.instances import Instance
from private_matching.matching import cost_matrix, match_hungarian
from private_matching.metrics import (
    competitive_ratio,
    compute_invariants,
    hamming_fraction,
    margin_mu,
    rand_cost,
    regret_ratio,
    second_best_cost,
)


def _tiny_instance():
    Q = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    R = np.array([[0.1, 0.0], [1.1, 0.0], [0.0, 1.1]])
    return Instance(Q=Q, R=R, family="test", params={}, instance_id="t1", seed=0)


def test_opt_n3():
    inst = _tiny_instance()
    opt, sigma = match_hungarian(inst.Q, inst.R)
    assert opt > 0
    assert sorted(sigma) == [0, 1, 2]


def test_rand_closed_form():
    inst = _tiny_instance()
    cost = cost_matrix(inst.Q, inst.R)
    assert rand_cost(cost) == pytest.approx(cost.sum() / 3)


def test_margin_nonnegative():
    inst = _tiny_instance()
    cost = cost_matrix(inst.Q, inst.R)
    opt, sigma_star = match_hungarian(inst.Q, inst.R)
    mu = margin_mu(cost, opt, sigma_star)
    assert mu >= 0


def test_competitive_ratio_bounds():
    assert competitive_ratio(10.0, 5.0) == pytest.approx(2.0)
    assert competitive_ratio(5.0, 5.0) == pytest.approx(1.0)


def test_regret_ratio_endpoints():
    opt, rand = 5.0, 15.0
    assert regret_ratio(opt, opt, rand) == pytest.approx(0.0)
    assert regret_ratio(rand, opt, rand) == pytest.approx(1.0)


def test_hamming():
    assert hamming_fraction(np.array([0, 1, 2]), np.array([0, 1, 2])) == 0.0
    assert hamming_fraction(np.array([1, 0, 2]), np.array([0, 1, 2])) == pytest.approx(2 / 3)


def test_second_best_n2():
    Q = np.array([[0.0, 0.0], [2.0, 0.0]])
    R = np.array([[0.0, 0.0], [2.0, 0.0]])
    cost = cost_matrix(Q, R)
    opt, sigma_star = match_hungarian(Q, R)
    second = second_best_cost(cost, sigma_star)
    assert second >= opt


def test_compute_invariants():
    inst = _tiny_instance()
    inv = compute_invariants(inst)
    assert inv.opt > 0
    assert inv.rand > inv.opt
    assert inv.margin >= 0


def test_competitive_regret_consistency():
    """competitive_ratio = 1 + regret_ratio * (RAND/OPT - 1) per evaluation."""
    opt, rand = 4.2, 11.7
    for cost in (opt, rand, 7.5, 15.0):
        comp = competitive_ratio(cost, opt)
        regret = regret_ratio(cost, opt, rand)
        expected = 1.0 + regret * (rand / opt - 1.0)
        assert comp == pytest.approx(expected, rel=1e-12, abs=1e-12)
