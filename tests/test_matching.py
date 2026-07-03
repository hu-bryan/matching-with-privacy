"""Tests for matching primitives."""

import numpy as np
import pytest

from private_matching.matching import (
    birkhoff_round,
    cost_matrix,
    hungarian,
    match_hungarian,
    planar_laplace,
    sinkhorn,
)


def test_hungarian_n2():
    Q = np.array([[0.0, 0.0], [1.0, 1.0]])
    R = np.array([[0.0, 1.0], [1.0, 0.0]])
    opt, sigma = match_hungarian(Q, R)
    assert opt == pytest.approx(2.0)
    assert set(sigma) == {0, 1}


def test_planar_laplace_mean_radius():
    rng = np.random.default_rng(0)
    pts = np.zeros((5000, 2))
    eps = 2.0
    noisy = planar_laplace(pts, eps, rng)
    radii = np.linalg.norm(noisy, axis=1)
    assert radii.mean() == pytest.approx(2.0 / eps, rel=0.1)


def test_birkhoff_round_is_permutation():
    rng = np.random.default_rng(1)
    n = 5
    P = np.ones((n, n)) / n
    sigma = birkhoff_round(P, rng)
    assert sorted(sigma) == list(range(n))


def test_sinkhorn_doubly_stochastic():
    n = 4
    K = np.random.default_rng(0).uniform(0.1, 1.0, (n, n))
    P = sinkhorn(K, 100)
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    assert np.allclose(row_sums, 1.0, atol=1e-3)
    assert np.allclose(col_sums, 1.0, atol=1e-3)


def test_cost_matrix_symmetry():
    Q = np.array([[0.0, 0.0], [1.0, 0.0]])
    R = np.array([[0.5, 0.5], [0.0, 1.0]])
    c = cost_matrix(Q, R)
    assert c.shape == (2, 2)
    assert c[0, 0] == pytest.approx(np.linalg.norm(Q[0] - R[0]))


def test_hungarian_returns_valid_assignment():
    rng = np.random.default_rng(42)
    n = 6
    Q = rng.uniform(0, 1, (n, 2))
    R = rng.uniform(0, 1, (n, 2))
    cost = cost_matrix(Q, R)
    total, sigma = hungarian(cost)
    assert sorted(sigma) == list(range(n))
    assert total == pytest.approx(cost[np.arange(n), sigma].sum())
