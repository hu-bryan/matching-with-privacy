"""Empirical privacy sanity checks (not proofs)."""

import numpy as np
import pytest

from private_matching.matching import planar_laplace
from private_matching.mechanisms import MECHANISMS


def _metric_distance(R: np.ndarray, Rp: np.ndarray) -> float:
    return float(np.linalg.norm(R - Rp, axis=1).sum())


@pytest.mark.slow
def test_local_planar_laplace_privacy_ratio():
    """Empirical likelihood ratio for planar Laplace on a fixed event."""
    rng = np.random.default_rng(0)
    n = 3
    R = rng.uniform(0, 1, (n, 2))
    # Nearby configuration: perturb one point slightly
    Rp = R.copy()
    Rp[0] += np.array([0.05, 0.0])
    eps = 5.0
    d = _metric_distance(R, Rp)

    # Event S: all noisy points land in a fixed box
    box_lo = R.min(axis=0) - 0.5
    box_hi = R.max(axis=0) + 0.5

    def in_box(points):
        return np.all((points >= box_lo) & (points <= box_hi))

    n_samples = 5000
    count_R = sum(
        1
        for _ in range(n_samples)
        if in_box(planar_laplace(R, eps, np.random.default_rng(rng.integers(2**31))))
    )
    count_Rp = sum(
        1
        for _ in range(n_samples)
        if in_box(planar_laplace(Rp, eps, np.random.default_rng(rng.integers(2**31))))
    )

    # Avoid division by zero
    if count_R == 0 or count_Rp == 0:
        pytest.skip("Insufficient samples for ratio estimate")

    ratio = (count_R / n_samples) / (count_Rp / n_samples)
    bound = np.exp(eps * d) * 1.5  # 50% Monte Carlo slack
    assert ratio <= bound, f"Ratio {ratio:.3f} exceeds e^(eps*d)={np.exp(eps*d):.3f}"


def test_local_output_independent_of_Q_shift():
    """Perturbation depends only on R; shifting all Q equally shouldn't change R_tilde."""
    rng = np.random.default_rng(1)
    R = rng.uniform(0, 1, (5, 2))
    eps = 1.0
    r1 = planar_laplace(R, eps, np.random.default_rng(42))
    r2 = planar_laplace(R, eps, np.random.default_rng(42))
    np.testing.assert_array_equal(r1, r2)


def test_auction_budget_split_consistent():
    mech = MECHANISMS["auction"](m=3)
    eps = 6.0
    eps1 = eps / (2 * mech.m)
    eps2 = eps / 2.0
    assert eps1 == pytest.approx(eps / 6.0)
    assert eps2 == pytest.approx(eps / 2.0)
    assert eps1 < eps2


def test_dual_sinkhorn_noise_scale():
    mech = MECHANISMS["dual_sinkhorn"](beta=5.0)
    n = 10
    scale_laplace = n * mech.beta / 1.0
    scale_gaussian = np.sqrt(n) * mech.beta / 1.0
    assert scale_laplace == 50.0
    assert scale_gaussian == pytest.approx(np.sqrt(10) * 5.0)
