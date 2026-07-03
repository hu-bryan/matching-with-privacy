"""Ground-truth checks for the block_alpha difficulty-graded generator."""

import numpy as np
import pytest

from private_matching.difficulty import instance_difficulty
from private_matching.instances import generate_instance
from private_matching.matching import cost_matrix, match_hungarian
from private_matching.metrics import margin_mu, rand_cost


def _gen(n=20, alpha=1.0, stakes_S=5.0, h=1.0, delta_max=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return generate_instance(
        "block_alpha", n, rng, alpha=alpha, stakes_S=stakes_S, h=h, delta_max=delta_max
    )


def test_shape_and_params():
    inst = _gen(n=20, alpha=2.0, stakes_S=5.0)
    assert inst.Q.shape == (20, 2)
    assert inst.R.shape == (20, 2)
    assert inst.family == "block_alpha"
    assert inst.params["alpha"] == 2.0
    assert inst.params["stakes_S"] == 5.0
    assert len(inst.params["deltas"]) == 10  # K = n/2 blocks
    assert instance_difficulty(inst) == {"alpha": 2.0, "stakes_S": 5.0}


def test_opt_is_2Kh():
    """Optimal matching is the vertical edges: cost 2h per block, 2*K*h overall."""
    h = 1.0
    inst = _gen(n=20, h=h)
    opt, _ = match_hungarian(inst.Q, inst.R)
    K = inst.n // 2
    assert opt == pytest.approx(2.0 * K * h, rel=1e-9)


def test_margin_equals_cheapest_block_flip():
    """Second-best flips exactly the smallest-margin block from vertical to crossed."""
    h = 1.0
    inst = _gen(n=20, h=h, alpha=1.0, seed=3)
    cost = cost_matrix(inst.Q, inst.R)
    opt, sigma_star = match_hungarian(inst.Q, inst.R)

    deltas = np.array(inst.params["deltas"])
    widths = np.sqrt(h * deltas)
    crossed_excess = 2.0 * np.sqrt(widths**2 + h**2) - 2.0 * h  # exact per-block excess
    expected_margin = float(crossed_excess.min())

    mu = margin_mu(cost, opt, sigma_star)
    assert mu == pytest.approx(expected_margin, rel=1e-6)


def test_stakes_makes_rand_exceed_opt():
    inst = _gen(n=20, stakes_S=5.0)
    cost = cost_matrix(inst.Q, inst.R)
    opt, _ = match_hungarian(inst.Q, inst.R)
    assert rand_cost(cost) / opt > 1.5  # far-apart blocks => costly average matching


def test_alpha_shifts_margin_distribution():
    """Larger alpha pushes margins higher (isolated optimum); smaller alpha piles near 0."""
    lo = np.array(_gen(alpha=0.5, seed=1).params["deltas"])
    hi = np.array(_gen(alpha=3.0, seed=1).params["deltas"])
    assert hi.mean() > lo.mean()


def test_invalid_params_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_instance("block_alpha", 15, rng)  # odd n
    with pytest.raises(ValueError):
        generate_instance("block_alpha", 20, rng, alpha=-1.0)  # alpha <= 0
