"""OPT, RAND, margin (mu), utility ratios, and instance invariants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_matching.instances import Instance
from private_matching.matching import assignment_cost, cost_matrix, hungarian, match_hungarian


@dataclass(frozen=True)
class InstanceInvariants:
    """Precomputed per-instance quantities for experiment rows."""

    opt: float
    sigma_star: np.ndarray
    rand: float
    margin: float
    margin_norm: float
    L_diam: float
    L_nn: float
    degenerate: bool


def rand_cost(cost: np.ndarray) -> float:
    """Expected cost of a uniformly random perfect matching."""
    n = cost.shape[0]
    return float(cost.sum() / n)


def second_best_cost(cost: np.ndarray, sigma_star: np.ndarray) -> float:
    """Minimum cost among assignments differing from sigma_star in at least one edge."""
    n = cost.shape[0]
    best = float("inf")
    for i in range(n):
        modified = cost.copy()
        modified[i, sigma_star[i]] = np.inf
        total, _ = hungarian(modified)
        best = min(best, total)
    return best


def margin_mu(cost: np.ndarray, opt: float, sigma_star: np.ndarray, tol: float = 1e-12) -> float:
    """Optimality margin mu = second_best - OPT."""
    second = second_best_cost(cost, sigma_star)
    return max(second - opt, 0.0) if second - opt > -tol else 0.0


def characteristic_length_diam(Q: np.ndarray, R: np.ndarray) -> float:
    """Diameter of Q ∪ R."""
    points = np.vstack([Q, R])
    if len(points) < 2:
        return 1.0
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    return float(dists.max())


def characteristic_length_nn(Q: np.ndarray, R: np.ndarray) -> float:
    """Mean nearest-neighbor spacing over Q ∪ R."""
    points = np.vstack([Q, R])
    n = len(points)
    if n < 2:
        return 1.0
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    return float(nn.mean())


def competitive_ratio(cost: float, opt: float) -> float:
    return cost / opt if opt > 0 else float("inf")


def regret_ratio(cost: float, opt: float, rand: float) -> float:
    denom = rand - opt
    if abs(denom) < 1e-15:
        return 0.0 if abs(cost - opt) < 1e-15 else float("inf")
    return (cost - opt) / denom


def hamming_fraction(sigma: np.ndarray, sigma_star: np.ndarray) -> float:
    """Fraction of positions where sigma differs from sigma_star."""
    n = len(sigma)
    return float(np.sum(sigma != sigma_star) / n)


def compute_invariants(inst: Instance, margin_tol: float = 1e-12) -> InstanceInvariants:
    """Precompute OPT, RAND, mu, and characteristic lengths for an instance."""
    cost = cost_matrix(inst.Q, inst.R)
    opt, sigma_star = match_hungarian(inst.Q, inst.R)
    rand = rand_cost(cost)
    mu = margin_mu(cost, opt, sigma_star, tol=margin_tol)
    L_diam = characteristic_length_diam(inst.Q, inst.R)
    L_nn = characteristic_length_nn(inst.Q, inst.R)
    margin_norm = mu / opt if opt > 0 else 0.0
    degenerate = mu <= margin_tol
    return InstanceInvariants(
        opt=opt,
        sigma_star=sigma_star,
        rand=rand,
        margin=mu,
        margin_norm=margin_norm,
        L_diam=L_diam,
        L_nn=L_nn,
        degenerate=degenerate,
    )


def evaluate_assignment(
    inst: Instance,
    sigma: np.ndarray,
    inv: InstanceInvariants,
) -> dict[str, float]:
    """Compute cost and utility metrics for a matching."""
    cost = cost_matrix(inst.Q, inst.R)
    u = assignment_cost(cost, sigma)
    return {
        "cost": u,
        "opt": inv.opt,
        "rand": inv.rand,
        "margin": inv.margin,
        "margin_norm": inv.margin_norm,
        "competitive_ratio": competitive_ratio(u, inv.opt),
        "regret_ratio": regret_ratio(u, inv.opt, inv.rand),
        "hamming": hamming_fraction(sigma, inv.sigma_star),
    }
