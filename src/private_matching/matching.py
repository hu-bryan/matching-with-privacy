"""Hungarian matching, planar Laplace noise, Sinkhorn, and Birkhoff rounding."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def cost_matrix(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean costs: c[i,j] = ||Q[i] - R[j]||."""
    diff = Q[:, np.newaxis, :] - R[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def hungarian(cost: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (minimum cost, sigma) where sigma[i] = matched column index."""
    row_ind, col_ind = linear_sum_assignment(cost)
    sigma = np.empty(cost.shape[0], dtype=int)
    sigma[row_ind] = col_ind
    total = float(cost[row_ind, col_ind].sum())
    return total, sigma


def match_hungarian(Q: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray]:
    """Optimal matching cost and permutation for Euclidean costs."""
    return hungarian(cost_matrix(Q, R))


def planar_laplace(
    points: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    """Planar Laplace (Geo-Ind) mechanism: Gamma(2, rate=epsilon) radius, uniform angle."""
    n = points.shape[0]
    nu = rng.gamma(shape=2.0, scale=1.0 / epsilon, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    offset = np.column_stack([nu * np.cos(theta), nu * np.sin(theta)])
    return points + offset


def detour_cost(
    Q: np.ndarray,
    R: np.ndarray,
    sigma: np.ndarray,
    rho: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """Crowdshipping detour objective (same argmin as pure Euclidean when balanced)."""
    total = 0.0
    for i, j in enumerate(sigma):
        q_norm = np.linalg.norm(Q[i])
        r_norm = np.linalg.norm(R[j])
        edge = np.linalg.norm(Q[i] - R[j])
        total += rho * (q_norm + edge - r_norm) + gamma
    return total


def assignment_cost(cost: np.ndarray, sigma: np.ndarray) -> float:
    """Sum cost[i, sigma[i]]."""
    return float(cost[np.arange(len(sigma)), sigma].sum())


def sinkhorn(
    K: np.ndarray,
    num_iters: int,
    tol: float = 1e-9,
) -> np.ndarray:
    """Sinkhorn-Knopp scaling to near-doubly-stochastic matrix."""
    n = K.shape[0]
    u = np.ones(n)
    v = np.ones(n)
    for _ in range(num_iters):
        Ku = K @ v
        Ku = np.maximum(Ku, tol)
        u = 1.0 / Ku
        Kv = K.T @ u
        Kv = np.maximum(Kv, tol)
        v = 1.0 / Kv
    P = (u[:, np.newaxis] * K) * v[np.newaxis, :]
    return P


def _permutation_from_support(P: np.ndarray, tol: float = 1e-12) -> np.ndarray | None:
    """Extract a permutation from positive support via maximum-weight matching."""
    n = P.shape[0]
    mask = P > tol
    if not mask.any():
        return None
    # Maximize sum of P[i,j] on a perfect matching → minimize -P
    big = P.max() + 1.0
    cost = np.where(mask, big - P, big + 1.0)
    _, col_ind = linear_sum_assignment(cost)
    if not all(P[i, col_ind[i]] > tol for i in range(n)):
        return None
    return col_ind


def birkhoff_round(P: np.ndarray, rng: np.random.Generator, tol: float = 1e-10) -> np.ndarray:
    """Birkhoff–von Neumann randomized rounding to a permutation."""
    n = P.shape[0]
    residual = P.copy()
    perms: list[np.ndarray] = []
    weights: list[float] = []

    for _ in range(n * n):
        sigma = _permutation_from_support(residual, tol=tol)
        if sigma is None:
            break
        w = min(float(residual[i, sigma[i]]) for i in range(n))
        if w <= tol:
            break
        perms.append(sigma.copy())
        weights.append(w)
        for i in range(n):
            residual[i, sigma[i]] -= w
        if residual.max() <= tol:
            break

    if not perms:
        # Fallback: Hungarian on -P
        _, sigma = hungarian(-P)
        return sigma

    weights_arr = np.array(weights, dtype=float)
    total = weights_arr.sum()
    if total <= tol:
        return perms[0]
    probs = weights_arr / total
    idx = rng.choice(len(perms), p=probs)
    return perms[idx]


def exponential_sample(
    scores: np.ndarray,
    epsilon: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> int:
    """Sample index j with P[j] ∝ exp(-epsilon * scores[j] / sensitivity)."""
    log_w = -epsilon * scores / sensitivity
    log_w -= log_w.max()
    probs = np.exp(log_w)
    probs /= probs.sum()
    return int(rng.choice(len(scores), p=probs))
