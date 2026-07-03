"""Private entropic dual ascent: privatize a smoothed kernel, Sinkhorn-scale, then round."""

from __future__ import annotations

import numpy as np

from private_matching.instances import Instance
from private_matching.matching import birkhoff_round, cost_matrix, sinkhorn
from private_matching.mechanisms.base import Mechanism, register_mechanism


@register_mechanism("dual_sinkhorn")
class DualSinkhornMechanism(Mechanism):
    """Privatize a smoothed clipped kernel, Sinkhorn-scale it, then Birkhoff round.

    The ONLY private step releases the kernel ``K[i,j] = exp(-beta * min(||q_i - r_j||, B))``.
    Clipping distances at ``B`` bounds the per-column (per-driver) sensitivity: moving one
    driver changes only its column, each of whose n entries moves by ≤ beta, so the per-driver
    ℓ1 sensitivity is ``n*beta`` (ℓ2 sensitivity ``sqrt(n)*beta``). Adding calibrated noise and
    clipping to [0,1] yields an ``εd``-private ``K̃``; everything after is post-processing:
    ``L`` Sinkhorn iterations give a near-doubly-stochastic ``P``, and Birkhoff–von Neumann
    randomized rounding samples a permutation with expectation ``P``.

    We round by **sampling**, not Hungarian: the Sinkhorn scalings are row/column potentials
    that wash out of any argmax (matching-invariance of dual potentials), so finishing with
    Hungarian would collapse the method to a noisy baseline.

    Known weakness (a finding, not a bug): the noise scale grows like ``n*beta`` — it is
    n-driven, not geometry-driven — so this mechanism is expected to trail local perturbation,
    and worse as n grows. Stress-test by sweeping n. Optional mitigations behind flags:
    ``use_gaussian`` and ``row_clip`` (per-customer contribution clipping).

    The default (Laplace) is **pure εd-privacy** (δ = 0, all distances). The ``use_gaussian``
    path gives only **approximate (εd, δ)** metric privacy: with a fixed δ the guarantee is
    meaningful for nearby inputs (roughly εd < 1) and degrades for far pairs, so it is off by
    default. Its scale is √(2 ln(1.25/δ))·√n·β / ε (ℓ2 sensitivity √n·β).

    Note ``B`` must exceed the cost scale that discriminates good from bad matchings, or the
    clip erases the signal before any noise is added.
    """

    def __init__(
        self,
        beta: float = 5.0,
        B: float = 1.0,
        num_iters: int = 50,
        use_gaussian: bool = False,
        row_clip: float | None = None,
        delta: float = 1e-5,
        **kwargs,
    ):
        super().__init__(
            beta=beta,
            B=B,
            num_iters=num_iters,
            use_gaussian=use_gaussian,
            row_clip=row_clip,
            delta=delta,
            **kwargs,
        )
        self.beta = float(beta)
        self.B = float(B)
        self.num_iters = int(num_iters)
        self.use_gaussian = bool(use_gaussian)
        self.row_clip = row_clip
        self.delta = float(delta)

    def _privatize_kernel(
        self, K: np.ndarray, epsilon: float, rng: np.random.Generator
    ) -> np.ndarray:
        n = K.shape[0]

        if self.row_clip is not None:
            # Bound each customer's (row's) total contribution before releasing the kernel.
            row_sums = K.sum(axis=1, keepdims=True)
            K = K * np.minimum(1.0, self.row_clip / np.maximum(row_sums, 1e-12))

        if self.use_gaussian:
            # Gaussian mechanism for (epsilon, delta)-metric-DP with ℓ2 sensitivity sqrt(n)*beta.
            l2_sensitivity = np.sqrt(n) * self.beta
            scale = np.sqrt(2.0 * np.log(1.25 / self.delta)) * l2_sensitivity / epsilon
            noise = rng.normal(0.0, scale, size=K.shape)
        else:
            # Laplace mechanism for epsilon-metric-DP with ℓ1 sensitivity n*beta.
            l1_sensitivity = n * self.beta
            scale = l1_sensitivity / epsilon
            noise = rng.laplace(0.0, scale, size=K.shape)

        return np.clip(K + noise, 0.0, 1.0)

    def match(
        self, inst: Instance, epsilon: float, rng: np.random.Generator
    ) -> np.ndarray:
        cost = cost_matrix(inst.Q, inst.R)
        K = np.exp(-self.beta * np.minimum(cost, self.B))
        K_tilde = self._privatize_kernel(K, epsilon, rng)
        P = sinkhorn(K_tilde, self.num_iters)
        return birkhoff_round(P, rng)
