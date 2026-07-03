"""Private ascending-price auction (modified Crawford–Knoer) with local-perturbation cleanup."""

from __future__ import annotations

import numpy as np

from private_matching.instances import Instance
from private_matching.matching import cost_matrix, exponential_sample, match_hungarian
from private_matching.mechanisms.base import Mechanism, register_mechanism
from private_matching.mechanisms.local import LocalMechanism


@register_mechanism("auction")
class AuctionMechanism(Mechanism):
    """Modified Crawford–Knoer ascending-price auction with exponential-mechanism bids.

    Bidders are the private points R (drivers); items are the public points Q (customers),
    each carrying an ascending price ``p[q]``. While an unassigned driver still has bids
    left, it bids on an item drawn from the **exponential mechanism** with score
    ``||q - r|| + p[q]`` (1-Lipschitz in r, so sensitivity Δ = 1) — cheaper, cheaper-priced
    items are more likely. Winning an item evicts its previous holder (who may re-bid); the
    item's price then rises by ``alpha``, so contested items become expensive and bidders
    spread out. Each driver gets at most ``m`` bids, guaranteeing termination in ≤ n·m bids.
    Drivers still unassigned after exhausting their bids — and the equally-many leftover
    items — are matched by a local-perturbation cleanup, so the output is always a perfect
    matching.

    Privacy (εd-private): each bid samples ``∝ exp(-ε1 · score / (2Δ))`` with the score
    1-Lipschitz in r (Δ = 1). The metric exponential mechanism with the **2Δ** denominator is
    ε1·d-private per draw — the data-dependent normalizer contributes the second factor
    ``Z(R)/Z(R') ≤ exp(ε1·d/2)``. Sequential composition over ≤ m bids ⇒ ≤ (m·ε1)·d = (ε/2)·d
    per driver; distinct drivers touch disjoint private coordinates (parallel composition); the
    cleanup spends ε2 = ε/2. A driver appearing in both phases spends ε/2 + ε/2 = ε. This pins
    the exponential-mechanism constant to ε/2Δ (the factor-2 question in PLAN.md §14); the
    empirical check lives in tests/test_privacy.py.
    """

    def __init__(self, m: int = 3, alpha: float = 0.1, sensitivity: float = 1.0, **kwargs):
        super().__init__(m=m, alpha=alpha, sensitivity=sensitivity, **kwargs)
        self.m = int(m)
        self.alpha = float(alpha)
        self.sensitivity = float(sensitivity)

    def match(
        self, inst: Instance, epsilon: float, rng: np.random.Generator
    ) -> np.ndarray:
        n = inst.n
        C = cost_matrix(inst.Q, inst.R)  # C[public item i, private bidder j] = ||Q[i]-R[j]||

        eps1 = epsilon / (2.0 * self.m)
        eps2 = epsilon / 2.0

        prices = np.zeros(n)  # price per public item
        item_holder = np.full(n, -1, dtype=int)  # item -> bidder (private idx) or -1
        bidder_item = np.full(n, -1, dtype=int)  # bidder -> item (public idx) or -1
        bids_used = np.zeros(n, dtype=int)

        # Each iteration spends exactly one bid, and a driver can bid at most m times,
        # so the loop runs at most n * m times.
        for _ in range(n * self.m):
            candidates = np.where((bidder_item < 0) & (bids_used < self.m))[0]
            if candidates.size == 0:
                break
            r = int(candidates[0])

            scores = C[:, r] + prices  # score is 1-Lipschitz in r ⇒ sensitivity Δ = 1
            # Metric exponential mechanism at budget eps1: sample ∝ exp(-eps1 * score / (2Δ)).
            # The 2Δ denominator (not Δ) is what makes one bid rigorously eps1·d-private — the
            # data-dependent normalizer contributes the second factor exp(eps1·d/2).
            q = exponential_sample(scores, eps1, 2.0 * self.sensitivity, rng)
            bids_used[r] += 1

            prev = item_holder[q]
            if prev >= 0:
                bidder_item[prev] = -1  # evict the previous holder
            item_holder[q] = r
            bidder_item[r] = q
            prices[q] += self.alpha

        # Partial matching so far: sigma[public item] = private bidder.
        sigma = np.full(n, -1, dtype=int)
        for r in range(n):
            if bidder_item[r] >= 0:
                sigma[bidder_item[r]] = r

        # Local-perturbation cleanup on the leftovers (equal counts by construction).
        unmatched_items = np.where(sigma < 0)[0]
        unmatched_bidders = np.where(bidder_item < 0)[0]
        if unmatched_items.size > 0:
            sub = Instance(
                Q=inst.Q[unmatched_items],
                R=inst.R[unmatched_bidders],
                family=inst.family,
                params=inst.params,
                instance_id=inst.instance_id + "_residual",
                seed=inst.seed,
            )
            sub_sigma = LocalMechanism().match(sub, eps2, rng)
            for k, item in enumerate(unmatched_items):
                sigma[item] = int(unmatched_bidders[sub_sigma[k]])

        # Safety net (should not trigger): guarantee a valid permutation.
        if (sigma < 0).any():
            _, sigma = match_hungarian(inst.Q, inst.R)
        return sigma
