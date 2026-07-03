"""Scale invariance of utility metrics under (lambda, epsilon/lambda) scaling."""

import numpy as np
import pytest

from private_matching.instances import Instance
from private_matching.mechanisms.local import LocalMechanism
from private_matching.metrics import competitive_ratio, compute_invariants, regret_ratio


def test_scale_invariance_competitive_regret():
    rng = np.random.default_rng(0)
    n = 8
    Q = rng.uniform(0, 1, (n, 2))
    R = rng.uniform(0, 1, (n, 2))
    inst = Instance(Q=Q, R=R, family="test", params={}, instance_id="s1", seed=0)
    inv = compute_invariants(inst)

    lam = 3.0
    eps = 2.0
    eps_scaled = eps / lam

    mech = LocalMechanism()
    trial_seeds = list(range(20))

    ratios_orig = []
    ratios_scaled = []
    regrets_orig = []
    regrets_scaled = []

    for t in trial_seeds:
        ss = np.random.SeedSequence([0, t]).spawn(1)
        rng1 = np.random.default_rng(ss[0])
        sigma1 = mech.match(inst, eps, rng1)
        cost1 = float(np.linalg.norm(Q - R[sigma1], axis=1).sum())
        ratios_orig.append(competitive_ratio(cost1, inv.opt))
        regrets_orig.append(regret_ratio(cost1, inv.opt, inv.rand))

        Qs = Q * lam
        Rs = R * lam
        inst_s = Instance(
            Q=Qs, R=Rs, family="test", params={}, instance_id="s1s", seed=0
        )
        inv_s = compute_invariants(inst_s)
        ss2 = np.random.SeedSequence([0, t]).spawn(1)
        rng2 = np.random.default_rng(ss2[0])
        sigma2 = mech.match(inst_s, eps_scaled, rng2)
        cost2 = float(np.linalg.norm(Qs - Rs[sigma2], axis=1).sum())
        ratios_scaled.append(competitive_ratio(cost2, inv_s.opt))
        regrets_scaled.append(regret_ratio(cost2, inv_s.opt, inv_s.rand))

    # Compare distributions with tolerance
    assert np.median(ratios_orig) == pytest.approx(np.median(ratios_scaled), rel=0.15)
    assert np.median(regrets_orig) == pytest.approx(np.median(regrets_scaled), rel=0.15)
