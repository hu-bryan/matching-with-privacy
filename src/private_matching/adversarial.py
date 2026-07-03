"""CMA-ES adversarial search for worst-case instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import SeedSequence

from private_matching.instances import Instance
from private_matching.mechanisms import MECHANISMS
from private_matching.metrics import compute_invariants, regret_ratio


@dataclass
class AdversarialConfig:
    """Settings for worst-case instance search."""

    n: int = 10
    target_epsilon_L: float = 1.0
    mechanism: str = "local"
    mechanism_params: dict[str, Any] | None = None
    num_trials: int = 10
    trial_seeds: list[int] | None = None
    base_seed: int = 0
    max_iter: int = 100
    population_size: int | None = None


def _project_unit_disk(coords: np.ndarray) -> np.ndarray:
    """Project 2D points into [0,1]^2 (unit square proxy for unit disk experiments)."""
    return np.clip(coords, 0.0, 1.0)


def _coords_to_instance(coords: np.ndarray, n: int, seed: int) -> Instance:
    Q = coords[:n].reshape(n, 2)
    R = coords[n:].reshape(n, 2)
    return Instance(
        Q=_project_unit_disk(Q),
        R=_project_unit_disk(R),
        family="adversarial",
        params={"search_seed": seed},
        instance_id=f"adv_{seed}",
        seed=seed,
    )


def estimate_regret(
    inst: Instance,
    mechanism_name: str,
    epsilon: float,
    num_trials: int,
    trial_seeds: list[int],
    base_seed: int,
    mech_params: dict[str, Any] | None = None,
) -> float:
    """Estimate mean regret ratio at fixed epsilon using CRN across trials."""
    inv = compute_invariants(inst)
    mech = MECHANISMS[mechanism_name](**(mech_params or {}))

    regrets = []
    for t, ts in enumerate(trial_seeds[:num_trials]):
        ss = SeedSequence([base_seed, hash(inst.instance_id) % (2**31), ts]).spawn(1)
        rng = np.random.default_rng(ss[0])
        sigma = mech.match(inst, epsilon, rng)
        cost = float(np.linalg.norm(inst.Q - inst.R[sigma], axis=1).sum())
        regrets.append(regret_ratio(cost, inv.opt, inv.rand))
    return float(np.mean(regrets))


def target_epsilon_over_L(inst: Instance, inv, target_epsilon_L: float) -> float:
    L = inv.L_diam
    return target_epsilon_L / L if L > 0 else target_epsilon_L


def search_worst_case(
    config: AdversarialConfig,
    rng: np.random.Generator | None = None,
) -> tuple[Instance, float]:
    """CMA-ES search maximizing estimated regret at target epsilon·L."""
    import cma

    if rng is None:
        rng = np.random.default_rng(config.base_seed)

    n = config.n
    dim = 4 * n  # Q and R coordinates flattened

    trial_seeds = config.trial_seeds
    if trial_seeds is None:
        trial_seeds = [int(rng.integers(0, 2**31)) for _ in range(config.num_trials)]

    x0 = rng.uniform(0.2, 0.8, size=dim)

    def objective(x: np.ndarray) -> float:
        inst = _coords_to_instance(x, n, config.base_seed)
        inv = compute_invariants(inst)
        eps = target_epsilon_over_L(inst, inv, config.target_epsilon_L)
        regret = estimate_regret(
            inst,
            config.mechanism,
            eps,
            config.num_trials,
            trial_seeds,
            config.base_seed,
            config.mechanism_params,
        )
        return -regret  # minimize negative regret

    opts = cma.CMAOptions()
    opts.set("maxiter", config.max_iter)
    opts.set("verb_disp", 0)
    opts.set("verb_log", 0)
    if config.population_size is not None:
        opts.set("popsize", config.population_size)

    es = cma.CMAEvolutionStrategy(x0, 0.1, opts)

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for sol in solutions:
            sol_proj = _project_unit_disk(sol.reshape(2 * n, 2)).flatten()
            fitnesses.append(objective(sol_proj))
        es.tell(solutions, fitnesses)

    best = _project_unit_disk(es.result.xbest.reshape(2 * n, 2)).flatten()
    best_inst = _coords_to_instance(best, n, config.base_seed)
    inv = compute_invariants(best_inst)
    eps = target_epsilon_over_L(best_inst, inv, config.target_epsilon_L)
    best_regret = estimate_regret(
        best_inst,
        config.mechanism,
        eps,
        config.num_trials,
        trial_seeds,
        config.base_seed,
        config.mechanism_params,
    )
    return best_inst, best_regret
