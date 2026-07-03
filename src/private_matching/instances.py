"""Instance dataclass, generators, and registry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

GENERATORS: dict[str, Callable[..., "Instance"]] = {}


def register_generator(name: str):
    """Decorator to register an instance generator."""

    def decorator(fn: Callable[..., "Instance"]):
        GENERATORS[name] = fn
        return fn

    return decorator


def _stable_instance_id(family: str, params: dict, seed: int) -> str:
    payload = json.dumps({"family": family, "params": params, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Instance:
    """Balanced bipartite matching instance: |Q| = |R| = n."""

    Q: np.ndarray  # (n, 2) public points
    R: np.ndarray  # (n, 2) private points
    family: str
    params: dict[str, Any]
    instance_id: str
    seed: int

    @property
    def n(self) -> int:
        return len(self.Q)


def _make_instance(
    Q: np.ndarray,
    R: np.ndarray,
    family: str,
    params: dict,
    seed: int,
) -> Instance:
    return Instance(
        Q=np.asarray(Q, dtype=float),
        R=np.asarray(R, dtype=float),
        family=family,
        params=params,
        instance_id=_stable_instance_id(family, params, seed),
        seed=seed,
    )


@register_generator("uniform")
def gen_uniform(n: int, rng: np.random.Generator, **params: Any) -> Instance:
    """Points drawn uniformly from the unit square [0,1]^2."""
    seed = int(rng.integers(0, 2**31))
    local_rng = np.random.default_rng(seed)
    Q = local_rng.uniform(0.0, 1.0, size=(n, 2))
    R = local_rng.uniform(0.0, 1.0, size=(n, 2))
    gen_params = {"n": n}
    return _make_instance(Q, R, "uniform", gen_params, seed)


@register_generator("two_gaussian")
def gen_two_gaussian(
    n: int,
    rng: np.random.Generator,
    separation: float = 0.5,
    std: float = 0.15,
    **params: Any,
) -> Instance:
    """Two planar Gaussians for Q and R centers, with jitter."""
    seed = int(rng.integers(0, 2**31))
    local_rng = np.random.default_rng(seed)
    center_q = np.array([0.25, 0.5])
    center_r = np.array([0.25 + separation, 0.5])
    Q = local_rng.normal(center_q, std, size=(n, 2))
    R = local_rng.normal(center_r, std, size=(n, 2))
    Q = np.clip(Q, 0.0, 1.0)
    R = np.clip(R, 0.0, 1.0)
    gen_params = {"n": n, "separation": separation, "std": std}
    return _make_instance(Q, R, "two_gaussian", gen_params, seed)


@register_generator("ring")
def gen_ring(
    n: int,
    rng: np.random.Generator,
    radius: float = 0.4,
    jitter: float = 0.02,
    **params: Any,
) -> Instance:
    """Interleaved points on concentric rings → many near-ties → small mu."""
    seed = int(rng.integers(0, 2**31))
    local_rng = np.random.default_rng(seed)
    angles_q = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_r = angles_q + np.pi / n  # interleave
    center = np.array([0.5, 0.5])
    Q = center + radius * np.column_stack([np.cos(angles_q), np.sin(angles_q)])
    R = center + (radius - jitter) * np.column_stack([np.cos(angles_r), np.sin(angles_r)])
    Q += local_rng.normal(0, jitter * 0.1, Q.shape)
    R += local_rng.normal(0, jitter * 0.1, R.shape)
    gen_params = {"n": n, "radius": radius, "jitter": jitter}
    return _make_instance(Q, R, "ring", gen_params, seed)


@register_generator("lattice")
def gen_lattice(
    n: int,
    rng: np.random.Generator,
    shift: float = 0.05,
    grid_scale: float = 0.9,
    **params: Any,
) -> Instance:
    """Coincident/shifted grids → near-degenerate matchings."""
    seed = int(rng.integers(0, 2**31))
    local_rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(0.1, 0.1 + grid_scale, side)
    ys = np.linspace(0.1, 0.1 + grid_scale, side)
    grid = np.array([[x, y] for x in xs for y in ys])[:n]
    Q = grid.copy()
    R = grid + np.array([shift, shift])
    Q += local_rng.normal(0, shift * 0.1, Q.shape)
    R += local_rng.normal(0, shift * 0.1, R.shape)
    gen_params = {"n": n, "shift": shift, "grid_scale": grid_scale}
    return _make_instance(Q, R, "lattice", gen_params, seed)


@register_generator("block_alpha")
def gen_block_alpha(
    n: int,
    rng: np.random.Generator,
    alpha: float = 1.0,
    stakes_S: float = 5.0,
    h: float = 1.0,
    delta_max: float = 0.2,
    separation_margin: float = 10.0,
    **params: Any,
) -> Instance:
    """Difficulty-graded test bed of far-apart rectangle gadgets (handoff §7).

    K = n/2 blocks; each has two public points on top and two private on the bottom,
    height h and width w_b. The optimal per-block matching is the two vertical edges
    (cost 2h); the only local alternative is the crossed edges, costing
    delta_b ≈ w_b^2 / h more. With blocks separated by D ≫ everything, every near-OPT
    matching is block-diagonal and its excess over OPT is a subset sum sum_{b in S} delta_b.

    Two independent knobs:
      - **stakes** via separation D = stakes_S * OPT  (RAND/OPT ≈ stakes_S).
      - **ruggedness** via the margin law: delta_b = delta_max * U^(1/alpha) draws from
        rho(delta) ∝ delta^(alpha-1), so N(x) ∝ x^alpha near the bottom. alpha is thus an
        *input*, not a fitted quantity.
    """
    if n % 2 != 0:
        raise ValueError(f"block_alpha requires even n, got {n}")
    if alpha <= 0:
        raise ValueError(f"block_alpha requires alpha > 0, got {alpha}")
    seed = int(rng.integers(0, 2**31))
    local_rng = np.random.default_rng(seed)
    K = n // 2

    U = local_rng.uniform(0.0, 1.0, size=K)
    deltas = delta_max * U ** (1.0 / alpha)
    widths = np.sqrt(h * deltas)

    opt_approx = 2.0 * K * h
    D = stakes_S * opt_approx

    # Decouple the two axes: the near-OPT band spans <= K*delta_max; cross-block edges
    # cost ~ D. Require D to dominate so stakes doesn't disturb alpha (handoff §7).
    near_opt_band = K * delta_max
    required = separation_margin * max(near_opt_band, h, float(widths.max()))
    if D < required:
        raise ValueError(
            f"block_alpha separation too small: D={D:.3g} < {required:.3g}. "
            f"Increase stakes_S (>= {required / opt_approx:.3g}) or reduce delta_max."
        )

    cols = int(np.ceil(np.sqrt(K)))
    Q_pts: list[list[float]] = []
    R_pts: list[list[float]] = []
    for b in range(K):
        cx = (b % cols) * D
        cy = (b // cols) * D
        wb = float(widths[b])
        Q_pts.append([cx - wb / 2.0, cy + h / 2.0])
        Q_pts.append([cx + wb / 2.0, cy + h / 2.0])
        R_pts.append([cx - wb / 2.0, cy - h / 2.0])
        R_pts.append([cx + wb / 2.0, cy - h / 2.0])

    gen_params = {
        "n": n,
        "alpha": alpha,
        "stakes_S": stakes_S,
        "h": h,
        "delta_max": delta_max,
        "D": D,
        "deltas": [float(x) for x in deltas],
    }
    return _make_instance(np.array(Q_pts), np.array(R_pts), "block_alpha", gen_params, seed)


def generate_instance(
    family: str,
    n: int,
    rng: np.random.Generator,
    **params: Any,
) -> Instance:
    """Generate an instance from a registered family."""
    if family not in GENERATORS:
        raise KeyError(f"Unknown instance family: {family!r}. Available: {list(GENERATORS)}")
    return GENERATORS[family](n, rng, **params)


def generate_instances(
    families: list[dict[str, Any]],
    num_instances: int,
    n: int,
    base_seed: int,
) -> list[Instance]:
    """Generate instances across families according to config."""
    rng = np.random.default_rng(base_seed)
    instances: list[Instance] = []
    for fam_cfg in families:
        family = fam_cfg["name"]
        params = {k: v for k, v in fam_cfg.items() if k != "name"}
        count = fam_cfg.get("count", num_instances)
        for _ in range(count):
            inst_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            instances.append(generate_instance(family, n, inst_rng, **params))
    return instances
