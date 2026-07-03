"""Config-driven experiment runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import SeedSequence

from private_matching.config import ExperimentConfig
from private_matching.difficulty import instance_difficulty
from private_matching.instances import Instance, generate_instances
from private_matching.mechanisms import MECHANISMS
from private_matching.metrics import InstanceInvariants, compute_invariants, evaluate_assignment


def get_git_sha(allow_dirty: bool = False) -> str:
    """Return current git SHA, optionally suffixed with -dirty."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        if dirty and not allow_dirty:
            raise RuntimeError(
                "Git working tree is dirty. Commit changes or pass --allow-dirty."
            )
        if dirty:
            return f"{sha}-dirty"
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {}
    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "pyarrow", "pyyaml"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _stable_hash(s: str) -> int:
    """Process-stable hash of a string.

    Unlike the built-in ``hash()``, which is randomized per interpreter (PYTHONHASHSEED),
    this is identical across processes — required for reproducible seeding under
    ProcessPoolExecutor, whose spawned workers would otherwise each derive a different RNG
    for the same (instance, epsilon, trial), breaking reproducibility and CRN.
    """
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def _trial_rng(
    base_seed: int,
    instance_id: str,
    epsilon_idx: int,
    trial: int,
    crn: bool,
    trial_seeds: list[int] | None,
) -> np.random.Generator:
    inst_hash = _stable_hash(instance_id)
    if crn and trial_seeds is not None:
        trial_seed = trial_seeds[trial % len(trial_seeds)]
        seeds = SeedSequence([base_seed, inst_hash, trial_seed]).spawn(1)
    else:
        seeds = SeedSequence([base_seed, inst_hash, epsilon_idx, trial]).spawn(1)
    return np.random.default_rng(seeds[0])


def _run_single_trial(args: tuple) -> dict[str, Any]:
    """Worker function for parallel execution."""
    (
        inst_dict,
        inv_dict,
        mech_name,
        mech_params,
        epsilon,
        epsilon_idx,
        trial,
        base_seed,
        crn,
        trial_seeds,
        L_type,
        git_sha,
        config_hash,
    ) = args

    inst = Instance(**inst_dict)
    inv = InstanceInvariants(**inv_dict)
    L = inv.L_diam if L_type == "diam" else inv.L_nn
    epsilon_L = epsilon * L

    mech_cls = MECHANISMS[mech_name]
    mech = mech_cls(**mech_params)
    rng = _trial_rng(base_seed, inst.instance_id, epsilon_idx, trial, crn, trial_seeds)
    sigma = mech.match(inst, epsilon, rng)
    metrics = evaluate_assignment(inst, sigma, inv)

    return {
        "mechanism": mech_name,
        "family": inst.family,
        "instance_id": inst.instance_id,
        "n": inst.n,
        "epsilon": epsilon,
        "epsilon_L": epsilon_L,
        "L_diam": inv.L_diam,
        "L_nn": inv.L_nn,
        "trial": trial,
        "seed": int(rng.integers(0, 2**31)),
        "degenerate": inv.degenerate,
        **instance_difficulty(inst),
        **metrics,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _instance_to_dict(inst: Instance) -> dict:
    return {
        "Q": inst.Q,
        "R": inst.R,
        "family": inst.family,
        "params": inst.params,
        "instance_id": inst.instance_id,
        "seed": inst.seed,
    }


def _inv_to_dict(inv: InstanceInvariants) -> dict:
    d = asdict(inv)
    return d


def _progress_line(done: int, total: int, start: float) -> str:
    elapsed = time.monotonic() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")
    pct = 100.0 * done / total
    return f"  [{done}/{total}] {pct:.0f}% | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"


def run_experiment(
    config: ExperimentConfig,
    allow_dirty: bool = False,
    instances: list[Instance] | None = None,
) -> pd.DataFrame:
    """Run full experiment sweep and write results."""
    git_sha = get_git_sha(allow_dirty=allow_dirty)
    config_hash = config.config_hash()

    if instances is None:
        instances = generate_instances(
            config.families, config.num_instances, config.n, config.base_seed
        )

    invariants = {
        inst.instance_id: compute_invariants(inst, margin_tol=config.margin_tol)
        for inst in instances
    }

    trial_seeds = config.trial_seeds
    if config.crn and trial_seeds is None:
        rng = np.random.default_rng(config.base_seed)
        trial_seeds = [int(rng.integers(0, 2**31)) for _ in range(config.num_trials)]

    rows: list[dict[str, Any]] = []
    tasks: list[tuple] = []

    for inst in instances:
        inv = invariants[inst.instance_id]
        for mech_cfg in config.mechanisms:
            mech_name = mech_cfg["name"]
            mech_params = {k: v for k, v in mech_cfg.items() if k != "name"}
            for epsilon_idx, epsilon in enumerate(config.epsilons):
                for trial in range(config.num_trials):
                    tasks.append(
                        (
                            _instance_to_dict(inst),
                            _inv_to_dict(inv),
                            mech_name,
                            mech_params,
                            float(epsilon),
                            epsilon_idx,
                            trial,
                            config.base_seed,
                            config.crn,
                            trial_seeds,
                            config.L_type,
                            git_sha,
                            config_hash,
                        )
                    )

    start = time.monotonic()
    report_every = max(1, len(tasks) // 40)
    mode = (
        f"parallel, {config.max_workers} workers"
        if (config.parallel and len(tasks) > 1)
        else "serial"
    )
    print(f"Running {len(tasks)} trials ({mode})...", flush=True)

    def _record(i: int, row: dict[str, Any]) -> None:
        rows.append(row)
        if i % report_every == 0 or i == len(tasks):
            print(_progress_line(i, len(tasks), start), flush=True)

    if config.parallel and len(tasks) > 1:
        # Coarse chunks so per-task IPC/pickling doesn't dominate the (often sub-ms) trials.
        chunksize = max(1, min(256, len(tasks) // (config.max_workers * 4)))
        with ProcessPoolExecutor(max_workers=config.max_workers) as pool:
            for i, row in enumerate(
                pool.map(_run_single_trial, tasks, chunksize=chunksize), start=1
            ):
                _record(i, row)
    else:
        for i, task in enumerate(tasks, start=1):
            _record(i, _run_single_trial(task))

    df = pd.DataFrame(rows)

    out_dir = Path(config.output_dir) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{config_hash}_{git_sha[:8]}.parquet"
    df.to_parquet(parquet_path, index=False)

    manifest_path = Path(config.output_dir) / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_entry = {
        "config": config.resolved(),
        "config_hash": config_hash,
        "git_sha": git_sha,
        "package_versions": package_versions(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "parquet": str(parquet_path),
    }
    with manifest_path.open("a") as f:
        f.write(json.dumps(manifest_entry) + "\n")

    return df
