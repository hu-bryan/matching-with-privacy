"""Experiment configuration dataclasses and YAML loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Fully-resolved experiment specification."""

    name: str = "default"
    n: int = 20
    num_instances: int = 10
    families: list[dict[str, Any]] = field(
        default_factory=lambda: [{"name": "uniform", "count": 10}]
    )
    mechanisms: list[dict[str, Any]] = field(
        default_factory=lambda: [{"name": "local"}]
    )
    epsilons: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    num_trials: int = 20
    base_seed: int = 42
    crn: bool = False
    trial_seeds: list[int] | None = None
    parallel: bool = False
    max_workers: int = 4
    L_type: str = "diam"  # "diam" or "nn"
    margin_tol: float = 1e-12
    adversarial: dict[str, Any] | None = None
    output_dir: str = "results"

    def resolved(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.resolved(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


DEFAULTS: dict[str, Any] = {
    "name": "default",
    "n": 20,
    "num_instances": 10,
    "families": [{"name": "uniform", "count": 10}],
    "mechanisms": [{"name": "local"}],
    "epsilons": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "num_trials": 20,
    "base_seed": 42,
    "crn": False,
    "trial_seeds": None,
    "parallel": False,
    "max_workers": 4,
    "L_type": "diam",
    "margin_tol": 1e-12,
    "adversarial": None,
    "output_dir": "results",
}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate experiment config from YAML."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    merged = _deep_merge(DEFAULTS, raw)
    return ExperimentConfig(**{k: merged[k] for k in DEFAULTS})
