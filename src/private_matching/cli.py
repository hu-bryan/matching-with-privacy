"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from private_matching.adversarial import AdversarialConfig, search_worst_case
from private_matching.config import load_config
from private_matching.experiment import run_experiment
from private_matching.plotting import (
    plot_regret_vs_margin,
    plot_regret_vs_n,
    plot_utility_vs_epsilon,
    plot_worst_case_envelope,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private bipartite matching experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run experiment from config")
    run_p.add_argument("--config", required=True, help="Path to YAML config")
    run_p.add_argument("--allow-dirty", action="store_true", help="Allow dirty git tree")

    adv_p = sub.add_parser("adversarial", help="Run adversarial instance search")
    adv_p.add_argument("--n", type=int, default=10)
    adv_p.add_argument("--epsilon-L", type=float, default=1.0)
    adv_p.add_argument("--mechanism", default="local")
    adv_p.add_argument("--max-iter", type=int, default=50)
    adv_p.add_argument("--seed", type=int, default=0)

    plot_p = sub.add_parser("plot", help="Generate figures from latest results")
    plot_p.add_argument("--parquet", help="Path to results parquet")
    plot_p.add_argument("--family", default="uniform")
    plot_p.add_argument("--epsilon-L", type=float, default=1.0)
    plot_p.add_argument("--output-dir", default="results/figures")

    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(args.config)
        df = run_experiment(config, allow_dirty=args.allow_dirty)
        print(f"Wrote {len(df)} rows")
        return 0

    if args.command == "adversarial":
        import numpy as np

        cfg = AdversarialConfig(
            n=args.n,
            target_epsilon_L=args.epsilon_L,
            mechanism=args.mechanism,
            max_iter=args.max_iter,
            base_seed=args.seed,
        )
        inst, regret = search_worst_case(cfg, np.random.default_rng(args.seed))
        print(f"Best adversarial instance: regret={regret:.4f}, mu={inst.params}")
        return 0

    if args.command == "plot":
        import pandas as pd

        parquet = args.parquet
        if parquet is None:
            manifest = Path("results/manifest.jsonl")
            if not manifest.exists():
                print("No results found", file=sys.stderr)
                return 1
            last = manifest.read_text().strip().split("\n")[-1]
            import json

            parquet = json.loads(last)["parquet"]

        df = pd.read_parquet(parquet)
        out = Path(args.output_dir)
        plot_utility_vs_epsilon(df, args.family, out / f"utility_{args.family}.png")
        plot_regret_vs_margin(df, args.epsilon_L, out / "regret_vs_margin.png")
        plot_worst_case_envelope(df, out / "regret_vs_epsilonL_sampled.png")
        try:
            plot_regret_vs_n(df, out / "regret_vs_n.png")
        except ValueError:
            pass
        print(f"Figures written to {out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
