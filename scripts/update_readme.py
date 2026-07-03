#!/usr/bin/env python3
"""Regenerate the README results block and figures from the latest experiment run.

Pipeline stage 3 of 3: read the latest raw results parquet, render a summary table and
figures, and rewrite the marked regions of README.md. Never hand-edit inside the
``RESULTS`` / ``FIGURES`` markers — run this script instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from private_matching.plotting import plot_regret_vs_epsilon, plot_utility_vs_epsilon

README_PATH = Path("README.md")
MANIFEST_PATH = Path("results/manifest.jsonl")
FIGURES_DIR = Path("results/figures")

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"
FIGURES_START = "<!-- FIGURES:START -->"
FIGURES_END = "<!-- FIGURES:END -->"
DESIGN_START = "<!-- DESIGN:START -->"
DESIGN_END = "<!-- DESIGN:END -->"


def load_latest_parquet() -> tuple[pd.DataFrame, dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No manifest at results/manifest.jsonl — run an experiment first.")
    lines = MANIFEST_PATH.read_text().strip().split("\n")
    entry = json.loads(lines[-1])
    df = pd.read_parquet(entry["parquet"])
    return df, entry


def _pluralize(n: int, singular: str, plural: str | None = None) -> str:
    if n == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def summary_table(df: pd.DataFrame, ref_epsilon: Iterable[float] = (0.5, 1.0, 2.0, 5.0)) -> str:
    """One row per (mechanism, family[, alpha]): median ratios + regret at reference ε."""
    ref_epsilon = list(ref_epsilon)
    has_alpha = "alpha" in df.columns and df["alpha"].notna().any()
    group_cols = ["mechanism", "family"] + (["alpha"] if has_alpha else [])

    rows: list[dict[str, str]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {c: k for c, k in zip(group_cols, keys)}
        if "alpha" in row:
            row["alpha"] = "—" if pd.isna(row["alpha"]) else f"{float(row['alpha']):g}"
        row["median_competitive"] = f"{sub['competitive_ratio'].median():.3f}"
        row["median_regret"] = f"{sub['regret_ratio'].median():.3f}"
        for e in ref_epsilon:
            near = sub[(sub["epsilon"] - e).abs() < 1e-9]
            row[f"regret@ε={e:g}"] = (
                f"{near['regret_ratio'].median():.3f}" if not near.empty else "—"
            )
        rows.append(row)

    if not rows:
        return "_No results yet._"

    rows.sort(key=lambda r: tuple(str(r.get(c, "")) for c in group_cols))
    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)
    return "\n".join([header, sep, body])


def _fmt_params(d: dict) -> str:
    items = [f"{k}={v}" for k, v in d.items()]
    return ", ".join(items) if items else "—"


def design_block(config: dict) -> str:
    """Render the experiment design + all hyperparameters from the run's recorded config."""
    fams = config.get("families", []) or []
    mechs = config.get("mechanisms", []) or []
    eps = config.get("epsilons", []) or []

    header = (
        f"- **n** = {config.get('n')} · **trials/config** = {config.get('num_trials')} · "
        f"**base seed** = {config.get('base_seed')} · **length scale** = {config.get('L_type')}"
        + (" · **CRN** on" if config.get("crn") else "")
    )
    sweep = "- **Privacy sweep**: ε ∈ {" + ", ".join(f"{float(e):g}" for e in eps) + "}"

    fam_lines = ["| family | count | params |", "| --- | --- | --- |"]
    for f in fams:
        f = dict(f)
        name = f.pop("name", "?")
        count = f.pop("count", config.get("num_instances"))
        fam_lines.append(f"| `{name}` | {count} | {_fmt_params(f)} |")

    mech_lines = ["| mechanism | hyperparameters |", "| --- | --- |"]
    for md in mechs:
        md = dict(md)
        name = md.pop("name", "?")
        mech_lines.append(f"| `{name}` | {_fmt_params(md)} |")

    return "\n".join(
        [
            header,
            sweep,
            "",
            "**Instance families**",
            "",
            *fam_lines,
            "",
            "**Mechanisms & hyperparameters**",
            "",
            *mech_lines,
        ]
    )


def replace_block(text: str, start: str, end: str, new_content: str) -> str:
    if start not in text or end not in text:
        return text + f"\n{start}\n{new_content}\n{end}\n"
    i = text.index(start) + len(start)
    j = text.index(end, i)
    return text[:i] + "\n" + new_content + "\n" + text[j:]


def main() -> int:
    df, entry = load_latest_parquet()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure_links: list[str] = []
    for fam in sorted(df["family"].unique()):
        regret_path = plot_regret_vs_epsilon(df, fam, FIGURES_DIR / f"regret_{fam}.png")
        figure_links.append(f"- [{fam}: regret vs privacy]({regret_path.as_posix()})")
        try:
            util_path = plot_utility_vs_epsilon(df, fam, FIGURES_DIR / f"utility_{fam}.png")
            figure_links.append(f"- [{fam}: competitive ratio]({util_path.as_posix()})")
        except ValueError:
            pass

    table = summary_table(df)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parquet_ref = Path(entry["parquet"]).as_posix()
    git_ref = entry["git_sha"][:8]
    n_mech = df["mechanism"].nunique()
    n_fam = df["family"].nunique()
    results_block = (
        f"_Auto-generated from `{parquet_ref}` (git `{git_ref}`, {timestamp})_\n\n"
        f"**{len(df)} trials** across {n_mech} {_pluralize(n_mech, 'mechanism')}, "
        f"{n_fam} {_pluralize(n_fam, 'family', 'families')}.\n\n"
        f"Regret ratio: 0 = optimal, 1 = as bad as a random matching.\n\n"
        f"{table}"
    )
    figures_block = "\n".join(figure_links)

    readme = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    readme = replace_block(readme, DESIGN_START, DESIGN_END, design_block(entry["config"]))
    readme = replace_block(readme, RESULTS_START, RESULTS_END, results_block)
    readme = replace_block(readme, FIGURES_START, FIGURES_END, figures_block)
    README_PATH.write_text(readme, encoding="utf-8")
    print(f"Updated {README_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
