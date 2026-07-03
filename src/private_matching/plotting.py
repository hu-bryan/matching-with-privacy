"""Figure generation from tidy results tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _aggregate_per_instance(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """One row per (mechanism, instance, …): instance invariants + mean metric over trials."""
    keys = ["mechanism", "instance_id"] + (group_cols or [])
    return (
        df.groupby(keys, as_index=False)
        .agg(
            margin_norm=("margin_norm", "first"),
            **{value_col: (value_col, "mean")},
        )
    )


def plot_utility_vs_epsilon(
    df: pd.DataFrame,
    family: str,
    output_path: str | Path,
    use_epsilon_L: bool = False,
    quantile: float = 0.9,
) -> Path:
    """Competitive ratio vs epsilon with median, IQR, and upper quantile.

    If the family carries a difficulty knob (``alpha``, e.g. ``block_alpha``), the plot is
    faceted into one panel per alpha, matching ``plot_regret_vs_epsilon``. 1 = OPT.
    """
    from matplotlib.ticker import FuncFormatter

    output_path = Path(output_path)
    _ensure_dir(output_path)

    sub = df[df["family"] == family].copy()
    if sub.empty:
        raise ValueError(f"No data for family {family!r}")

    x_col = "epsilon_L" if use_epsilon_L else "epsilon"
    x_label = r"$\varepsilon \cdot L$" if use_epsilon_L else r"$\varepsilon$"
    p_label = f"p{int(round(quantile * 100))}"

    if "alpha" in sub.columns and sub["alpha"].notna().any():
        alphas: list[float | None] = sorted(sub.loc[sub["alpha"].notna(), "alpha"].unique())
    else:
        alphas = [None]

    ncols = len(alphas)
    fig, axes = plt.subplots(
        1, ncols, figsize=(4.8 * ncols, 4.5), sharey=True, squeeze=False
    )
    row = axes[0]

    for ax, a in zip(row, alphas):
        if a is None:
            panel = sub
            title = family
        else:
            panel = sub[sub["alpha"] == a]
            title = rf"{family}, $\alpha={a:g}$"

        for mech in sorted(panel["mechanism"].unique()):
            msub = panel[panel["mechanism"] == mech]
            grouped = msub.groupby(x_col)["competitive_ratio"]
            xs = sorted(msub[x_col].unique())
            medians = [grouped.get_group(x).median() for x in xs]
            q25 = [grouped.get_group(x).quantile(0.25) for x in xs]
            q75 = [grouped.get_group(x).quantile(0.75) for x in xs]
            q_hi = [grouped.get_group(x).quantile(quantile) for x in xs]
            (line,) = ax.plot(xs, medians, label=mech, marker="o")
            ax.fill_between(xs, q25, q75, alpha=0.2, color=line.get_color())
            ax.plot(
                xs,
                q_hi,
                linestyle="--",
                color=line.get_color(),
                alpha=0.7,
                label=f"{mech} ({p_label})",
            )

        ax.axhline(1.0, color="black", linestyle=":", label="OPT (ratio=1)")
        rand_med = float((panel["rand"] / panel["opt"]).median())
        if np.isfinite(rand_med):
            ax.axhline(rand_med, color="gray", linestyle="--", label="Random (median)")

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(x_label)
        ax.set_title(title)

    row[0].set_ylabel("Competitive ratio")
    row[0].legend(fontsize=8)
    fig.suptitle(f"Utility vs privacy — {family}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_regret_vs_margin(
    df: pd.DataFrame,
    epsilon_L: float,
    output_path: str | Path,
    band: tuple[float, float] | None = None,
) -> Path:
    """Scatter mean regret per instance vs mu/OPT at a fixed epsilon·L band."""
    output_path = Path(output_path)
    _ensure_dir(output_path)

    if band is None:
        band = (0.9 * epsilon_L, 1.1 * epsilon_L)
    lo, hi = band
    sub = df[(df["epsilon_L"] >= lo) & (df["epsilon_L"] <= hi)].copy()
    if sub.empty:
        raise ValueError(f"No data with epsilon_L in [{lo}, {hi}]")

    # One point per instance: (margin_norm, mean regret over trials)
    agg = _aggregate_per_instance(sub, "regret_ratio")

    fig, ax = plt.subplots(figsize=(8, 5))
    mechanisms = sorted(agg["mechanism"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(mechanisms)))

    for mech, color in zip(mechanisms, colors):
        msub = agg[agg["mechanism"] == mech]
        ax.scatter(
            msub["margin_norm"],
            msub["regret_ratio"],
            alpha=0.5,
            s=25,
            color=color,
            label=mech,
        )
        if len(msub) >= 5:
            bins = np.linspace(msub["margin_norm"].min(), msub["margin_norm"].max(), 12)
            digitized = np.digitize(msub["margin_norm"], bins)
            xs, ys = [], []
            for b in range(1, len(bins)):
                mask = digitized == b
                if mask.sum() >= 2:
                    xs.append(msub.loc[mask, "margin_norm"].median())
                    ys.append(msub.loc[mask, "regret_ratio"].median())
            if xs:
                order = np.argsort(xs)
                ax.plot(
                    np.array(xs)[order],
                    np.array(ys)[order],
                    color=color,
                    linewidth=2,
                )

    ax.set_xlabel(r"Normalized margin $\mu / \mathrm{OPT}$")
    ax.set_ylabel("Regret ratio (mean over trials)")
    ax.set_title(rf"Regret vs $\mu$ at $\varepsilon L \in [{lo:.2f}, {hi:.2f}]$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_worst_case_envelope(
    df: pd.DataFrame,
    output_path: str | Path,
    adversarial_family: str = "adversarial",
) -> Path:
    """Regret vs epsilon·L for sampled instances (per-instance scatter + median trend).

    TODO: overlay adversarial (CMA-ES) worst-case curve once Phase 4 lands.
    """
    output_path = Path(output_path)
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_col = "epsilon_L"

    adv = df[df["family"] == adversarial_family]
    sampled = df[df["family"] != adversarial_family]

    # Per-instance mean regret at each epsilon·L
    agg = _aggregate_per_instance(sampled, "regret_ratio", group_cols=[x_col])

    for mech in sorted(agg["mechanism"].unique()):
        msub = agg[agg["mechanism"] == mech]
        ax.scatter(
            msub[x_col],
            msub["regret_ratio"],
            alpha=0.12,
            s=12,
            label=f"{mech} (instances)",
        )
        trend = (
            msub.groupby(x_col)["regret_ratio"]
            .median()
            .sort_index()
        )
        ax.plot(
            trend.index,
            trend.values,
            marker="o",
            linewidth=2,
            label=f"{mech} (median)",
        )

    if not adv.empty:
        adv_agg = _aggregate_per_instance(adv, "regret_ratio", group_cols=[x_col])
        for mech in sorted(adv_agg["mechanism"].unique()):
            msub = adv_agg[adv_agg["mechanism"] == mech]
            trend = (
                msub.groupby(x_col)["regret_ratio"]
                .median()
                .sort_index()
            )
            ax.plot(
                trend.index,
                trend.values,
                linestyle="--",
                marker="x",
                label=f"{mech} (adversarial)",
            )

    ax.set_xlabel(r"$\varepsilon \cdot L$")
    ax.set_ylabel("Regret ratio")
    ax.set_title("Regret vs $\\varepsilon L$ — sampled instances")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_regret_vs_n(
    df: pd.DataFrame,
    output_path: str | Path,
    mechanism: str = "dual_sinkhorn",
    epsilon_L: float = 1.0,
    tol: float = 0.2,
) -> Path:
    """Regret vs n for dual_sinkhorn's n-driven degradation."""
    output_path = Path(output_path)
    _ensure_dir(output_path)

    sub = df[
        (df["mechanism"] == mechanism)
        & (np.abs(df["epsilon_L"] - epsilon_L) < tol * max(epsilon_L, 1e-9))
    ]
    if sub.empty:
        raise ValueError(f"No data for {mechanism} near epsilon_L={epsilon_L}")

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = sub.groupby("n")["regret_ratio"]
    ns = sorted(sub["n"].unique())
    medians = [grouped.get_group(n).median() for n in ns]
    ax.plot(ns, medians, marker="o")
    ax.set_xlabel("n")
    ax.set_ylabel("Regret ratio (median)")
    ax.set_title(f"{mechanism}: regret vs n at $\\varepsilon L \\approx {epsilon_L}$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_regret_vs_epsilon(
    df: pd.DataFrame,
    family: str,
    output_path: str | Path,
    use_epsilon_L: bool = False,
) -> Path:
    """Median regret ratio vs epsilon, one line per mechanism.

    If the family carries a difficulty knob (``alpha``, e.g. ``block_alpha``), the plot
    is faceted into one panel per alpha so you can read how the mechanism comparison
    shifts with instance difficulty. 0 = OPT, 1 = Random.
    """
    from matplotlib.ticker import FuncFormatter

    output_path = Path(output_path)
    _ensure_dir(output_path)

    sub = df[df["family"] == family].copy()
    if sub.empty:
        raise ValueError(f"No data for family {family!r}")

    x_col = "epsilon_L" if use_epsilon_L else "epsilon"
    x_label = r"$\varepsilon \cdot L$" if use_epsilon_L else r"$\varepsilon$"

    if "alpha" in sub.columns and sub["alpha"].notna().any():
        alphas: list[float | None] = sorted(sub.loc[sub["alpha"].notna(), "alpha"].unique())
    else:
        alphas = [None]

    ncols = len(alphas)
    fig, axes = plt.subplots(
        1, ncols, figsize=(4.8 * ncols, 4.5), sharey=True, squeeze=False
    )
    row = axes[0]

    for ax, a in zip(row, alphas):
        if a is None:
            panel = sub
            title = family
        else:
            panel = sub[sub["alpha"] == a]
            title = rf"{family}, $\alpha={a:g}$"
        for mech in sorted(panel["mechanism"].unique()):
            m = panel[panel["mechanism"] == mech]
            g = m.groupby(x_col)["regret_ratio"].median().sort_index()
            ax.plot(g.index, g.values, marker="o", label=mech)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1, label="OPT")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Random")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(x_label)
        ax.set_title(title)

    row[0].set_ylabel("Regret ratio (median over trials)")
    row[0].legend(fontsize=8)
    fig.suptitle(f"Regret vs privacy — {family}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
