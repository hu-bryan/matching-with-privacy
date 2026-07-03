# Private Metric-DP Bipartite Matching

Empirical measurement of the **privacy–utility tradeoff** for metric-differentially-private
bipartite matching mechanisms. We generate planar point sets (including a difficulty-graded
family), run three mechanisms across a privacy sweep, record the raw results, and render this
README's table and figures from them.

## Problem Setup

- Public set \(Q = \{q_i\}_{i=1}^n \subset \mathbb{R}^2\) (e.g. customers). **Public / not protected.**
- Private set \(R = \{r_j\}_{j=1}^n \subset \mathbb{R}^2\) (e.g. driver home locations). **Protected.**
- Balanced: \(|Q| = |R| = n\), so a perfect matching always exists.
- Cost \(c_{ij} = \|q_i - r_j\|_2\); an assignment is a bijection \(\sigma:[n]\to[n]\) with cost
  \(u(\sigma \mid R) = \sum_i c_{i,\sigma(i)}\); \(\mathrm{OPT} = \min_\sigma u\) (Hungarian).

**Metric privacy.** With the product metric \(d(R,R') = \sum_j \|r_j - r'_j\|_2\), a mechanism
\(\mathcal{M}\) is **\(\varepsilon d\)-private** if
\(\Pr[\mathcal{M}(R)\in S] \le e^{\varepsilon d(R,R')}\Pr[\mathcal{M}(R')\in S]\) for all \(R,R',S\).
The planar Laplace (Geo-Ind) mechanism perturbs each point by \(\nu \sim \mathrm{Gamma}(2,\varepsilon)\)
at a uniform angle.

> Privacy is a worst-case, **proven** property — the experiments measure **utility only**.

**Scale symmetry.** Scaling all points by \(\lambda\) and \(\varepsilon \to \varepsilon/\lambda\)
gives an identical problem, so the meaningful privacy axis is the dimensionless \(\varepsilon\!\cdot\!L\)
(\(L\) = a length scale such as the diameter).

## Utility metrics

- **Competitive ratio** \(= E[u(\tilde\sigma)]/\mathrm{OPT} \ge 1\) — "×OPT cost" (poster-style curves).
- **Regret ratio** \(= \dfrac{E[u(\tilde\sigma)] - \mathrm{OPT}}{\mathrm{RAND} - \mathrm{OPT}}\) — 0 at OPT,
  1 at a random matching; comparable across instances. \(\mathrm{RAND} = \frac1n\sum_{i,j} c_{ij}\).

## Instance families

| Family | Kind | Notes |
|--------|------|-------|
| `uniform` | plausible | points i.i.d. in the unit square |
| `two_gaussian` | plausible | \(Q,R\) from two separated planar Gaussians |
| `ring`, `lattice` | near-degenerate | many near-ties (small optimality margin) |
| `block_alpha` | **difficulty-graded** | far-apart rectangle gadgets with tunable difficulty |

`block_alpha` is the knob for generating instances at a chosen difficulty:

- **`alpha`** — near-optimum *ruggedness*. Small `alpha` packs many near-optimal competitors just
  above OPT (hard to single out the winner); large `alpha` isolates the optimum. `alpha` is a
  generator input, so difficulty is set, not fitted.
- **`stakes_S`** — sets `RAND/OPT`, i.e. how much an average mistake costs (how much the instance
  actually tests). Blocks are separated by \(D = \texttt{stakes\_S}\cdot\mathrm{OPT}\) so the two knobs
  decouple.

## Mechanisms

| Name | Description |
|------|-------------|
| `local` | Planar Laplace on \(R\), then Hungarian (baseline, has theory) |
| `auction` | Private ascending-price auction (exponential-mechanism bids) + local cleanup |
| `dual_sinkhorn` | Noisy smoothed kernel + Sinkhorn + Birkhoff randomized rounding |

## How it works: pseudocode & privacy math

Every guarantee below is **metric privacy** (εd-privacy / Geo-Indistinguishability): for the product
metric \(d(R,R') = \sum_j \|r_j - r'_j\|\), a mechanism \(\mathcal{M}\) is εd-private if
\(\Pr[\mathcal{M}(R)\in S] \le e^{\varepsilon d(R,R')}\Pr[\mathcal{M}(R')\in S]\) for all \(R,R',S\).
Protection scales with distance (learn the city, not the block). We rely on two facts:
**post-processing** (any function of an εd-private output is still εd-private) and **parallel
composition** (independent mechanisms on disjoint coordinates combine at the max ε under the product
metric).

Two noise primitives appear, and the difference is the crux of the metric-privacy question:

- **Laplace / planar Laplace → pure εd (δ = 0), unconditionally.** The Laplace privacy loss is exactly
  linear in \(\|\cdot\|_1\), which matches the metric, so calibrating the noise to the ℓ1 sensitivity
  gives a guarantee that holds for every pair at every distance. This is the default everywhere.
- **Gaussian → approximate (εd, δ), near inputs only.** `(ε,δ)` generalizes to approximate metric
  privacy \(\Pr[\mathcal{M}(R)\in S] \le e^{\varepsilon d}\Pr[\mathcal{M}(R')\in S] + \delta\). The
  Gaussian mechanism achieves it, but the classic calibration \(\sigma = \sqrt{2\ln(1.25/\delta)}\,L/\varepsilon\)
  only certifies (εd, δ) while \(\varepsilon d < 1\); for far pairs the achievable δ drifts to 1
  (vacuous — as does \(e^{\varepsilon d}\) itself). So it is **not** the clean unconditional guarantee
  Laplace gives, and it is off by default. Treat its (ε,δ) as *approximate* metric privacy.

### `local` — pure εd
```
for each private point r_j:
    ν ~ Gamma(shape=2, rate=ε);   θ ~ Uniform[0, 2π]
    r̃_j = r_j + ν·(cos θ, sin θ)          # planar Laplace, ε-Geo-Ind per point
σ = Hungarian( ‖q_i − r̃_j‖ )              # post-processing
```
Each planar Laplace is ε-Geo-Ind; parallel composition over the n independent coordinates plus
post-processing by Hungarian ⇒ **εd-private**. In code: `planar_laplace` draws `Gamma(2, 1/ε)` radii,
`match_hungarian` solves the assignment.

### `auction` — pure εd
Private drivers bid on public customers; contested items' prices ascend.
```
ε1 = ε/(2m);   ε2 = ε/2;   prices p[q] = 0
while some driver r is unassigned with < m bids:
    score[q] = ‖q − r‖ + p[q]                        # 1-Lipschitz in r ⇒ Δ = 1
    q* ~ sample ∝ exp( −ε1 · score[q] / (2Δ) )       # metric exponential mechanism
    if q* is held by r':  unassign r'                # eviction
    assign r → q*;   p[q*] += α
match the leftover drivers/customers with `local` at budget ε2
```
Math: the metric exponential mechanism sampling \(\propto \exp(-\varepsilon_1\, q/(2\Delta))\) with a
Δ-Lipschitz score is **ε1·d-private per draw** — the `2Δ` denominator (not Δ) is essential, because the
data-dependent normalizer contributes a second factor \(Z(R)/Z(R') \le e^{\varepsilon_1 d/2}\). Sequential
composition over ≤ m bids ⇒ ≤ (m·ε1)·d = (ε/2)·d per driver; distinct drivers touch disjoint
coordinates (parallel composition); the cleanup spends ε2 = ε/2. A driver in both phases spends
ε/2 + ε/2 ⇒ **εd-private**. In code the `2Δ` shows up as `exponential_sample(scores, eps1, 2*Δ, rng)`;
the hard cap of m bids guarantees termination in ≤ n·m bids.

### `dual_sinkhorn` — pure εd (Laplace); approximate (εd,δ) (Gaussian)
Only the kernel touches R; everything after is post-processing.
```
K[i,j] = exp( −β · min(‖q_i − r_j‖, B) )              # clip at B ⇒ bounded sensitivity
K̃ = clip( K + Laplace(scale = n·β / ε), 0, 1 )        # εd-private release
P = Sinkhorn(K̃, L iters);   σ ~ BirkhoffRound(P)       # post-processing — SAMPLE, not argmax
```
Math: moving one driver \(r_j\) by \(\delta\) changes only column j, and each of its n entries moves by
≤ βδ (the map \(r_j \mapsto \exp(-\beta\min(\|q-r_j\|,B))\) is β-Lipschitz), so
\(\|K(R)-K(R')\|_1 \le n\beta\, d(R,R')\) — the per-driver **ℓ1 sensitivity is n·β**. Laplace noise at
scale n·β/ε makes the ℓ1 privacy loss ≤ ε·d ⇒ **εd-private**; clip and Sinkhorn/Birkhoff are
post-processing. Rounding must **sample** (Birkhoff–von Neumann), not Hungarian: the Sinkhorn scalings
are dual potentials that cancel in any argmax, so Hungarian would discard exactly what the private
kernel encoded. Because the noise scale n·β is **n-driven**, this mechanism is expected to trail — the
unavoidable cost of releasing the whole kernel. The optional Gaussian flag uses ℓ2 sensitivity √n·β and
scale \(\sqrt{2\ln(1.25/\delta)}\,\sqrt{n}\beta/\varepsilon\) (approximate (εd,δ); see above). `B` must
exceed the cost scale that separates good from bad matchings, or the clip erases the signal before any
noise is added.

### Instance generation — `block_alpha` (difficulty with known ground truth)
K = n/2 far-apart rectangle gadgets, each with two public points on top and two private on the bottom,
height h, width \(w_b\):
```
for each block b:  δ_b = δ_max · U_b^(1/α),  U_b ~ Uniform(0,1);   w_b = √(h · δ_b)
D = stakes_S · (2·K·h)                        # separation sets RAND/OPT
centers on a grid of spacing D;   per block:
  Q_b = {(c_x ± w_b/2, c_y + h/2)}   (public, top)
  R_b = {(c_x ± w_b/2, c_y − h/2)}   (private, bottom)
```
Math: per block the two **vertical** edges cost 2h (optimal); the two **crossed** edges cost
\(2\sqrt{w_b^2+h^2} = 2h + \delta_b\) to first order (since \(w_b=\sqrt{h\delta_b}\Rightarrow w_b^2/h=\delta_b\)).
With \(D \gg\) everything, every near-optimal matching is **block-diagonal**, so \(\mathrm{OPT} = 2Kh\) and
the excess of any near-optimal matching over OPT is a **subset sum** \(\sum_{b\in S}\delta_b\) — the
near-optimum spectrum is fixed entirely by \(\{\delta_b\}\). Since
\(\#\{b:\delta_b\le x\} = K\Pr[\delta\le x]\) and \(\delta_b = \delta_{\max}U^{1/\alpha}\Rightarrow\Pr[\delta\le x]\propto x^{\alpha}\),
the number of near-ties grows as \(N(x)\propto x^{\alpha}\): **α is the ruggedness exponent, set as an
input** (small α = many near-ties just above OPT = hard; large α = isolated optimum). `stakes_S` sets
RAND/OPT independently, as long as \(D \gg K\delta_{\max}\). The generator stores \(\{\delta_b\}\) in the
instance params, and `test_block_alpha.py` checks OPT = 2Kh and that the optimality margin equals the
smallest block's crossed excess.

## Pipeline

Two commands. Instances are defined reproducibly by a config (family + params + seed), so
"generation" happens inside the run; the report is a pure function of the recorded results.

```bash
uv sync
uv run pytest -m "not slow"

# 1+2. Generate point sets and run all mechanisms, recording raw results to results/raw/*.parquet
uv run python scripts/run_experiment.py --config configs/plausible.yaml    --allow-dirty
uv run python scripts/run_experiment.py --config configs/block_alpha.yaml  --allow-dirty

# 3. Render this README's table + figures from the latest run
uv run python scripts/update_readme.py
```

Each result row records `mechanism, family, alpha, n, epsilon, epsilon_L, cost, opt, rand,
competitive_ratio, regret_ratio, ...` plus `git_sha` / `config_hash` for provenance. Configs live in
`configs/`; add a mechanism or instance family by writing one file and registering it (see
[PLAN.md](PLAN.md)).

## Experiment design

_Auto-filled from the latest run's config: instance families and their parameters, all
mechanism hyperparameters, and the sweep settings._

<!-- DESIGN:START -->
- **n** = 20 · **trials/config** = 30 · **base seed** = 7 · **length scale** = diam
- **Privacy sweep**: ε ∈ {0.1, 0.5, 1, 2, 5, 10}

**Instance families**

| family | count | params |
| --- | --- | --- |
| `block_alpha` | 30 | alpha=0.5, stakes_S=5.0 |
| `block_alpha` | 30 | alpha=1.0, stakes_S=5.0 |
| `block_alpha` | 30 | alpha=2.0, stakes_S=5.0 |
| `block_alpha` | 30 | alpha=3.0, stakes_S=5.0 |

**Mechanisms & hyperparameters**

| mechanism | hyperparameters |
| --- | --- |
| `local` | — |
| `auction` | m=3, alpha=0.1 |
| `dual_sinkhorn` | beta=5.0, B=2.0, num_iters=50 |
<!-- DESIGN:END -->

## Results

<!-- RESULTS:START -->
_Auto-generated from `results/raw/0681928e4c35_305fb4d4.parquet` (git `305fb4d4`, 2026-07-03 16:19 UTC)_

**64800 trials** across 3 mechanisms, 1 family.

Regret ratio: 0 = optimal, 1 = as bad as a random matching.

| mechanism | family | alpha | median_competitive | median_regret | regret@ε=0.5 | regret@ε=1 | regret@ε=2 | regret@ε=5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| auction | block_alpha | 0.5 | 1.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| auction | block_alpha | 1 | 1.028 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| auction | block_alpha | 2 | 1.035 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| auction | block_alpha | 3 | 1.039 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_sinkhorn | block_alpha | 0.5 | 166.103 | 1.005 | 1.011 | 1.007 | 0.998 | 1.004 |
| dual_sinkhorn | block_alpha | 1 | 165.759 | 1.003 | 1.000 | 1.011 | 1.003 | 1.006 |
| dual_sinkhorn | block_alpha | 2 | 165.947 | 1.004 | 1.005 | 1.009 | 0.998 | 1.004 |
| dual_sinkhorn | block_alpha | 3 | 165.787 | 1.003 | 1.007 | 0.999 | 1.008 | 1.003 |
| local | block_alpha | 0.5 | 1.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| local | block_alpha | 1 | 1.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| local | block_alpha | 2 | 1.022 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| local | block_alpha | 3 | 1.024 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
<!-- RESULTS:END -->

## Figures

<!-- FIGURES:START -->
- [block_alpha: regret vs privacy](results/figures/regret_block_alpha.png)
- [block_alpha: competitive ratio](results/figures/utility_block_alpha.png)
<!-- FIGURES:END -->
