# PLAN.md — Private Bipartite Matching: Experiment Codebase

> **Audience:** a coding agent (and human collaborators) implementing this project from scratch.
> **Goal:** a Python codebase that empirically measures the **privacy–utility tradeoff** of several
> metric-differentially-private bipartite matching mechanisms, is **easy to extend** with new
> mechanisms/instance families, is **reproducible**, and lets agents **auto-update the README**
> with results once data exists.
>
> Read this whole document before writing code. Implement in **phases** (see [§11](#11-milestones--build-order)).
> Use **Python ≥ 3.11** and manage dependencies with **uv**. **Gurobi is NOT required** — `scipy`'s
> Hungarian solver is sufficient. (Gurobi is available to the maintainer if a future exact-ILP idea needs it;
> keep any such use behind an optional extra so the core installs without it.)

> **Scope note (current).** The shipped pipeline is deliberately lean: generate plausible
> point sets (plus the difficulty-graded `block_alpha` family), run the three mechanisms over an
> ε sweep, record raw results, and render the README table + figures. The headline output is
> **regret vs ε, faceted by difficulty (`alpha`)** — see `configs/plausible.yaml`,
> `configs/block_alpha.yaml`, and `scripts/update_readme.py`. The μ-scatter, worst-case
> envelope, and adversarial-search deliverables described below (§1.2–1.3, §3-Phase-4) are **not**
> part of the default report; the code remains for anyone who wants it, but the two-command
> flow in the README is the supported path.

---

## 1. What we are building

A research harness that, for each **mechanism** and each **problem instance**, sweeps the privacy
budget $\varepsilon$ and records how far the private matching's cost lands above optimal. Three
mechanisms ship in v1 (local perturbation, private auction, private dual/Sinkhorn). New mechanisms
and new instance families must be addable by writing one file and registering it — no changes to the
runner.

The headline deliverables are:

1. **Per-configuration utility curves:** competitive ratio vs $\varepsilon$ (poster-style), for each
   instance family, with variance bands + upper quantile.
2. **$\mu$-conditioned scatter:** regret ratio (at fixed dimensionless $\varepsilon\!\cdot\!L$) vs the
   normalized optimality gap $\mu$, aggregated over many instances. This is the transferable result:
   performance as a function of the geometric quantity the theory says controls it, rather than as a
   function of an arbitrarily chosen sampling distribution.
3. **Worst-case envelope:** curves for adversarially-searched hard instances (small $\mu$), plotted
   against the sampled-distribution averages.

---

## 2. Problem setup & definitions (also the source of truth for README §Problem Setup)

Copy this section, lightly reworded, into the README. Keep the math identical.

### 2.1 Instances

- Public set $Q = \{q_i\}_{i=1}^n \subset \mathbb{R}^2$ (e.g. customers). **Public / not protected.**
- Private set $R = \{r_j\}_{j=1}^n \subset \mathbb{R}^2$ (e.g. driver home locations). **Protected.**
- Balanced: $|Q| = |R| = n$, so a perfect matching always exists.
- Cost matrix $c_{ij} = \lVert q_i - r_j \rVert_2$.
- An **assignment** is a bijection $\sigma:[n]\to[n]$ (`sigma[i] = j` means public point $i$ is matched
  to private point $j$). Cost $u(\sigma \mid R) = \sum_{i=1}^n c_{i,\sigma(i)}$.
- $\mathrm{OPT}(R) = \min_\sigma u(\sigma \mid R)$, achieved by $\sigma^\*$ (via Hungarian).

**Note (detour objective reduces to this).** The crowdshipping pay
$\hat u(\sigma)=\sum_i\big(\rho(\lVert q_i\rVert+\lVert q_i-r_{\sigma(i)}\rVert-\lVert r_{\sigma(i)}\rVert)+\gamma\big)$
has the same argmin as $\sum_i \lVert q_i - r_{\sigma(i)}\rVert$ when balanced, because
$\rho\lVert q_i\rVert$, $\gamma$, and $-\rho\sum_j\lVert r_j\rVert$ are permutation-invariant constants.
So we optimize and report the pure Euclidean cost. (Provide a `detour_cost()` helper anyway for reporting.)

### 2.2 Metric privacy

Product metric on private configurations: $d(R,R') = \sum_{j=1}^n \lVert r_j - r'_j\rVert_2$.

**Definition (metric / $\varepsilon d$-privacy).** A randomized mechanism $\mathcal{M}$ is
$\varepsilon d$-private if for all $R, R'$ and all measurable $S$:
$$\Pr[\mathcal{M}(R)\in S] \le e^{\varepsilon\, d(R,R')}\,\Pr[\mathcal{M}(R')\in S].$$

**Planar Laplace (Geo-Ind) mechanism.** To privatize one point $r$: draw
$\theta\sim\mathrm{Unif}[0,2\pi]$ and radius $\nu\sim\mathrm{Gamma}(\text{shape}=2,\ \text{rate}=\varepsilon)$,
output $\tilde r = r + \nu(\cos\theta,\sin\theta)$. This is $\varepsilon$-Geo-Ind.

**Composition tools we rely on.** Post-processing preserves privacy; parallel composition over the $n$
independent private coordinates and sequential composition over repeated queries to the same
coordinate combine as in the source paper.

> **Important:** privacy is a *worst-case, proven* property — the experiments do **not** verify it and
> must not claim to. Experiments measure **utility only**. Privacy is checked separately by unit tests
> that empirically sanity-check the mechanisms against the definition ([§9](#9-testing)).

### 2.3 Hardness parameter: the optimality margin $\mu$

> **Notation.** We write the best/second-best cost gap as $\mu$ (mnemonic: **m**argin), **not** $\Delta$.
> In DP, $\Delta$ is reserved for the **sensitivity / Lipschitz constant** of a query; we keep that
> convention (it appears that way in the auction and dual-Sinkhorn sensitivity analyses). $\mu$ is a
> property of the instance geometry; $\Delta$ is a property of a query.

$$\mu(R) = \min_{\sigma \neq \sigma^\*}\big(u(\sigma\mid R) - \mathrm{OPT}(R)\big),$$
the cost gap between the best and **second-best** assignment. Large $\mu$ = well-separated optimum
(easy: noise rarely flips the winner); small $\mu$ = near-ties (hard). Theory for the local mechanism
gives $E[u(\tilde\sigma\mid R)] \le \mathrm{OPT} + O(e^{-\mu\varepsilon/5})$, so $\mu$ is the
geometric knob governing the tradeoff for the perturbation-based methods.

**Units / normalization.** $\mu$ has units of length; $\varepsilon$ has units of 1/length, so
$\mu\varepsilon$ is dimensionless. Report $\mu$ normalized (e.g. $\mu/\mathrm{OPT}$ or $\mu/L$
for a characteristic length $L$) so it is comparable across instances. Metric privacy has an exact
**scale symmetry**: scaling all points by $\lambda$ and setting $\varepsilon\to\varepsilon/\lambda$ yields
an identical problem. The meaningful privacy axis is therefore the dimensionless $\varepsilon\!\cdot\!L$.

### 2.4 Utility metrics (both reported; see roles below)

- Random-matching baseline (closed form, no sampling): $\mathrm{RAND}(R) = \frac1n\sum_{i,j} c_{ij}$
  (= expected cost of a uniformly random bijection).
- **Competitive ratio** $= E[u(\tilde\sigma)] / \mathrm{OPT} \ \ge 1$. Operational reading: "×OPT cost."
  **Role:** y-axis of per-configuration $\varepsilon$-sweep curves. Fragile when OPT is tiny → do not
  average across instances; if aggregated, use the **median**.
- **Regret ratio** $= \dfrac{E[u(\tilde\sigma)] - \mathrm{OPT}}{\mathrm{RAND} - \mathrm{OPT}} \in [0,1]$
  (can exceed 1 if worse than random — keep, don't clip in stored data). **Role:** cross-instance
  aggregation and the $\mu$-scatter y-axis. Comparable across scales and dynamic ranges.
- Hamming distance $d_H(\tilde\sigma, \sigma^\*)$ (fraction of positions differing) — diagnostic.

For a single fixed instance the two ratios are affine reparametrizations of each other; the reason to
keep both is reader intuition + robustness under cross-instance aggregation. **Do not plot both on the
same per-instance curve.**

---

## 3. Mechanisms (v1)

All must **terminate** and return a **valid perfect matching** for any $\varepsilon>0$ and any instance.
See `docs/algorithms.md` for full pseudocode (the agent should write it there from the specs below).

### 3.1 `local` — local perturbation (baseline, has theory)
Perturb every private point with the planar Laplace mechanism, then run Hungarian on
$\tilde c_{ij} = \lVert q_i - \tilde r_j\rVert$. $\varepsilon d$-private by parallel composition +
post-processing. This is the **bar to beat**.

### 3.2 `auction` — private ascending-price auction
Modified Crawford–Knoer. Each private point (driver) gets a hard cap of `m` bids; a bid is an
**exponential mechanism** draw over public points with score $\lVert q_j - r_i\rVert + p_j$
(1-Lipschitz in $r_i$, so sensitivity 1); prices $p_j$ rise by increment $\alpha$ on demand. Budget
split $\varepsilon_1=\varepsilon/(2m)$ per bid, $\varepsilon_2=\varepsilon/2$ for cleanup.
**Cleanup (practical, tractable):** run the `local` mechanism at budget $\varepsilon_2$ on the residual
unmatched drivers/customers (equal in number by construction), then Hungarian. Terminates in $\le nm$
bids + one cleanup. **Flag for implementer:** the "ideal" cleanup is an exponential mechanism over
residual permutations, which is intractable; use the local-perturbation cleanup and note the swap in a
docstring. Verify the exact privacy constants with the privacy unit test rather than trusting the
budget split blindly.

### 3.3 `dual_sinkhorn` — private entropic dual ascent
Privatize a **single** object: a smoothed, clipped kernel
$K_{ij} = \exp(-\beta\,\min(\lVert q_i - r_j\rVert, B))$. Per-column ($=$ per-driver) $\ell_1$
sensitivity is $\le n\beta$, so add Laplace noise of scale $n\beta/\varepsilon$ per entry and clip to
$[0,1]$ → releasing $\tilde K$ is $\varepsilon d$-private. Everything after is post-processing: run `L`
Sinkhorn iterations to get a near-doubly-stochastic $P$, then **Birkhoff–von Neumann randomized
rounding** to a permutation (decompose $P$ into permutations, sample one $\propto$ its weight).
**Do not** finish with Hungarian: by matching-invariance of dual potentials, the Sinkhorn scalings wash
out of any argmax and the method collapses to a noisy baseline — only randomized rounding makes the
dual iterations matter. **Known weakness to surface in results:** noise scale grows like $n\beta$
($n$-driven, not geometry-driven), so this is expected to trail; stress-test by sweeping $n$, not
geometry. Offer optional mitigations behind flags: Gaussian mechanism with $\ell_2$ sensitivity
$\sqrt{n}\beta$ for $(\varepsilon,\delta)$-metric-DP; per-customer (row) contribution clipping.

---

## 4. Repository layout

```
private-matching/
├── README.md                  # problem setup + definitions + AUTO-UPDATED results
├── PLAN.md                    # this file
├── pyproject.toml             # uv-managed; core deps + optional [gurobi] extra
├── uv.lock
├── .gitignore
├── LICENSE
├── configs/                   # declarative experiment specs (YAML)
│   ├── default.yaml
│   ├── sweep_uniform.yaml
│   ├── sweep_gaussian.yaml
│   ├── margin_scatter.yaml
│   └── adversarial.yaml
├── src/private_matching/
│   ├── __init__.py
│   ├── instances.py           # Instance dataclass + generators + registry
│   ├── matching.py            # Hungarian wrapper, planar-Laplace sampler, Birkhoff rounding
│   ├── metrics.py             # OPT, RAND, μ (second-best), competitive/regret ratio, hamming
│   ├── mechanisms/
│   │   ├── __init__.py         # registry: name -> class
│   │   ├── base.py             # Mechanism ABC
│   │   ├── local.py
│   │   ├── auction.py
│   │   └── dual_sinkhorn.py
│   ├── experiment.py          # config-driven runner: sweeps, trials, seeding, writes results
│   ├── adversarial.py         # CMA-ES worst-case search (phase 3)
│   ├── plotting.py            # figures from results tables
│   ├── config.py              # config dataclasses + YAML loader + validation
│   └── cli.py                 # `python -m private_matching ...`
├── scripts/
│   ├── run_experiment.py       # thin wrapper over cli
│   └── update_readme.py        # regenerates README results block + figures
├── tests/
│   ├── test_matching.py
│   ├── test_metrics.py
│   ├── test_mechanisms.py      # validity: perfect matching + termination, all mechanisms
│   ├── test_privacy.py         # empirical privacy sanity checks
│   └── test_invariance.py      # scale-invariance of the tradeoff
├── results/
│   ├── raw/                    # one parquet per run, named <config_hash>_<git_sha>.parquet
│   ├── figures/                # PNG/SVG produced by plotting.py
│   └── manifest.jsonl          # append-only run log for reproducibility
└── docs/
    └── algorithms.md           # full pseudocode for each mechanism
```

---

## 5. Core abstractions (design for extensibility)

Keep these interfaces small and stable. New ideas plug in by implementing an interface and registering.

### 5.1 `Instance` (in `instances.py`)
```python
@dataclass(frozen=True)
class Instance:
    Q: np.ndarray          # (n, 2) public points
    R: np.ndarray          # (n, 2) private points
    family: str            # e.g. "uniform", "two_gaussian", "ring", "lattice"
    params: dict           # generation params, for provenance
    instance_id: str       # stable hash of (family, params, seed)
```

### 5.2 Instance generators
Functions `gen(n, rng, **params) -> Instance`, registered in a dict `GENERATORS: dict[str, Callable]`.
Ship: `uniform` (unit square), `two_gaussian` (two planar Gaussians), and the adversarial hand-built
families `ring` (interleaved points on a regular n-gon / concentric rings → many near-ties → small
$\mu$) and `lattice` (coincident/shifted grids). A decorator `@register_generator("name")` adds new
ones without editing the runner.

### 5.3 `Mechanism` ABC (in `mechanisms/base.py`)
```python
class Mechanism(ABC):
    name: str
    def __init__(self, **hyperparams): ...
    @abstractmethod
    def match(self, inst: Instance, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        """Return sigma: np.ndarray shape (n,), a permutation; sigma[i] = matched private index."""
```
A decorator `@register_mechanism("name")` populates `MECHANISMS: dict[str, type[Mechanism]]`.
`match` must return a valid permutation (asserted in tests). Hyperparameters (`m`, `alpha`, `beta`,
`B`, `L`, mechanism used for cleanup, etc.) come from config and are stored in provenance.

### 5.4 Result record
Each `(mechanism, instance, epsilon, trial)` evaluation appends one row:
```
mechanism, family, instance_id, n, epsilon, epsilon_L (=epsilon*L), trial, seed,
cost, opt, rand, margin, margin_norm, competitive_ratio, regret_ratio, hamming,
git_sha, config_hash, timestamp
```
Store as a tidy (long-format) pandas DataFrame → parquet. All downstream plots read this; nothing
recomputes from raw points. This is what makes results reproducible and re-plottable.

---

## 6. Key algorithms — implementation notes

- **Hungarian:** `scipy.optimize.linear_sum_assignment` on the cost matrix. Wrap it; that's OPT and
  $\sigma^\*$.
- **Planar Laplace sampler:** `nu = rng.gamma(shape=2, scale=1/epsilon); theta = rng.uniform(0, 2*pi)`.
  Vectorize over all `n` points. **Unit-test** that empirical mean radius ≈ $2/\varepsilon$ and that it
  scales correctly (see invariance test).
- **$\mu$ / second-best assignment:** compute $\sigma^\*$; for each matched edge $(i,\sigma^\*(i))$,
  set that entry to $+\infty$, re-solve Hungarian, record cost; the minimum of these $n$ costs is the
  **second-best assignment cost**, and $\mu = \text{second\_best} - \mathrm{OPT}$. Cost: $n$
  Hungarian solves per instance — fine for $n\le 100$. Handle exact ties (general-position failure):
  if $\mu = 0$ or below a tolerance, mark instance `degenerate=True` (these are the hardest cases and
  should be *kept*, not discarded — they anchor the small-$\mu$ end).
- **Characteristic length $L$:** define once and reuse for the dimensionless axis; suggest
  $L = \text{diam}(Q\cup R)$ or mean nearest-neighbor spacing. Store both so plots can choose.
- **Birkhoff rounding (dual_sinkhorn):** repeatedly extract a permutation from the support of $P$
  (perfect matching on positive entries), subtract the minimum along it as its weight, repeat until $P$
  is exhausted (≤ $n^2-2n+2$ permutations); then sample one permutation $\propto$ weight. Renormalize
  weights to sum to 1; guard against numerical residue.

---

## 7. Experiment runner (`experiment.py`)

Driven entirely by a config file. Responsibilities:

1. Build the instance set: either sample `num_instances` from named families at given `n`, or generate
   the fixed adversarial families. Assign each a stable `instance_id`.
2. Precompute per-instance invariants once: `opt`, `sigma_star`, `rand`, `margin`, `L`.
3. For each `mechanism × instance × epsilon × trial`, seed a fresh
   `rng = np.random.default_rng(SeedSequence([base_seed, hash(instance_id), epsilon_idx, trial]))`,
   call `match`, compute `cost` and the metrics, append a row.
4. Write the tidy DataFrame to `results/raw/<config_hash>_<git_sha>.parquet` and append a line to
   `results/manifest.jsonl` with `{config, config_hash, git_sha, package_versions, timestamp, n_rows}`.

**Common random numbers (CRN):** for the adversarial search and any A/B geometry comparison, use a
**fixed list of trial seeds shared across candidate configurations**, so differences reflect geometry,
not noise draws. (Across *different mechanisms* CRN is not clean because they consume randomness
differently — rely on enough trials there instead.) Expose `crn: true/false` and a `trial_seeds` list in
config.

**Parallelism:** embarrassingly parallel over rows; use `concurrent.futures` (process pool). Keep it
optional and deterministic given seeds.

---

## 8. Reproducibility requirements

- Every result row carries `git_sha` and `config_hash`. Refuse to run on a dirty git tree unless
  `--allow-dirty` is passed (then record `git_sha` as `<sha>-dirty`).
- `config_hash` = stable hash of the fully-resolved config (after defaults merged).
- `uv.lock` committed; `manifest.jsonl` records exact package versions per run.
- No global mutable RNG — always thread an explicit `np.random.Generator`.
- Raw points are regenerable from `(family, params, seed)`; don't store point clouds in results, store
  the recipe. (Optionally cache instance sets under `results/instances/` keyed by hash.)
- One command reproduces a figure end to end:
  `uv run python scripts/run_experiment.py --config configs/margin_scatter.yaml && uv run python scripts/update_readme.py`.

---

## 9. Testing (`pytest`, run via `uv run pytest`)

- **Validity (`test_mechanisms.py`):** for every registered mechanism, over a grid of $n$, $\varepsilon$
  (including tiny and huge), and instance families, assert `match` returns a permutation (sorted equals
  `range(n)`), terminates within a time budget, and never errors. This is the "correct no matter how bad
  the cost / how good the privacy" guarantee.
- **Metrics (`test_metrics.py`):** hand-checkable tiny instances (e.g. $n=2,3$) where OPT, second-best,
  $\mu$, and RAND are computable by hand. Verify $\text{competitive}\ge 1$, $\mu\ge 0$,
  regret ratio $=0$ when cost $=$ OPT and $=1$ when cost $=$ RAND.
- **Privacy sanity (`test_privacy.py`):** empirical, not a proof. For the `local` mechanism on a fixed
  output event $S$ and two nearby inputs $R, R'$, estimate the likelihood ratio over many samples and
  check it stays within $e^{\varepsilon d(R,R')}$ up to Monte-Carlo slack. For `auction` and
  `dual_sinkhorn`, at minimum test that only the private set influences the released private object /
  bid distribution and that the claimed budget split is internally consistent. Mark expensive tests
  `@pytest.mark.slow`.
- **Scale invariance (`test_invariance.py`):** scaling all points by $\lambda$ and using
  $\varepsilon\to\varepsilon/\lambda$ must leave competitive ratio and regret ratio unchanged in
  distribution (compare with CRN + a statistical tolerance). This validates the dimensionless
  $\varepsilon\!\cdot\!L$ axis.
- **Linting:** `ruff` clean.

---

## 10. Plotting (`plotting.py`) & the auto-updated README

All plots read the tidy parquet, never raw points. Functions each take a DataFrame + output path.

1. `plot_utility_vs_epsilon(df, family)` — **competitive ratio** vs $\varepsilon$ (or $\varepsilon L$),
   one line per mechanism + Random and Non-Private OPT reference lines; median line with IQR band and a
   dashed upper-quantile (e.g. p90 or max) to show worst-case-within-support.
2. `plot_regret_vs_margin(df, epsilonL)` — scatter of **regret ratio** at a fixed $\varepsilon L$ vs
   normalized $\mu$, colored by mechanism; this is the transferable result. Overlay a smoothed trend.
3. `plot_worst_case_envelope(df)` — adversarial (small-$\mu$) instance curves vs sampled-distribution
   averages vs random baseline.
4. Optional: `plot_regret_vs_n(df)` for `dual_sinkhorn`'s $n$-driven degradation.

### README auto-update workflow
- README has fenced regions bounded by HTML comment markers, e.g.
  `<!-- RESULTS:START -->` … `<!-- RESULTS:END -->` and `<!-- FIGURES:START -->` … `<!-- FIGURES:END -->`.
- `scripts/update_readme.py`:
  1. loads the latest run(s) from `manifest.jsonl` + parquet,
  2. regenerates figures into `results/figures/`,
  3. builds a summary table (per mechanism × family: median competitive ratio and regret ratio at a
     couple of reference $\varepsilon L$ values, plus the $\varepsilon L$ needed for 5% regret),
  4. rewrites **only** the text between the markers, embedding the table and figure links, and stamps
     the source `git_sha` / timestamp.
- **Agents must never hand-edit inside the markers**; they run the script. Everything outside the
  markers (problem setup, definitions, usage) is human/agent-authored prose.

The README's Problem Setup + Definitions section is [§2](#2-problem-setup--definitions-also-the-source-of-truth-for-readme-problem-setup) of this plan, reworded.

---

## 11. Milestones / build order

Implement and land each phase with tests before moving on.

- **Phase 0 — scaffold.** `uv init`; add deps (below); repo layout; `config.py` loader; empty registries;
  CI-style `uv run pytest` green on placeholders; README skeleton with markers + §2 prose.
- **Phase 1 — core + baseline.** `matching.py` (Hungarian, planar Laplace), `metrics.py` (OPT, RAND,
  $\mu$, ratios, hamming), `instances.py` (`uniform`, `two_gaussian`), `local` mechanism,
  `experiment.py`, `plotting.plot_utility_vs_epsilon`. Reproduce a poster-style curve. Tests: validity,
  metrics, invariance.
- **Phase 2 — central mechanisms.** `auction` and `dual_sinkhorn` (+ Birkhoff rounding). Validity +
  privacy-sanity tests. Compare all three on `uniform`/`two_gaussian`.
- **Phase 3 — the $\mu$ story.** `ring`/`lattice` generators; batch over many instances; second-best
  $\mu$; `plot_regret_vs_margin`. This is the scientific core.
- **Phase 4 — adversarial worst case.** `adversarial.py`: CMA-ES (dep `cma`) maximizing estimated regret
  at a target $\varepsilon L$ over point coordinates, projecting candidates back into the unit disk each
  step, using CRN across candidates; `plot_worst_case_envelope`.
- **Phase 5 — polish.** `update_readme.py` wired end to end; docs/algorithms.md; parallel runner;
  optional `dual_sinkhorn` mitigations (Gaussian mechanism, per-row clipping) and `plot_regret_vs_n`.

---

## 12. Environment & commands (uv)

```bash
uv init
uv add numpy scipy matplotlib pandas pyarrow pyyaml
uv add cma                      # Phase 4 (adversarial search)
uv add --dev pytest ruff
# Optional, only if a future exact-ILP idea needs it (keep behind an extra, not a core dep):
# uv add --optional gurobi gurobipy

uv run pytest
uv run pytest -m "not slow"     # skip expensive privacy tests
uv run python scripts/run_experiment.py --config configs/sweep_uniform.yaml
uv run python scripts/run_experiment.py --config configs/margin_scatter.yaml
uv run python scripts/update_readme.py
```

Pin `requires-python = ">=3.11"` in `pyproject.toml`. Keep the core install Gurobi-free.

---

## 13. Design principles (keep these in mind while coding)

1. **Registries over conditionals.** Adding a mechanism/instance/plot = one new file + one decorator,
   never an edit to the runner. This is what makes it collaborative and extensible.
2. **Tidy data is the contract.** Everything downstream consumes the long-format results table; nothing
   re-derives from raw points. Plots and README are pure functions of that table.
3. **Utility is measured; privacy is proven.** Never let an experiment imply it "verifies" privacy.
4. **Report distribution, not just mean.** Bands + upper quantiles everywhere; the worst case matters
   more than the average here.
5. **Index by $\mu$, not by sampling distribution.** The distribution is a means of generating a
   range of $\mu$; the science is performance-vs-$\mu$.
6. **Reproducible by construction.** Explicit RNGs, config hashes, git SHAs, locked deps, regenerable
   instances.

---

## 14. Open questions to leave as `TODO`/issues (not blockers)

- Exact privacy constant for the `auction` exponential-mechanism bids — the $\varepsilon/2\Delta$ vs
  $\varepsilon/\Delta$ factor, where $\Delta$ here is the **query sensitivity** (Lipschitz constant,
  $=1$ for the bid score), *not* the optimality margin $\mu$. Settle via the privacy unit test and a
  short derivation in `docs/algorithms.md`.
- Whether the local mechanism's $E[u]\le \mathrm{OPT}+2\,\mathrm{diam}(R)\,E[d_H]$ bound can be turned
  into a worst-case-over-configurations statement once diameter and min-spacing are fixed (a proof,
  parallel to the empirical envelope). Track separately from code.
- Better rounding for `dual_sinkhorn` than vanilla Birkhoff (e.g. sampling proportional to entropy-
  regularized coupling directly).
