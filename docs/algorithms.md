# Algorithms

Pseudocode for each v1 mechanism. Privacy constants should be verified via unit tests, not assumed.

## Hungarian (OPT)

```
cost[i,j] = ||Q[i] - R[j]||
(sigma, opt) = argmin_assignment sum_i cost[i, sigma[i]]   # scipy linear_sum_assignment
```

## Planar Laplace (Geo-Ind)

For each private point `r`:
```
theta ~ Uniform(0, 2*pi)
nu ~ Gamma(shape=2, rate=epsilon)
r_tilde = r + nu * (cos(theta), sin(theta))
```

Parallel composition over `n` independent coordinates gives `epsilon * d`-privacy for the released point set.

## Local mechanism

```
R_tilde = PlanarLaplace(R, epsilon)
sigma = Hungarian(Q, R_tilde)
```

Post-processing of `epsilon d`-private `R_tilde` preserves privacy.

## Auction mechanism

Parameters: bid cap `m`, price increment `alpha`, sensitivity `Delta=1`.

Budget split: `epsilon_1 = epsilon / (2m)` per bid, `epsilon_2 = epsilon / 2` for cleanup.

```
prices[q] = 0 for each public item q
bids_used[r] = 0 for each private bidder r
while some bidder r is unassigned and has bids_used[r] < m:
    scores[q] = ||Q[q] - R[r]|| + prices[q]        # over ALL items q
    q* ~ ExponentialMechanism(scores, epsilon_1, Delta)
    bids_used[r] += 1
    if q* is held by another bidder r': unassign r'   # eviction
    assign r -> q*;  prices[q*] += alpha              # contested items get pricier

residual = unmatched (items, bidders)                 # equal counts by construction
sigma_residual = LocalMechanism(residual, epsilon_2)
merge into full sigma
```

Bidders are the private points (drivers); items are the public points (customers). A bidder bids only while unassigned, wins by eviction, and is capped at `m` bids, so the loop runs in at most `n*m` bids.

**Cleanup note:** the ideal cleanup is an exponential mechanism over residual permutations (intractable). We use local perturbation instead; verify privacy constants empirically.

## Dual Sinkhorn mechanism

Parameters: `beta`, clip bound `B`, `L` Sinkhorn iterations.

```
K[i,j] = exp(-beta * min(||Q[i] - R[j]||, B))
sensitivity per column (driver j): <= n * beta  (L1)
noise ~ Laplace(0, n*beta/epsilon) added to each entry
K_tilde = clip(K + noise, 0, 1)

P = Sinkhorn(K_tilde, L iterations)
sigma ~ BirkhoffVonNeumannRound(P)   # NOT Hungarian
```

Optional mitigations (flags):
- Gaussian noise with `ell_2` sensitivity `sqrt(n)*beta` for `(epsilon, delta)`-DP:
  `scale = sqrt(2 ln(1.25/delta)) * sqrt(n)*beta / epsilon`
- Per-customer (row) contribution clipping before column release

## Margin mu (second-best)

```
(opt, sigma*) = Hungarian(Q, R)
for each i in 1..n:
    cost' = cost with cost[i, sigma*[i]] = infinity
    opt_i = Hungarian(cost')
second_best = min_i opt_i
mu = second_best - opt
```

## Birkhoff–von Neumann rounding

```
perms = [], weights = []
while P has positive entries:
    sigma = max_weight_perfect_matching(P)
    w = min_i P[i, sigma[i]]
    append (sigma, w); subtract w along sigma
sample sigma with probability proportional to weight
```
