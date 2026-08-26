# Spatial block cross-validation of smoothing bandwidth

**Question.** Is one global Gaussian bandwidth (`SIGMA=8`, ~100 km) defensible, or
must `sigma` be fitted per question? A referee argued that signal sharpness varies
enormously across questions (`yinz` is a Pittsburgh point feature; `you guys` a
national gradient), so one bandwidth over-smooths the sharp, diagnostic features.

**Answer.** Keep the global bandwidth. Block-CV of the recovered dot maps shows a
per-question gain over the best *global* sigma of **0.005–0.08 %** (max across five
variants 0.077 %), the per-question optima do not spread across `[3,16]` (they pile
at the top of any grid), and the CV-optimal sigma has no stable correlation with a
measurable sharpness statistic. The referee's objection is real in principle — the
sharpness genuinely varies, and the pre-registered sharp/diffuse predictions hold at
the per-choice level — but it is negligible in practice for this model.

## Design (`model/bandwidth.py`)

The coverage rasters are the data. For each question the choices' coverage rasters
(`Surfaces.cov`, one row per (question, choice), fractional dot coverage in a
200×456 plate-carrée grid, 1 cell ≈ 12.7 km) are the multinomial "counts" we try to
predict.

- **Folds.** A checkerboard of square blocks; two complementary folds cover every
  land cell exactly once as held-out. Contiguous blocks (not random cells) are
  essential: dialect surfaces are strongly autocorrelated, so cell-wise holdout
  leaks — a cell's own neighbours reconstruct it.
- **Observation** in a held-out cell = the raw coverage vector across the question's
  choices, treated as multinomial counts and weighted by the cell's total coverage
  (`tot = sum_choices cov`). Weighting by total coverage (rather than rescaling each
  cell to sum 1) matches how the external tuning in `tune.py` weights responses and
  keeps low-coverage ocean/edge cells from dominating.
- **Prediction.** The full deployed pipeline — `density (box=9 → saturation
  inversion → Gaussian sigma) → gamma=1.5 contrast → per-state rake → alpha=0.02
  floor` — is run on the coverage raster with the held-out block **zeroed**, and
  `P(answer | q, cell)` is read in the held-out cells. Score = coverage-weighted
  multinomial log-loss (nats per response).
- **Sweep.** `sigma ∈ {2,3,4,5,6,8,10,12,16,20,24,32}`. The grid was extended past
  16 because the optimum otherwise runs off the top of the grid (see below), which
  would make `best_sigma` degenerate at the censoring value.
- **Vectorised.** All choices of a question are smoothed in one call with per-axis
  sigma `(0, s, s)`; the mask denominators (below) depend only on (fold, sigma), not
  on the question, so they are precomputed once.

## The leakage hazard and the two precautions

Zeroing a block and then Gaussian-smoothing pulls mass in from the block's edges, so
cells just inside the boundary are predicted almost entirely by their immediate
neighbours — which favours small sigma spuriously. Two independent guards, both
implemented:

- **(a) Interior erosion / distance banding.** Score only held-out cells whose
  distance to the nearest training cell lies in a band `[dlo, dhi]`, with `dlo = 4`
  (kills the trivial ≤3-cell neighbour leak) and `dhi = block/2 − 2`. This is the
  conservative guard and the one to believe if the two ever disagree.
- **(b) Normalised convolution.** Smooth `cov · valid` and divide by
  `gaussian(valid)`, and likewise mask-renormalise the box stage, so held-out cells
  read as *missing*, not as zero density. `valid = 1` everywhere except the held-out
  block; ocean keeps `valid = 1` with true-zero coverage so the deployed coastal
  decay is preserved and only the artificial gap is renormalised.
- A per-sigma **truncate widening** `trunc = max(3, (reach+3)/sigma)` lets even
  `sigma=2` reach across the gap; without it, small-sigma predictions collapse to the
  prior in the block interior and produce a spurious W-shaped curve. At `sigma ≥ 8`
  over these gaps it equals the deployed `truncate=3`.

**Do they agree?** Yes. The renormalised (primary) and naive (guard-(a)-only)
variants give the same verdict — both prefer heavy smoothing and both put the
per-question gain under 0.1 % — so the machinery is not producing the answer.

## The crux caveat: block-CV scores *extrapolation*, not *estimation bandwidth*

Holding out a contiguous block asks the model to predict across a spatial **void**.
Across a void the smoothest regional estimate always wins, so block-CV is
structurally biased toward large sigma — its optimum sits at the top of whatever grid
it is given (here 32 ≈ 400 km, which would erase the sub-regional structure the model
exists to represent). Classic KDE theory (`h* ∝ curvature^(−1/5)`) concerns
estimation *where data is present*, not gap extrapolation, and a sharp point feature
that is held out wholesale cannot be recovered by *any* sigma. So block-CV cannot be
read as "deploy sigma=32", and it cannot by itself set an estimation bandwidth.

What it *can* answer legitimately is the **comparative** question the referee raised:
do different questions need materially different sigma? That comparison is fair — the
same bias applies to every question — and the answer is a clean no.

**Raking.** Rake targets are external per-state published percentages, not derived
from held-out cells, so they do not leak held-out truth. But they *do* hand the model
the block's state-level answer, which makes the raked run a test of *within-state
shape* extrapolation. The `norake` variant removes that anchor; both are reported.

## Results (122 questions; five variants)

| variant (block, guard, rake) | global-best σ | deployed-8 loss | per-q-best | **per-q vs global-best** | best-σ median [IQR] |
|---|---|---|---|---|---|
| block25, renorm, **rake** (primary) | 32 | 1.4533 | 1.4021 | **0.0004 nats = 0.029 %** | 32 [24,32] |
| block14, renorm, rake | 32 | 1.4537 | 1.4130 | 0.0009 nats = 0.065 % | 32 [24,32] |
| block40, renorm, rake | 32 | 1.4995 | 1.4113 | 0.0001 nats = 0.005 % | 32 [32,32] |
| block25, **naive**, rake | 32 | 1.4379 | 1.4009 | 0.0011 nats = 0.077 % | 32 [24,32] |
| block25, renorm, **norake** | 32 | 1.3972 | 1.3970 | 0.0002 nats = 0.015 % | 32 [32,32] |

1. **Per-question CV-optimal sigma** (`data/model/bandwidth.csv`): degenerate at the
   grid ceiling — median 32, and 69–95 % of questions pinned at the top — because of
   the extrapolation bias above. Not an estimation optimum.
2. **Spread.** Not the `[3,16]` spread the per-question hypothesis predicts; it is a
   spike at `σ ≥ 24` with deployed-8 sitting to the *left* of the entire distribution
   (`data/model/bandwidth.png`, left panel). Every per-question loss curve flattens
   into a basin from `σ ≈ 8` onward (right panel).
3. **Correlation with sharpness.** Weak and sign-unstable across variants
   (`norake`: Spearman(best_σ, TV_weighted) = −0.28, p=0.001, the expected
   sharper→smaller-σ sign; but primary `rake`: +0.15; `block14`: ≈0). No robust,
   sign-stable correlation → block-CV gives no cheap sharpness-based rule for sigma.
4. **The deciding number.** Per-question vs the best *global* sigma buys **0.005–
   0.08 %** (max 0.077 %, i.e. ≥ 13× below the 1 % threshold). Deployed-8 is 2.6–6.3 %
   above the CV's preferred σ=32, but that gap is the extrapolation artifact, not a
   per-question effect, and σ=32 is undeployable.
5. **Pre-registered sanity check (per-choice sharpness).** Predictions logged before
   fitting held cleanly. Autocorrelation length (ACL, cells) / total variation (TV):
   `yinz` 11/0.27 (sharpest), `yous` 11/0.25, `bubbler` 11/0.22, `grinder` 12/0.21,
   `hoagie` 16/0.22 — all **sharp**, as predicted; `you guys` 22/0.15, `soda` 21/0.15,
   `water fountain` 36/0.13, `sub` 40/0.13, `coke` 40/0.17, `y'all` 40/0.16 — all
   **diffuse**, as predicted. One partial miss: `the City = NYC` came out moderately
   diffuse (ACL 26) because that choice is chosen broadly across the Northeast
   (38 % national), not as a point feature. The sharpness is therefore real and
   measurable — but it lives in the *maps*, and block-CV does not convert it into a
   per-question sigma preference.

## Recommendation

**Keep the single global `SIGMA=8`.** Block cross-validation of the recovered maps
finds no material cross-question spread in optimal bandwidth (per-question tuning buys
< 0.1 %) and no stable link between CV-optimal sigma and question sharpness; the
referee's objection is correct in principle but negligible in practice, and 8 sits at
the left shoulder of a flat CV basin while still preserving the sub-regional detail
that the CV's own (extrapolation-biased) optimum of ~32 would erase.
