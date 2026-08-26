"""Spatial block cross-validation of the recovered dot maps, to decide whether a
single global smoothing bandwidth sigma is defensible or must be fitted per
question.

WHY THIS EXISTS
---------------
sigma=8 was fitted against ONE external dataset (Pop vs. Soda, HDS q105) plus a
hand-curated isogloss test. A referee objects: signal sharpness varies across
questions ("yinz" is a Pittsburgh point feature, "you guys" a smooth national
gradient), and optimal KDE bandwidth scales inversely with feature curvature, so
one global sigma necessarily over-smooths the sharp features and under-smooths
the diffuse ones. We need a per-question criterion that uses NO external data
and is not circular.

THE TEST
--------
The coverage rasters ARE the data. Hold out a spatially contiguous checkerboard
of blocks, fit the density surfaces from the remaining cells only, and score how
well each held-out cell's observed dot composition is predicted by the full
density -> gamma -> rake -> alpha pipeline. Because dialect surfaces are
spatially autocorrelated, random cell-wise holdout would leak (a cell's
neighbours reconstruct it); contiguous blocks larger than the bandwidths tested
force genuine extrapolation across a gap.

Observation in a held-out cell x: the vector of coverage values across the
question's choices, cov_a(x). Treated as multinomial counts (counts_a = cov_a),
so the per-cell log-loss is weighted by total coverage w(x)=sum_a cov_a(x). This
is identical to "weight the cross-entropy by total coverage in the cell"; it is
the natural multinomial reading and matches how tune.py scores Pop vs. Soda.

LEAKAGE PRECAUTIONS (the crux)
------------------------------
Zeroing a block and Gaussian-smoothing pulls mass in from the block edges, so a
cell just inside the boundary is predicted almost entirely by its immediate
(training) neighbour and trivially favours small sigma. Two independent guards,
both implemented here:

  (b) MASK RENORMALISATION (normalized convolution). Smooth cov*valid and divide
      by the Gaussian of the valid mask, so held-out cells read as MISSING, not
      as zero density. Held-out cells are then predicted by a properly
      renormalised Gaussian average of the surrounding training cells. This is
      the primary prediction method (renorm=True).

  (a) INTERIOR EROSION. Score only cells eroded from the block boundary by a
      margin, so the nearest training cell is at least `margin` away and no
      score comes from an immediately-adjacent neighbour. We sweep the margin.

We ALSO run the naive "zero and smooth, no renormalisation" variant to exhibit
the leakage (it should favour smaller sigma). If (a) and (b) disagree, (a) is
the more conservative and is believed.

Block size sets the spatial scale the model must extrapolate across and shifts
the absolute optimum, so we sweep block size and report sensitivity. The
scientifically meaningful quantity is not the absolute optimum but the SPREAD of
optimal sigma ACROSS QUESTIONS at fixed block size, and whether it tracks a
sharpness statistic.

RAKING. Raking targets are per-state published percentages computed from raw
records, not from the held-out cells, so they do not leak the held-out truth.
But they DO supply the held-out block's state-level answer, so with rake=True
this is a test of WITHIN-STATE SHAPE extrapolation. A no-rake variant is also
reported.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    distance_transform_edt, gaussian_filter, uniform_filter,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_util import state_raster  # noqa: E402
from likelihood import (  # noqa: E402
    MAX_FRACTION, BOX, Surfaces, national_pct, published_state_pct, rake,
    state_sample_sizes,
)
from tensor import SIGMA, GAMMA, ALPHA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

SIGMAS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]  # task-required minimum
# extended upward because block-CV of these already-smooth surfaces has its
# optimum above 16 for most questions (it scores gap extrapolation); bracketing
# the optimum is necessary to measure any real cross-question spread.
GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]
DEPLOYED = 8.0
EPS = 1e-6


def trunc_for(sigma, reach):
    """Gaussian truncate so the kernel reaches at least `reach` cells.

    The deployed pipeline uses truncate=3. For the CV we must let even the
    smallest sigma reach across the held-out gap, otherwise a deep cell reads as
    zero density and collapses to the global prior -- an artefact that spuriously
    rewards small sigma. So we widen the kernel just enough to span the deepest
    scored cell (reach), never below the deployed 3.0. At sigma=8 over a <=21
    cell gap this is 3.0, i.e. identical to deployment.
    """
    return max(3.0, (reach + 3.0) / sigma)


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def checkerboard(shape, block, offset=0):
    """Two complementary checkerboard hold-out masks of square blocks.

    Returns a list of boolean (H, W) arrays; True means the cell is HELD OUT.
    Every cell is held out in exactly one of the two folds.
    """
    h, w = shape
    yi = (np.arange(h) + offset) // block
    xi = (np.arange(w) + offset) // block
    parity = (yi[:, None] + xi[None, :]) % 2
    return [parity == 0, parity == 1]


# --------------------------------------------------------------------------- #
# masked density  (vectorised over a question's choices)
# --------------------------------------------------------------------------- #
def masked_density(cov_q, valid, sigma, box, renorm, truncate,
                   gden_box=None, gden_g=None):
    """Saturation-corrected, pooled density with a held-out block masked out.

    cov_q : (C, H, W) raw coverage for one question's choices.
    valid : (H, W) float, 1 where a cell is training data, 0 where held out.
    renorm: if True use normalized convolution (guard b); if False the naive
            zero-and-smooth pipeline (leakage-prone baseline).
    truncate: Gaussian truncation; widened by trunc_for so small sigma reaches
            across the gap instead of collapsing to zero.
    gden_box, gden_g : optional precomputed uniform/Gaussian filters of `valid`
            (they depend only on the fold and sigma, not the question).

    Returns d of shape (C, H, W).
    """
    covm = cov_q * valid[None]
    if renorm:
        num = uniform_filter(covm, (0, box, box), mode="constant")
        den = gden_box
        if den is None:
            den = uniform_filter(valid, box, mode="constant")
        f = np.where(den > EPS, num / np.maximum(den, EPS), 0.0)
    else:
        f = uniform_filter(covm, (0, box, box), mode="constant")
    d = -np.log1p(-np.clip(f, 0.0, MAX_FRACTION))

    if renorm:
        gnum = gaussian_filter(d * valid[None], (0, sigma, sigma),
                               mode="constant", truncate=truncate)
        gden = gden_g
        if gden is None:
            gden = gaussian_filter(valid, sigma, mode="constant",
                                   truncate=truncate)
        d = np.where(gden[None] > EPS, gnum / np.maximum(gden, EPS)[None], 0.0)
    else:
        d = gaussian_filter(d, (0, sigma, sigma), mode="constant",
                            truncate=truncate)
    return d.astype(np.float32)


def masked_build(cov_q, choices, valid, sigma, box, gamma, alpha, renorm,
                 truncate, states=None, table_q=None, sizes=None,
                 national_q=None, m=0.0, gden_box=None, gden_g=None):
    """Full pipeline on masked coverage: density -> gamma -> rake -> alpha.

    Returns P of shape (C, H, W), a proper distribution over choices at every
    cell, matching build() in likelihood.py.
    """
    d = masked_density(cov_q, valid, sigma, box, renorm, truncate,
                       gden_box=gden_box, gden_g=gden_g)
    if gamma != 1.0:
        d = d ** gamma
    if states is not None and table_q is not None:
        d = rake(d, choices, states, table_q, sizes=sizes, national=national_q, m=m)
    share = d.sum(axis=(1, 2))
    share = share / max(share.sum(), 1e-12)
    scale = float(d.sum(axis=0).mean())
    prior = alpha * scale * share[:, None, None]
    num = d + prior
    p = num / np.maximum(num.sum(axis=0, keepdims=True), 1e-12)
    return p.astype(np.float32)


# --------------------------------------------------------------------------- #
# cross-validation
# --------------------------------------------------------------------------- #
class BlockCV:
    def __init__(self, surfaces, block=25, rake_on=True, dlo=4, dhi=None,
                 sigmas=GRID):
        self.s = surfaces
        self.block = block
        self.shape = surfaces.shape
        self.rake_on = rake_on
        self.sigmas = list(sigmas)
        self.states = state_raster() if rake_on else None
        self.table = published_state_pct() if rake_on else None
        self.sizes = state_sample_sizes() if rake_on else None
        self.national = national_pct() if rake_on else None
        self.land = (self.states != "") if rake_on else _land_from(surfaces)
        self.folds = checkerboard(self.shape, block)
        # scored distance band: nearest training >= dlo (kills trivial leakage),
        # <= dhi (the block half-width, the extrapolation gap the model bridges)
        self.dlo = dlo
        self.dhi = dhi if dhi is not None else block // 2 - 2
        self.reach = self.dhi
        # distance-to-training and the scored band, per fold
        self.dist = [distance_transform_edt(held) for held in self.folds]
        # precompute mask denominators shared across all questions
        self._gden_box = {}
        self._gden_g = {}
        self._trunc = {sg: trunc_for(sg, self.reach) for sg in self.sigmas}
        for fi, held in enumerate(self.folds):
            valid = (~held).astype(np.float32)
            self._gden_box[fi] = uniform_filter(valid, BOX, mode="constant")
            for sg in self.sigmas:
                self._gden_g[(fi, sg)] = gaussian_filter(
                    valid, sg, mode="constant", truncate=self._trunc[sg])

    def scored_mask(self, fi):
        held = self.folds[fi]
        d = self.dist[fi]
        return held & self.land & (d >= self.dlo) & (d <= self.dhi)

    def question(self, q, sigmas=None, renorm=True, floor_cov=0.05):
        """Weighted multinomial log-loss curve over sigma for one question.

        Scores held-out cells in the distance band that are land and carry
        coverage. Returns dict sigma -> (ll_sum, w_sum) across both folds.
        """
        sigmas = self.sigmas if sigmas is None else sigmas
        idx = self.s.rows[q]
        choices = list(self.s.choice[idx])
        cov_q = self.s.cov[idx]                       # (C, H, W)
        tot = cov_q.sum(axis=0)                        # observed weight
        obs = cov_q / np.maximum(tot[None], 1e-12)     # observed composition
        table_q = (self.table or {}).get(q, {}) if self.rake_on else None
        nat_q = (self.national or {}).get(q) if self.rake_on else None

        acc = {sg: [0.0, 0.0] for sg in sigmas}
        for fi in range(len(self.folds)):
            valid = (~self.folds[fi]).astype(np.float32)
            scored = self.scored_mask(fi) & (tot > floor_cov)
            if not scored.any():
                continue
            ys, xs = np.nonzero(scored)
            w = tot[ys, xs]
            o = obs[:, ys, xs]                          # (C, n)
            for sg in sigmas:
                p = masked_build(
                    cov_q, choices, valid, sg, BOX, GAMMA, ALPHA, renorm,
                    self._trunc[sg], states=self.states, table_q=table_q,
                    sizes=self.sizes, national_q=nat_q,
                    gden_box=self._gden_box[fi], gden_g=self._gden_g[(fi, sg)])
                pc = np.clip(p[:, ys, xs], 1e-6, 1.0)
                pc = pc / pc.sum(axis=0, keepdims=True)
                ll = -(o * np.log(pc)).sum(axis=0) * w
                acc[sg][0] += float(ll.sum())
                acc[sg][1] += float(w.sum())
        return {sg: (a[0], a[1]) for sg, a in acc.items()}


def _land_from(surfaces):
    z = np.load(DATA / "model" / "state_raster.npy", allow_pickle=True)
    return z != ""


# --------------------------------------------------------------------------- #
# sharpness statistics of the dominant-choice surface
# --------------------------------------------------------------------------- #
def sharpness(surfaces, q, land, ref_sigma=2.0):
    """Sharpness proxies for a question, from the raw coverage at a fixed light
    smoothing (ref_sigma), so every question is measured at the same scale and
    the statistic is not contaminated by the sigma under test. Larger = sharper.

    Reported both for the dominant (modal) choice surface and as a
    choice-share-weighted mean over all choices, because the modal choice is
    often the diffuse national default while the sharp signal lives in the
    minority regional choices.
    """
    idx = surfaces.rows[q]
    cov_q = surfaces.cov[idx]
    share = cov_q.sum(axis=(1, 2))
    share = share / max(share.sum(), 1e-12)
    dom = int(share.argmax())

    def surf(i):
        f = uniform_filter(cov_q[i].astype(np.float32), BOX, mode="constant")
        d = -np.log1p(-np.clip(f, 0.0, MAX_FRACTION))
        return gaussian_filter(d, ref_sigma, mode="constant", truncate=3.0)

    def stats(d):
        mu = float(d[land].mean())
        if mu <= 0:
            return 0.0, 0.0, float("nan")
        gy, gx = np.gradient(d)
        tv = float(np.sqrt(gy ** 2 + gx ** 2)[land].mean() / mu)
        return tv, _moran(d, land), _autocorr_len(d, land)

    tv_d, mor_d, acl_d = stats(surf(dom))
    tv_w = mor_w = acl_w = 0.0
    wsum = 0.0
    for i in range(len(idx)):
        wi = float(share[i])
        if wi < 1e-4:
            continue
        t, mo, ac = stats(surf(i))
        if not np.isfinite(ac):
            ac = 40.0
        tv_w += wi * t
        mor_w += wi * mo
        acl_w += wi * ac
        wsum += wi
    if wsum > 0:
        tv_w /= wsum
        mor_w /= wsum
        acl_w /= wsum
    return {"tv_dom": tv_d, "moran_dom": mor_d, "acl_dom": acl_d,
            "tv_wt": tv_w, "moran_wt": mor_w, "acl_wt": acl_w}


def _moran(d, land, maxlag=None):
    """Lag-1 Moran's I over 4-connected land neighbours."""
    m = land
    z = d - d[m].mean()
    z = np.where(m, z, 0.0)
    num = 0.0
    n_pairs = 0.0
    # horizontal neighbours
    a = z[:, :-1]
    b = z[:, 1:]
    ok = m[:, :-1] & m[:, 1:]
    num += float((a * b * ok).sum())
    n_pairs += float(ok.sum())
    # vertical neighbours
    a = z[:-1, :]
    b = z[1:, :]
    ok = m[:-1, :] & m[1:, :]
    num += float((a * b * ok).sum())
    n_pairs += float(ok.sum())
    denom = float((z[m] ** 2).sum())
    if denom <= 0 or n_pairs <= 0:
        return 0.0
    n = float(m.sum())
    return (n / n_pairs) * (num / denom)


def _autocorr_len(d, land, lags=range(1, 41)):
    """Spatial autocorrelation length: lag (in cells) where the horizontal+
    vertical autocorrelation of the surface first drops below 1/e."""
    m = land
    z = d - d[m].mean()
    z = np.where(m, z, 0.0)
    var = float((z[m] ** 2).mean())
    if var <= 0:
        return np.nan
    thr = 1.0 / np.e
    for k in lags:
        num = 0.0
        cnt = 0.0
        a = z[:, :-k]
        b = z[:, k:]
        ok = m[:, :-k] & m[:, k:]
        num += float((a * b * ok).sum())
        cnt += float(ok.sum())
        a = z[:-k, :]
        b = z[k:, :]
        ok = m[:-k, :] & m[k:, :]
        num += float((a * b * ok).sum())
        cnt += float(ok.sum())
        if cnt <= 0:
            return float(k)
        r = (num / cnt) / var
        if r < thr:
            return float(k)
    return float(max(lags))


def choice_stats(surfaces, land, q, choice, ref_sigma=2.0):
    """Sharpness proxies (tv, moran, acl) for a single (question, choice)
    surface, at the same fixed light smoothing as sharpness(). Used for the
    per-choice sanity check (yinz should be sharp/short-range, you guys diffuse).
    """
    idx = surfaces.rows[q]
    letters = list(surfaces.choice[idx])
    i = idx[letters.index(choice)]
    f = uniform_filter(surfaces.cov[i].astype(np.float32), BOX, mode="constant")
    d = -np.log1p(-np.clip(f, 0.0, MAX_FRACTION))
    d = gaussian_filter(d, ref_sigma, mode="constant", truncate=3.0)
    mu = float(d[land].mean())
    if mu <= 0:
        return {"tv": 0.0, "moran": 0.0, "acl": float("nan"), "natl_share": 0.0}
    gy, gx = np.gradient(d)
    tv = float(np.sqrt(gy ** 2 + gx ** 2)[land].mean() / mu)
    share = surfaces.cov[idx].sum(axis=(1, 2))
    natl = float(share[letters.index(choice)] / max(share.sum(), 1e-12))
    return {"tv": tv, "moran": _moran(d, land), "acl": _autocorr_len(d, land),
            "natl_share": natl}


# --------------------------------------------------------------------------- #
def load_texts():
    out = {}
    with open(DATA / "hds" / "questions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["question"]] = (r["text"], int(r["n_choices"]))
    return out


def parabola_min(sigmas, lls):
    """Sub-grid optimum by fitting a parabola to the 3 points around the min."""
    i = int(np.argmin(lls))
    if i == 0 or i == len(sigmas) - 1:
        return float(sigmas[i])
    x0, x1, x2 = sigmas[i - 1], sigmas[i], sigmas[i + 1]
    y0, y1, y2 = lls[i - 1], lls[i], lls[i + 1]
    d1 = (y2 - y1) / (x2 - x1) - (y1 - y0) / (x1 - x0)
    if abs(d1) < 1e-12:
        return float(x1)
    a = d1 / (x2 - x0)
    b = (y2 - y1) / (x2 - x1) - a * (x2 + x1)
    xv = -b / (2 * a)
    return float(min(max(xv, sigmas[0]), sigmas[-1]))


def spearman(x, y):
    """Spearman rank correlation and a two-sided t-approx p-value."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), n
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    r = float(np.corrcoef(rx, ry)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0, n
    t = r * np.sqrt((n - 2) / (1 - r * r))
    # normal approximation to the two-sided p-value
    import math
    p = 2.0 * 0.5 * (1.0 + math.erf(-abs(t) / math.sqrt(2)))
    return r, float(p), n


# --------------------------------------------------------------------------- #
# full run + analysis
# --------------------------------------------------------------------------- #
def run_full(cv, texts, renorm=True, out_csv=None, tag=""):
    """CV every question, add sharpness, write the per-question CSV, return rows."""
    sigmas = cv.sigmas
    rows = []
    n = len(cv.s.questions)
    for k, q in enumerate(cv.s.questions, 1):
        res = cv.question(q, renorm=renorm)
        lls = np.array([res[sg][0] / max(res[sg][1], 1e-9) for sg in sigmas])
        wtot = float(np.mean([res[sg][1] for sg in sigmas]))
        best = float(sigmas[int(lls.argmin())])
        best_c = parabola_min(sigmas, lls)
        sh = sharpness(cv.s, q, cv.land)
        text, ncho = texts.get(q, ("", len(cv.s.rows[q])))
        row = {"question": q, "text": text, "n_choices": ncho,
               "w_total": wtot, "best_sigma": best, "best_sigma_cont": best_c,
               "ll_deployed8": float(lls[sigmas.index(DEPLOYED)])}
        for sg, ll in zip(sigmas, lls):
            row[f"ll_s{int(sg)}"] = float(ll)
        row.update(sh)
        rows.append(row)
        if k % 20 == 0 or k == n:
            print(f"  [{tag}] {k}/{n}")
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = (["question", "text", "n_choices", "w_total", "best_sigma",
                   "best_sigma_cont", "ll_deployed8"]
                  + [f"ll_s{int(sg)}" for sg in sigmas]
                  + ["tv_dom", "moran_dom", "acl_dom",
                     "tv_wt", "moran_wt", "acl_wt"])
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"  wrote {out_csv}")
    return rows


def global_optimum(rows, sigmas):
    """Coverage-weighted total log-loss at each sigma, and the global-best sigma.

    This is the single sigma that minimises the pooled CV loss over all
    questions -- the fair 'best global' baseline to judge per-question against.
    """
    tot = {sg: 0.0 for sg in sigmas}
    wsum = 0.0
    for r in rows:
        w = r["w_total"]
        wsum += w
        for sg in sigmas:
            tot[sg] += w * r[f"ll_s{int(sg)}"]
    curve = {sg: tot[sg] / wsum for sg in sigmas}
    g = min(curve, key=curve.get)
    return g, curve, wsum


def per_question_gain(rows, sigmas, g):
    """Nats/response saved by per-question sigma vs the global-best sigma g."""
    num_glob = num_pq = wsum = 0.0
    for r in rows:
        w = r["w_total"]
        wsum += w
        num_glob += w * r[f"ll_s{int(g)}"]
        num_pq += w * min(r[f"ll_s{int(sg)}"] for sg in sigmas)
    glob = num_glob / wsum
    pq = num_pq / wsum
    return glob, pq, (glob - pq), (glob - pq) / glob


def analyze(rows, sigmas, tag, deployed=DEPLOYED):
    best = np.array([r["best_sigma"] for r in rows])
    bc = np.array([r["best_sigma_cont"] for r in rows])
    g, curve, _ = global_optimum(rows, sigmas)
    glob, pq, gain, gain_pct = per_question_gain(rows, sigmas, g)
    ll_dep = np.average([r["ll_deployed8"] for r in rows],
                        weights=[r["w_total"] for r in rows])
    print(f"\n===== {tag} =====")
    print(f"best_sigma (grid): median {np.median(best):.1f}  "
          f"IQR [{np.percentile(best,25):.1f},{np.percentile(best,75):.1f}]  "
          f"min {best.min():.0f} max {best.max():.0f}  "
          f"at-max% {(best>=sigmas[-1]).mean()*100:.0f}")
    print(f"best_sigma (parabola): median {np.median(bc):.1f}  "
          f"IQR [{np.percentile(bc,25):.1f},{np.percentile(bc,75):.1f}]")
    print("pooled log-loss vs sigma:")
    print("   " + "  ".join(f"{int(sg)}:{curve[sg]:.4f}" for sg in sigmas))
    print(f"global-best sigma = {g:.0f}  (deployed {deployed:.0f} -> "
          f"{ll_dep:.4f}; delta {ll_dep-curve[g]:+.4f} nats)")
    print(f"per-question vs global-{g:.0f}: global {glob:.4f}  per-q {pq:.4f}  "
          f"gain {gain:.4f} nats/response ({gain_pct*100:.2f}%)")
    for key in ("tv_dom", "acl_dom", "tv_wt", "acl_wt", "moran_wt"):
        vals = [r[key] for r in rows]
        rr, pp, nn = spearman(vals, best)
        print(f"  Spearman(best_sigma, {key:8}) = {rr:+.3f}  p={pp:.3g}  n={nn}")
    return {"best": best, "curve": curve, "global": g, "gain": gain,
            "gain_pct": gain_pct, "ll_dep": ll_dep}


def write_png(rows, sigmas, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = np.array([r["best_sigma"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    edges = np.array(sigmas + [sigmas[-1] * 1.3])
    ax[0].hist(best, bins=edges, color="#4477aa", edgecolor="white")
    ax[0].axvline(DEPLOYED, color="crimson", ls="--", label=f"deployed {DEPLOYED:.0f}")
    ax[0].set_xlabel("CV-optimal sigma (cells; 1 cell ~ 12.7 km)")
    ax[0].set_ylabel("questions")
    ax[0].set_title(f"Per-question optimal sigma (n={len(best)})")
    ax[0].legend()
    _, curve, _ = global_optimum(rows, sigmas)
    for r in rows:
        ax[1].plot(sigmas, [r[f"ll_s{int(s)}"] for s in sigmas],
                   color="0.8", lw=0.5)
    ax[1].plot(sigmas, [curve[s] for s in sigmas], color="#ee6677", lw=2.5,
               label="pooled")
    ax[1].axvline(DEPLOYED, color="crimson", ls="--")
    ax[1].set_xlabel("sigma (cells)")
    ax[1].set_ylabel("CV log-loss (nats/response)")
    ax[1].set_title("Per-question log-loss curves (grey) + pooled (red)")
    ax[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"  wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--dlo", type=int, default=4)
    ap.add_argument("--dhi", type=int, default=0)
    ap.add_argument("--questions", default="")
    ap.add_argument("--no-renorm", action="store_true")
    ap.add_argument("--no-rake", action="store_true")
    ap.add_argument("--full", action="store_true", help="run all questions, write CSV")
    ap.add_argument("--out", default="")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--choice-sanity", action="store_true",
                    help="print per-choice sharpness table (pre-registered check)")
    args = ap.parse_args()

    surfaces = Surfaces()

    if args.choice_sanity:
        # Pre-registered sharp (S) vs diffuse (D) predictions, logged before any
        # fit; sharp => short autocorrelation length (ACL) and high total
        # variation (TV). See data/model/bandwidth.md item 5.
        land = state_raster() != ""
        items = [("50", "f", "yinz/yins", "S"), ("50", "b", "yous/youse", "S"),
                 ("50", "d", "you guys", "D"), ("50", "i", "y'all", "D"),
                 ("103", "a", "bubbler", "S"), ("103", "d", "water fountain", "D"),
                 ("64", "c", "hoagie", "S"), ("64", "b", "grinder", "S"),
                 ("64", "a", "sub", "D"), ("95", "a", "the City=NYC", "S"),
                 ("105", "a", "soda", "D"), ("105", "c", "coke", "D"),
                 ("28", "b", "cot=caught", "D"), ("15", "a", "M/m/m same", "D")]
        print(f"{'q':>4} {'ch':>2} {'gloss':>15} {'pre':>3} "
              f"{'natl%':>6} {'TV':>5} {'ACL':>4} {'Moran':>6}")
        rows = []
        for q, c, g, pr in items:
            st = choice_stats(surfaces, land, q, c)
            rows.append((q, c, g, pr, st))
            print(f"{q:>4} {c:>2} {g:>15} {pr:>3} {st['natl_share']*100:>6.1f} "
                  f"{st['tv']:>5.2f} {st['acl']:>4.0f} {st['moran']:>6.3f}")
        print("\nsorted by ACL (sharpest first):")
        for q, c, g, pr, st in sorted(rows, key=lambda r: r[4]['acl']):
            print(f"  ACL={st['acl']:>4.0f} TV={st['tv']:>5.2f}  "
                  f"q{q}{c} {g} [{pr}]")
        sys.exit(0)

    cv = BlockCV(surfaces, block=args.block, rake_on=not args.no_rake,
                 dlo=args.dlo, dhi=(args.dhi or None))
    renorm = not args.no_renorm
    tag = (f"block{args.block} band[{cv.dlo},{cv.dhi}] "
           f"{'renorm' if renorm else 'naive'} "
           f"{'rake' if not args.no_rake else 'norake'}")
    print(tag)
    print("trunc per sigma: " +
          " ".join(f"{sg:.0f}:{cv._trunc[sg]:.1f}" for sg in cv.sigmas))

    if args.full:
        texts = load_texts()
        out = Path(args.out) if args.out else (DATA / "model" / "bandwidth.csv")
        rows = run_full(cv, texts, renorm=renorm, out_csv=out, tag=tag)
        analyze(rows, cv.sigmas, tag)
        if args.png:
            write_png(rows, cv.sigmas, DATA / "model" / "bandwidth.png")
    else:
        qs = args.questions.split(",") if args.questions else surfaces.questions
        for q in qs:
            res = cv.question(q, renorm=renorm)
            lls = [res[sg][0] / max(res[sg][1], 1e-9) for sg in cv.sigmas]
            best = cv.sigmas[int(np.argmin(lls))]
            print(f"q{q:>4} best={best:>4.0f}  " +
                  " ".join(f"{ll:.4f}" for ll in lls))


