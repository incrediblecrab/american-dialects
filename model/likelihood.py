"""Build P(answer | location) for every Harvard Dialect Survey question.

Two independent sources are fused here.

The dot maps give, per answer, a fractional-coverage surface. They carry
sub-state spatial structure that exists nowhere else, but they are unreliable
about *level* in dense metros: where respondents are packed together every
answer's dots merge, coverage pins near 1 for several choices at once, and the
ratio between them collapses toward equality. New York City reads as 55% soda
when the state as a whole is 82%.

The survey's published per-state percentages were computed from the raw
respondent records, not from the maps, so they carry the correct level but no
geography below the state.

So: take shape from the maps, level from the table. `rake` rescales each
surface until its population-weighted state mean matches the published number,
which fixes the metro compression without touching within-state structure.

Three corrections are applied before that:

  saturation  overlapping dots pin coverage at 1, so -ln(1-f) recovers relative
              density. This must happen at the scale dots actually merge at (a
              few cells); applying it after wide smoothing does nothing, since
              -ln(1-f) is linear in f once f is small.
  smoothing   respondents are a sample, so density is noisy; a Gaussian kernel
              of bandwidth sigma pools neighbouring evidence.
  shrinkage   where few respondents live the local split is unstable, so it is
              pulled toward the national split with weight alpha.

sigma and alpha are fitted in tune.py against an external dataset. Note that
after raking, validating against the published state percentages is circular by
construction; only popvssoda and YGDP remain as honest tests.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

MAX_FRACTION = 0.995
BOX = 9  # dot-merge scale, matches scrape/hds_geo.py
MIN_PUBLISHED = 0.002  # floor so a raked surface never asserts impossibility


class Surfaces:
    """Raw coverage grids plus the question/answer index."""

    def __init__(self):
        z = np.load(DATA / "hds" / "geo" / "grid.npz", allow_pickle=True)
        self.cov = z["grid"].astype(np.float32)
        self.question = np.array([str(q) for q in z["question"]])
        self.choice = np.array([str(c) for c in z["choice"]])
        self.lats = z["lats"]
        self.lons = z["lons"]
        self.questions = sorted(set(self.question), key=int)
        self.rows = {q: np.nonzero(self.question == q)[0] for q in self.questions}

    @property
    def shape(self):
        return self.cov.shape[1:]

    def cell(self, lat, lon):
        return (
            int(np.argmin(np.abs(self.lats - lat))),
            int(np.argmin(np.abs(self.lons - lon))),
        )


def density(cov, sigma, box=BOX):
    """Saturation-corrected density, un-saturated at the dot scale then pooled."""
    f = uniform_filter(cov.astype(np.float32), box, mode="constant")
    d = -np.log1p(-np.clip(f, 0.0, MAX_FRACTION))
    if sigma > 0:
        d = gaussian_filter(d, sigma, mode="constant", truncate=3.0)
    return d


def published_state_pct():
    """{question: {choice: {state: fraction}}} from the survey's own tables."""
    out = {}
    with open(DATA / "hds" / "state_pct.csv") as fh:
        for r in csv.DictReader(fh):
            q = out.setdefault(r["question"], {})
            q.setdefault(r["choice"], {})[r["state"]] = float(r["pct"]) / 100.0
    return out


def national_pct():
    """{question: {choice: fraction}} for the survey as a whole."""
    out = {}
    with open(DATA / "hds" / "answers.csv") as fh:
        for r in csv.DictReader(fh):
            if r["pct_national"]:
                out.setdefault(r["question"], {})[r["choice"]] = \
                    float(r["pct_national"]) / 100.0
    return out


def state_sample_sizes():
    """{state: respondents}, recovered in scrape/hds_counts.py."""
    path = DATA / "hds" / "state_n.csv"
    if not path.exists():
        return {}
    with open(path) as fh:
        return {r["state"]: int(r["n"]) for r in csv.DictReader(fh)}


def rake(d, choices, states, table, iters=12, sizes=None, national=None, m=0.0):
    """Rescale densities so each state's weighted mean matches the published split.

    Cells outside every state are left alone, and a choice the table reports as
    absent is floored rather than zeroed, so no surface ever claims an answer is
    outright impossible.

    The published percentages are not equally trustworthy. Delaware's rest on 44
    respondents and California's on 1,943, yet both are printed to two decimals.
    Given `sizes` and `national`, each state's target is first pulled toward the
    national split with weight m, so a small state contributes its shape without
    imposing its noise.
    """
    d = d.copy()
    labels = [s for s in np.unique(states) if s]
    masks = {s: (states == s) for s in labels}

    nat = None
    if national and m > 0:
        nat = np.array([national.get(c, 0.0) for c in choices])
        if nat.sum() > 0:
            nat = nat / nat.sum()
        else:
            nat = None

    target = {}
    for s in labels:
        col = np.array(
            [max(table.get(c, {}).get(s, 0.0), MIN_PUBLISHED) for c in choices]
        )
        tot = col.sum()
        if tot <= 0:
            continue
        col = col / tot
        if nat is not None:
            n = float((sizes or {}).get(s, 0))
            col = (n * col + m * nat) / (n + m)
        target[s] = col

    total = d.sum()
    for _ in range(iters):
        w = d.sum(axis=0)
        p = d / np.maximum(w, 1e-12)
        for s, msk in masks.items():
            if s not in target:
                continue
            wm = w[msk]
            tw = wm.sum()
            if tw <= 0:
                continue
            share = (p[:, msk] * wm).sum(axis=1) / tw
            factor = np.where(share > 1e-9, target[s] / np.maximum(share, 1e-9), 1.0)
            d[:, msk] *= np.clip(factor, 0.02, 50.0)[:, None]
        d *= total / max(d.sum(), 1e-12)
    return d


def build(surfaces, sigma=3.0, alpha=0.05, gamma=1.0, questions=None,
          states=None, table=None, sizes=None, national=None, m=0.0, box=BOX):
    """Return {question: (choices, P)} where P has shape (n_choices, H, W).

    alpha is expressed as a fraction of the national mean density, so it acts
    like a Dirichlet prior worth alpha * (typical local sample) observations.

    gamma undoes contrast compression. A plotted dot is about a cell wide, so
    every dot smears its owner's answer across neighbouring cells and mixes in
    whatever their neighbours said; clipping saturated coverage at MAX_FRACTION
    truncates the rest. Both pull local log-odds toward zero by roughly a
    constant factor, so raising density to a power restores them. Fitted
    externally, not chosen.

    Pass `states` (a label raster) and `table` (published_state_pct) to rake,
    and `sizes`/`national`/`m` to shrink small states' targets first.
    """
    out = {}
    for q in (questions or surfaces.questions):
        idx = surfaces.rows[q]
        choices = list(surfaces.choice[idx])
        d = np.stack([density(surfaces.cov[i], sigma, box) for i in idx])
        if gamma != 1.0:
            d = d ** gamma
        if states is not None and table is not None:
            d = rake(d, choices, states, table.get(q, {}), sizes=sizes,
                     national=(national or {}).get(q), m=m)
        share = d.sum(axis=(1, 2))
        share = share / max(share.sum(), 1e-12)
        scale = float(d.sum(axis=0).mean())
        prior = alpha * scale * share[:, None, None]
        num = d + prior
        p = num / np.maximum(num.sum(axis=0, keepdims=True), 1e-12)
        out[q] = (np.array(choices), p.astype(np.float32))
    return out


def region_shares(p, raster, weight):
    """Weighted mean of P over the cells of each labelled region."""
    shares = {}
    flat_r = raster.ravel()
    flat_w = weight.ravel()
    flat_p = p.reshape(p.shape[0], -1)
    order = np.argsort(flat_r)
    sorted_r = flat_r[order]
    uniq = np.unique(sorted_r)
    edges = np.searchsorted(sorted_r, uniq)
    for i, key in enumerate(uniq):
        if key == "":
            continue
        lo = edges[i]
        hi = edges[i + 1] if i + 1 < len(edges) else len(sorted_r)
        cells = order[lo:hi]
        w = flat_w[cells]
        tot = w.sum()
        if tot <= 0:
            continue
        shares[key] = (flat_p[:, cells] * w).sum(axis=1) / tot
    return shares
