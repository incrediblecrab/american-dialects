"""Freeze the fitted likelihood into one array the inference engine can load.

Everything upstream of this file is measurement: recovering geography from
pixels, fusing it with the published state table, fitting three parameters
against an independent survey. This is where that stops and the model begins.

The settings below are not adjustable knobs. sigma, gamma and alpha were fitted
in tune.py against Pop vs. Soda, a survey the surfaces never saw, and checked in
isogloss.py against documented dialect boundaries. Those two checks are related,
not independent: four of the sixteen isogloss cases are drawn from Pop vs. Soda
itself, and the pass thresholds are hand-set, so treat the isogloss pass as a
guard against over-smoothing rather than as confirmation. The quantitative
evidence for sigma=8 is the external log-loss, which is flat from about 6 to 10.

The saturation window BOX was swept against the same external target across
1 to 17 cells. It is flat from 5 to 17 and worse at 1, so the objection that a
114 km window is far wider than a dot is correct in principle and worth about
0.0006 nats per response in practice. Left at 9.

A single global sigma is also open to the objection that signal sharpness
varies: yinz is a point feature, you guys is a national gradient, and one
bandwidth cannot suit both. model/bandwidth.py tested it by spatial block
cross-validation of the maps themselves, and the sharpness is real -- measured
per choice, yinz, yous, bubbler, grinder and hoagie come out sharp and you guys,
soda, y'all, sub and coke come out diffuse, all as predicted before fitting.
Fitting sigma per question nonetheless buys at most 0.077% of the log-loss, so
the objection is real in principle and negligible in practice. Left global.

One caveat on that test, because it matters for reading its output: block CV
scores prediction across a held-out void, which structurally rewards maximal
smoothing, and its per-question optima pile up at the top of whatever sweep it
is given. It cannot set the bandwidth's LEVEL. It can only answer the
comparative question of whether per-question beats global, and it does.

Stored over land cells only. Two thirds of the 200x456 grid is ocean, Canada or
Mexico, and a posterior has no business putting mass there.
"""

import sys
from pathlib import Path

import numpy as np

from geo_util import state_raster
from likelihood import Surfaces, build, published_state_pct

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

SIGMA = 8.0
GAMMA = 1.5
ALPHA = 0.02
RAKE = True

PATH = DATA / "model" / "likelihood.npz"
FLOOR = 1e-4  # no answer is ever impossible anywhere


def build_tensor(force=False):
    if PATH.exists() and not force:
        return
    surfaces = Surfaces()
    states = state_raster()
    table = published_state_pct() if RAKE else None
    mask = states != ""
    ys, xs = np.nonzero(mask)
    print(f"{mask.sum()} land cells, {len(surfaces.questions)} questions")

    rows, qs, chs = [], [], []
    for i, q in enumerate(surfaces.questions, 1):
        choices, p = build(surfaces, sigma=SIGMA, alpha=ALPHA, gamma=GAMMA,
                           questions=[q], states=states if RAKE else None,
                           table=table)[q]
        p = np.maximum(p[:, ys, xs], FLOOR)
        p /= p.sum(axis=0, keepdims=True)
        rows.append(np.log(p).astype(np.float32))
        qs.extend([q] * len(choices))
        chs.extend(choices)
        if i % 20 == 0 or i == len(surfaces.questions):
            print(f"  {i}/{len(surfaces.questions)}")

    PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PATH,
        logp=np.concatenate(rows), question=np.array(qs), choice=np.array(chs),
        cell_y=ys.astype(np.int16), cell_x=xs.astype(np.int16),
        state=states[ys, xs], lats=surfaces.lats, lons=surfaces.lons,
        settings=np.array([SIGMA, GAMMA, ALPHA, float(RAKE)]),
    )
    print(f"wrote {PATH} ({PATH.stat().st_size / 1e6:.0f} MB)")


class Tensor:
    """log P(answer | cell) for every answer, over land cells."""

    def __init__(self):
        z = np.load(PATH, allow_pickle=True)
        self.logp = z["logp"]
        self.question = np.array([str(q) for q in z["question"]])
        self.choice = np.array([str(c) for c in z["choice"]])
        self.cell_y = z["cell_y"].astype(int)
        self.cell_x = z["cell_x"].astype(int)
        self.state = np.array([str(s) for s in z["state"]])
        self.lats = z["lats"]
        self.lons = z["lons"]
        self.questions = sorted(set(self.question), key=int)
        self.rows = {q: np.nonzero(self.question == q)[0] for q in self.questions}
        self.cell_lat = self.lats[self.cell_y]
        self.cell_lon = self.lons[self.cell_x]

    @property
    def n_cells(self):
        return self.logp.shape[1]

    def nearest(self, lat, lon):
        """Index of the land cell closest to a point, by great-circle distance."""
        return int(np.argmin(haversine(lat, lon, self.cell_lat, self.cell_lon)))

    def grid(self, values):
        """Scatter a per-cell vector back onto the full 200x456 grid."""
        g = np.full((len(self.lats), len(self.lons)), np.nan, dtype=np.float64)
        g[self.cell_y, self.cell_x] = values
        return g


def haversine(lat1, lon1, lat2, lon2):
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dp = r2 - r1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dl / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


if __name__ == "__main__":
    build_tensor(force="--force" in sys.argv)
    t = Tensor()
    print(f"loaded: {t.logp.shape[0]} answers x {t.n_cells} cells")
