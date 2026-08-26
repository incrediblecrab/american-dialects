"""Fit the smoothing bandwidth and shrinkage against external data.

The Harvard survey's own numbers cannot tune the surfaces that were derived
from its own maps without circularity. Pop vs. Soda is independent: a different
survey, run by different people over a different period, with ~294,000
geolocated responses aggregated to 3,141 counties. HDS question 105 asks the
same thing, so predicting the county splits is a genuine out-of-sample test of
whether the recovered geography is right at sub-state resolution.

Scored by multinomial log-loss per response, so large counties count more.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_util import county_raster, state_raster  # noqa: E402
from likelihood import (  # noqa: E402
    Surfaces, build, density, national_pct, published_state_pct,
    region_shares, state_sample_sizes,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

SODA_Q = "105"
# HDS choice -> Pop vs. Soda category
GROUPS = {"soda": ["a"], "pop": ["b"], "coke": ["c", "g"],
          "other": ["d", "e", "f", "h", "i", "j"]}
CATS = ["pop", "soda", "coke", "other"]


def load_counties():
    obs = {}
    with open(DATA / "popvssoda" / "counties.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = int(r["SUMCOUNT"] or 0)
            if n <= 0:
                continue
            counts = np.array([int(r["SUMPOP"] or 0), int(r["SUMSODA"] or 0),
                               int(r["SUMCOKE"] or 0), int(r["SUMOTHER"] or 0)],
                              dtype=np.float64)
            if counts.sum() <= 0:
                continue
            obs[r["FIPS_combo"].zfill(5)] = counts
    return obs


def score(surfaces, raster, obs, sigma, alpha, gamma=1.0, states=None, table=None,
          sizes=None, national=None, m=0.0, box=None):
    from likelihood import BOX as _BOX
    box = _BOX if box is None else box
    choices, p = build(surfaces, sigma=sigma, alpha=alpha, gamma=gamma,
                       questions=[SODA_Q], states=states, table=table,
                       sizes=sizes, national=national, m=m, box=box)[SODA_Q]
    pos = {c: i for i, c in enumerate(choices)}
    grouped = np.stack([
        p[[pos[c] for c in GROUPS[cat] if c in pos]].sum(axis=0) for cat in CATS
    ])
    grouped = grouped / np.maximum(grouped.sum(axis=0, keepdims=True), 1e-12)

    weight = density(surfaces.cov[surfaces.rows[SODA_Q]].sum(axis=0), sigma, box)
    pred = region_shares(grouped, raster, weight)

    ll, n, rows = 0.0, 0.0, []
    for fips, counts in obs.items():
        q = pred.get(fips)
        if q is None:
            continue
        q = np.clip(q, 1e-6, 1.0)
        q = q / q.sum()
        ll += float((counts * np.log(q)).sum())
        n += float(counts.sum())
        rows.append((counts / counts.sum(), q, counts.sum()))
    if not rows:
        return None
    a = np.array([r[0] for r in rows])
    b = np.array([r[1] for r in rows])
    w = np.array([r[2] for r in rows])
    r_pop = float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
    r_soda = float(np.corrcoef(a[:, 1], b[:, 1])[0, 1])
    r_coke = float(np.corrcoef(a[:, 2], b[:, 2])[0, 1])
    modal = float((a.argmax(1) == b.argmax(1)).astype(float) @ w / w.sum())
    return {"logloss": -ll / n, "counties": len(rows), "responses": int(n),
            "r_pop": r_pop, "r_soda": r_soda, "r_coke": r_coke, "modal": modal}


def shrink_stage(surfaces, raster, obs, bl):
    """At the fitted settings, how hard should small states be pulled home?"""
    from tensor import SIGMA, GAMMA, ALPHA
    table, states = published_state_pct(), state_raster()
    sizes, national = state_sample_sizes(), national_pct()
    print(f"sigma={SIGMA} gamma={GAMMA} alpha={ALPHA}, sweeping shrinkage m\n")
    print(f"{'m':>7} {'logloss':>9} {'r_pop':>7} {'r_soda':>7} {'r_coke':>7} {'modal':>7}")
    rows = []
    for m in [0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]:
        s = score(surfaces, raster, obs, SIGMA, ALPHA, GAMMA, states, table,
                  sizes, national, m)
        rows.append((s["logloss"], m, s))
        print(f"{m:7.0f} {s['logloss']:9.4f} {s['r_pop']:7.3f} {s['r_soda']:7.3f} "
              f"{s['r_coke']:7.3f} {s['modal']:7.3f}")
    rows.sort()
    print(f"\nbest m={rows[0][1]:.0f} logloss={rows[0][0]:.4f} "
          f"(baseline {bl:.4f}, improvement {bl - rows[0][0]:.4f} nats/response)")


def box_stage(surfaces, raster, obs, bl):
    """Does the saturation window belong at the dot scale or the pooling scale?

    -log1p(-f) inverts saturation only if f is measured over a window where
    dots really do land independently. Too wide and the convexity of the log
    blunts sharp features before pooling; too narrow and a couple of dots
    already fill the window, so the correction clips instead of inverting.
    Box and sigma both blur, so they are swept together rather than in turn.
    """
    from tensor import GAMMA, ALPHA
    table, states = published_state_pct(), state_raster()
    print(f"gamma={GAMMA} alpha={ALPHA}, sweeping saturation box x sigma")
    print("(box is in 12.7 km cells; box=9 is a 114 km window)\n")
    print(f"{'box':>5} {'km':>6} {'sigma':>6} {'logloss':>9} {'r_pop':>7} "
          f"{'r_soda':>7} {'modal':>7}")
    rows = []
    for box in [1, 3, 5, 7, 9, 13, 17]:
        for sigma in [4.0, 6.0, 8.0, 10.0]:
            s = score(surfaces, raster, obs, sigma, ALPHA, GAMMA, states, table,
                      box=box)
            rows.append((s["logloss"], box, sigma, s))
            print(f"{box:5d} {box * 12.7:6.0f} {sigma:6.1f} {s['logloss']:9.4f} "
                  f"{s['r_pop']:7.3f} {s['r_soda']:7.3f} {s['modal']:7.3f}")
        print()
    rows.sort()
    ll, box, sigma, s = rows[0]
    print(f"best box={box} ({box * 12.7:.0f} km) sigma={sigma} logloss={ll:.4f} "
          f"(baseline {bl:.4f})")
    cur = [r for r in rows if r[1] == 9 and r[2] == 8.0]
    if cur:
        print(f"deployed box=9 sigma=8 logloss={cur[0][0]:.4f}, "
              f"difference {cur[0][0] - ll:+.4f} nats/response")


def main():
    surfaces = Surfaces()
    raster = county_raster()
    obs = load_counties()
    print(f"external target: {len(obs)} counties, "
          f"{int(sum(c.sum() for c in obs.values()))} responses\n")

    baseline = np.zeros(4)
    for c in obs.values():
        baseline += c
    baseline /= baseline.sum()
    bl = -sum(float((c * np.log(np.clip(baseline, 1e-9, 1))).sum())
              for c in obs.values()) / sum(float(c.sum()) for c in obs.values())
    print(f"baseline (national split, no geography): logloss {bl:.4f}\n")

    table = published_state_pct()
    states = state_raster()

    if "--box" in sys.argv:
        box_stage(surfaces, raster, obs, bl)
        return

    if "--shrink" in sys.argv:
        shrink_stage(surfaces, raster, obs, bl)
        return

    print(f"{'rake':>5} {'sigma':>6} {'gamma':>6} {'alpha':>7} {'logloss':>9} "
          f"{'r_pop':>7} {'r_soda':>7} {'r_coke':>7} {'modal':>7}")
    results = []
    for rk in [False, True]:
        for sigma in [2.0, 3.0, 4.0, 6.0, 8.0]:
            for gamma in [1.0, 1.5, 2.0, 2.5, 3.0]:
                for alpha in [0.0, 0.02, 0.05, 0.15]:
                    s = score(surfaces, raster, obs, sigma, alpha, gamma,
                              states if rk else None, table if rk else None)
                    if not s:
                        continue
                    results.append((s["logloss"], sigma, alpha, gamma, rk, s))
                    print(f"{str(rk):>5} {sigma:6.1f} {gamma:6.1f} {alpha:7.2f} "
                          f"{s['logloss']:9.4f} {s['r_pop']:7.3f} "
                          f"{s['r_soda']:7.3f} {s['r_coke']:7.3f} {s['modal']:7.3f}")

    results.sort()
    best = results[0]
    print(f"\nbest: rake={best[4]} sigma={best[1]} gamma={best[3]} alpha={best[2]} "
          f"logloss={best[0]:.4f} "
          f"(baseline {bl:.4f}, improvement {bl - best[0]:.4f} nats/response)")
    out = DATA / "model"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tuning.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rake", "sigma", "gamma", "alpha", "logloss",
                    "r_pop", "r_soda", "r_coke", "modal"])
        for ll, sg, al, gm, rk, s in sorted(results, key=lambda x: (x[4], x[1], x[3], x[2])):
            w.writerow([int(rk), sg, gm, al, f"{ll:.5f}", f"{s['r_pop']:.4f}",
                        f"{s['r_soda']:.4f}", f"{s['r_coke']:.4f}", f"{s['modal']:.4f}"])
    print(f"wrote {out / 'tuning.csv'}")


if __name__ == "__main__":
    main()
