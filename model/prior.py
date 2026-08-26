"""Where a person could plausibly be from, before they say anything.

The likelihood alone is not enough. It is a smooth surface, so an empty cell in
eastern Montana can match a set of answers about as well as Chicago does, and
the posterior will happily put mass in country where almost nobody grew up.
Weighting by population fixes that, and it is the difference between naming a
city and naming a rectangle of high desert.

Which population, though, matters more than it looks. The default is 2003,
the survey year. Today's adults grew up before the Sun Belt reached its present
size: Maricopa County has roughly doubled since 1990, so a 2024 prior would
quietly assume Phoenix raised as many people as it currently houses. 2003 is
both closer to the childhoods in question and consistent with the population
the dialect surfaces were drawn from.

The comparison against respondent density looks like a measurement of who the
survey reached, and it is not. Respondent density is read off the same dot maps,
so a city whose dots overlap into a solid blob reads as far fewer respondents
than it had: New York comes out at about a twenty-sixth of its population share,
which is saturation, not sampling. Use it as a rough diagnostic of where the
surfaces are dense, never as a bias estimate. The only clean measurement of the
survey's reach is data/hds/state_n.csv, recovered from rounding granularity in
scrape/hds_counts.py, and it is per state.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from geo_util import county_raster
from tensor import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

SMOOTH = 1.5  # county boundaries are not population boundaries
FLOOR = 1e-9


def county_population(vintage="pop2003"):
    pop = {}
    with open(DATA / "census" / "counties.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = r.get(vintage, "")
            if v:
                pop[r["fips"]] = float(v)
    return pop


def population_prior(tensor, vintage="pop2003", smooth=SMOOTH):
    """Per-cell prior aligned with `tensor`'s cell ordering, summing to 1.

    County totals are spread evenly across the cells of each county, which is
    wrong inside a big western county but right at the scale anything else here
    operates at. A light blur follows, because a county line is an
    administrative fact and not a demographic one.
    """
    counties = county_raster()
    pop = county_population(vintage)
    per_cell = np.zeros(counties.shape, dtype=np.float64)

    labels, counts = np.unique(counties, return_counts=True)
    size = dict(zip(labels, counts))
    for fips, total in pop.items():
        n = size.get(fips, 0)
        if n:
            per_cell[counties == fips] = total / n

    missing = sum(t for f, t in pop.items() if not size.get(f))
    if smooth:
        per_cell = gaussian_filter(per_cell, smooth, mode="constant")

    v = per_cell[tensor.cell_y, tensor.cell_x]
    v = np.maximum(v, FLOOR)
    return v / v.sum(), missing


def respondent_prior(tensor):
    """Where the survey's respondents actually were, for comparison."""
    from likelihood import Surfaces, density
    s = Surfaces()
    total = np.zeros(s.shape, dtype=np.float64)
    for q in s.questions:
        total += density(s.cov[s.rows[q]].sum(axis=0), 8.0)
    v = np.maximum(total[tensor.cell_y, tensor.cell_x], FLOOR)
    return v / v.sum()


def main():
    t = Tensor()
    pop, missing = population_prior(t)
    resp = respondent_prior(t)
    print(f"{t.n_cells} cells; {missing:,.0f} people in counties with no cell "
          f"(Alaska, Hawaii, tiny islands)")

    ratio = np.log2(resp / np.maximum(pop, 1e-15))
    order = np.argsort(ratio)
    print("\nsurvey sampling bias, log2(respondent share / population share)")
    print("\nmost over-sampled")
    for i in order[::-1][:8]:
        print(f"  {t.state[i]}  {t.cell_lat[i]:5.1f} {t.cell_lon[i]:7.1f}  {ratio[i]:+5.2f}")
    print("\nmost under-sampled")
    for i in order[:8]:
        print(f"  {t.state[i]}  {t.cell_lat[i]:5.1f} {t.cell_lon[i]:7.1f}  {ratio[i]:+5.2f}")

    by_state = {}
    for s in sorted(set(t.state)):
        m = t.state == s
        by_state[s] = np.log2(resp[m].sum() / max(pop[m].sum(), 1e-15))
    ranked = sorted(by_state.items(), key=lambda kv: kv[1])
    print("\nby state, under-sampled first")
    for s, v in ranked[:6]:
        print(f"  {s} {v:+5.2f}")
    print("  ...")
    for s, v in ranked[-6:]:
        print(f"  {s} {v:+5.2f}")

    out = DATA / "model" / "prior.npz"
    np.savez_compressed(out, population=pop, respondent=resp,
                        state=t.state, lat=t.cell_lat, lon=t.cell_lon)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
