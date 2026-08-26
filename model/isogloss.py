"""Check that the fitted surfaces keep small-scale isoglosses apart.

External tuning (tune.py) scores county-level agreement over the whole country,
which rewards broad regional accuracy. That metric cannot see whether a setting
has smeared away a city-sized feature, and city-sized features are what a
geolocator lives on. This script tests documented contrasts instead.

Two kinds of case, marked in the `src` column:

  pvs   ground truth from Pop vs. Soda county data, an independent survey
  lex   lexical isoglosses documented in the dialectology literature

The within-state cases matter most. Raking applies one factor per state, so it
cannot invent a contrast between Pittsburgh and Philadelphia; anything the model
gets right there came from the recovered map geography alone.
"""

import argparse

import numpy as np

from geo_util import state_raster
from likelihood import Surfaces, build, published_state_pct

CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "Philadelphia": (39.95, -75.17),
    "Boston": (42.36, -71.06),
    "Providence": (41.82, -71.41),
    "NYC": (40.71, -74.01),
    "Buffalo": (42.89, -78.88),
    "Milwaukee": (43.04, -87.91),
    "Minneapolis": (44.98, -93.27),
    "New Orleans": (29.95, -90.07),
    "Atlanta": (33.75, -84.39),
    "Chicago": (41.88, -87.63),
    "Seattle": (47.61, -122.33),
}

# question, choice, label, high city, low city, min gap (points), source, same-state
CASES = [
    ("105", "b", "pop: Pittsburgh vs Philadelphia", "Pittsburgh", "Philadelphia", 40, "pvs", True),
    ("105", "b", "pop: Buffalo vs NYC", "Buffalo", "NYC", 40, "pvs", True),
    ("105", "a", "soda: Milwaukee vs Minneapolis", "Milwaukee", "Minneapolis", 30, "pvs", False),
    ("105", "c", "coke: Atlanta vs Boston", "Atlanta", "Boston", 30, "pvs", False),
    ("50", "f", "yins/yinz Pittsburgh", "Pittsburgh", "Philadelphia", 6, "lex", True),
    ("64", "c", "hoagie Philadelphia", "Philadelphia", "Pittsburgh", 12, "lex", True),
    ("64", "b", "grinder Boston", "Boston", "NYC", 10, "lex", False),
    ("64", "e", "poor boy New Orleans", "New Orleans", "Atlanta", 10, "lex", False),
    ("63", "b", "frappe Boston", "Boston", "NYC", 12, "lex", False),
    ("63", "c", "cabinet Providence", "Providence", "Boston", 3, "lex", False),
    ("50", "i", "y'all Atlanta", "Atlanta", "Minneapolis", 40, "lex", False),
    ("50", "d", "you guys Chicago", "Chicago", "Atlanta", 30, "lex", False),
    ("103", "a", "bubbler Milwaukee", "Milwaukee", "Atlanta", 10, "lex", False),
    ("73", "c", "gymshoes Chicago", "Chicago", "Seattle", 8, "lex", False),
    ("66", "a", "crawfish New Orleans", "New Orleans", "Seattle", 25, "lex", False),
    ("65", "a", "lightning bug Atlanta", "Atlanta", "Seattle", 30, "lex", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="4,6,8")
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--no-rake", action="store_true")
    args = ap.parse_args()

    surfaces = Surfaces()
    states = None if args.no_rake else state_raster()
    table = None if args.no_rake else published_state_pct()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    pts = {n: surfaces.cell(la, lo) for n, (la, lo) in CITIES.items()}

    print(f"gamma={args.gamma} alpha={args.alpha} rake={not args.no_rake}\n")
    head = f"{'case':<34}{'src':<5}{'in-st':<7}"
    print(head + "".join(f"{s:>15.0f}" for s in sigmas))
    print("-" * (len(head) + 15 * len(sigmas)))

    cache, passed = {}, {s: [] for s in sigmas}
    for q, ch, label, hi, lo, need, src, same in CASES:
        row = f"{label:<34}{src:<5}{('yes' if same else '-'):<7}"
        for s in sigmas:
            if (q, s) not in cache:
                cache[(q, s)] = build(surfaces, sigma=s, alpha=args.alpha,
                                      gamma=args.gamma, questions=[q],
                                      states=states, table=table)[q]
            choices, p = cache[(q, s)]
            k = list(choices).index(ch)
            (yh, xh), (yl, xl) = pts[hi], pts[lo]
            a, b = p[k, yh, xh] * 100, p[k, yl, xl] * 100
            ok = (a - b) >= need
            passed[s].append(ok)
            row += f"{a:>6.1f}/{b:<5.1f}{'ok  ' if ok else 'FAIL'}"
        print(row)

    print("-" * (len(head) + 15 * len(sigmas)))
    print(f"{'passed':<46}" + "".join(f"{sum(passed[s]):>11}/{len(CASES)}" for s in sigmas))


if __name__ == "__main__":
    main()
