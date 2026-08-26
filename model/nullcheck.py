"""A negative control for the calibrator: people who obey the model exactly.

The calibrator's job is to detect that the model is overconfident and to say by
how much. Before believing it about real people, it is worth checking that it
does not cry wolf. So this builds a population of simulated respondents drawn
from the model's own surfaces, with each answer drawn independently given the
true cell, and hands them to the calibrator.

That simulated world satisfies conditional independence by construction, so
tau=1 is correct there and eps=0 is correct there, and there is genuinely
nothing to fit. If the calibrator comes back with tau=1, eps=0 and a flat
probability-integral-transform histogram, it is not inventing corrections. If it
comes back with tau<1 here, the fitting procedure is biased and any tau it
reports on real people is suspect.

This is deliberately NOT a validation of the model. It cannot be: the audit of
this project made the point sharply, that a simulated world built from the
model's own beliefs is not merely uninformative about accuracy but actively
hides the one assumption most likely to be wrong. Its only use is as a control
on the instrument.

Both passes came back clean. Fitting tau and eps directly recovers tau=1.0,
eps=0.0, coverage 0.510/0.800/0.955 against nominal 0.50/0.80/0.95, and a PIT
mean of 0.497. Fitting the k-dependent parameterisation instead (--auto, where
tau_for(k) = base / (1 + (k-1)*RHO)) pins base at the top of its grid, which is
the same answer wearing different clothes: with no within-person correlation in
this world there is no design effect to discount, so the fitter tries to cancel
the RHO term and runs out of grid. A base that pinned LOW here would mean the
procedure manufactures overconfidence corrections out of nothing.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from infer import Geolocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

SEED = 20130801
DEFAULT_QUESTIONS = ["105", "50", "64", "103", "73", "63"]


def build(n=400, questions=None, seed=SEED, out="nullcheck"):
    questions = questions or DEFAULT_QUESTIONS
    rng = np.random.default_rng(seed)
    g = Geolocator()
    t = g.t
    cells = rng.choice(t.n_cells, size=n, p=g.prior)

    root = DATA / out
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "people.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person", "lat", "lon", "state", "truth_cell"])
        for i, c in enumerate(cells):
            w.writerow([f"n{i:05d}", f"{t.cell_lat[c]:.5f}",
                        f"{t.cell_lon[c]:.5f}", t.state[c], int(c)])

    with open(root / "answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person", "question", "choice"])
        for i, c in enumerate(cells):
            for q in questions:
                rows = t.rows[q]
                p = np.exp(t.logp[rows, c].astype(np.float64))
                p /= p.sum()
                w.writerow([f"n{i:05d}", q, t.choice[rows[rng.choice(len(p), p=p)]]])

    print(f"wrote {n} simulated respondents answering {len(questions)} "
          f"questions to {root}")
    print("now run:  ../.venv/bin/python calibrate.py --set " + out)
    print("expect tau=1.0, eps=0.0 and a flat histogram. Anything else means")
    print("the calibrator is biased, not that the model needs correcting.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--questions", default=",".join(DEFAULT_QUESTIONS))
    ap.add_argument("--out", default="nullcheck")
    a = ap.parse_args()
    build(a.n, [q for q in a.questions.split(",") if q], out=a.out)
