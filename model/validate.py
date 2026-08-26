"""How well does it actually work, and after how many questions?

This is the file that answers the question the project was started to answer:
what is the smallest number of questions that will place someone, and how
confident may you honestly be after asking them.

Everything here is measured on people whose locations are known and who are not
simulated. That distinction is the whole point. model/choose.py also prints an
error curve, and that curve is drawn from respondents invented by the model
itself, which answer independently given their location and therefore obey the
model's assumptions perfectly. It is a description of the model's beliefs about
its own competence. This file is the check on those beliefs.

Three resolutions get reported, because "where are you from" means different
things to different people and a single number hides which one succeeded:

    region  census division, nine of them, the level a party trick can be
            confident about after very few questions
    state   familiar, but a bad unit linguistically; the Pittsburgh and
            Philadelphia dialects share a state and share almost nothing else
    city    great-circle error from the estimate to the truth, reported as
            median and 90th percentile because the mean is meaningless when
            the distribution has a tail of people the model simply cannot place

Alongside each, the coverage of the 50, 80 and 95 percent regions. Accuracy
without calibration is only half an answer: a model that is right 40 percent of
the time and says so is more useful at a party than one that is right 60 percent
of the time and always claims certainty.

There is a floor on all of this that no number of questions can cross. Dialect
maps to location many to one. Two people from opposite ends of the Midland speak
alike; a person raised in three states speaks like none of them. For those
people the honest output is a wide posterior, and a correctly calibrated model
must produce one. Once calibration is fixed, some fraction of answers become
"I cannot place you", and that is the model working, not failing.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import infer
from calibrate import LEVELS
from infer import Geolocator, credible_cells
from people import People
from tensor import haversine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

DIVISION = {
    "CT": "New England", "ME": "New England", "MA": "New England",
    "NH": "New England", "RI": "New England", "VT": "New England",
    "NJ": "Middle Atlantic", "NY": "Middle Atlantic", "PA": "Middle Atlantic",
    "IL": "East North Central", "IN": "East North Central",
    "MI": "East North Central", "OH": "East North Central",
    "WI": "East North Central",
    "IA": "West North Central", "KS": "West North Central",
    "MN": "West North Central", "MO": "West North Central",
    "NE": "West North Central", "ND": "West North Central",
    "SD": "West North Central",
    "DE": "South Atlantic", "DC": "South Atlantic", "FL": "South Atlantic",
    "GA": "South Atlantic", "MD": "South Atlantic", "NC": "South Atlantic",
    "SC": "South Atlantic", "VA": "South Atlantic", "WV": "South Atlantic",
    "AL": "East South Central", "KY": "East South Central",
    "MS": "East South Central", "TN": "East South Central",
    "AR": "West South Central", "LA": "West South Central",
    "OK": "West South Central", "TX": "West South Central",
    "AZ": "Mountain", "CO": "Mountain", "ID": "Mountain", "MT": "Mountain",
    "NV": "Mountain", "NM": "Mountain", "UT": "Mountain", "WY": "Mountain",
    "AK": "Pacific", "CA": "Pacific", "HI": "Pacific", "OR": "Pacific",
    "WA": "Pacific",
}


def region_of(state):
    return DIVISION.get(str(state), "")


def score_at(g, people, order, k, tau, eps):
    """Ask the first k questions of `order` and see how everyone does."""
    err, region_ok, state_ok = [], [], []
    hits = {lv: [] for lv in LEVELS}
    allowed = set(order[:k])
    used = []
    for row, answers in people:
        subset = [(q, c) for q, c in answers if q in allowed]
        used.append(len(subset))
        post = g.posterior(subset, tau=tau or "auto", eps=eps)
        best = int(np.argmax(post))
        cell = g.t.nearest(row["lat"], row["lon"])
        err.append(haversine(row["lat"], row["lon"],
                             g.t.cell_lat[best], g.t.cell_lon[best]))
        truth_state = str(g.t.state[cell])
        state_ok.append(str(g.t.state[best]) == truth_state)
        region_ok.append(region_of(g.t.state[best]) == region_of(truth_state)
                         and region_of(truth_state) != "")
        for lv in LEVELS:
            hits[lv].append(cell in set(credible_cells(post, lv).tolist()))
    err = np.array(err)
    return {
        "k": k,
        "asked_median": float(np.median(used)),
        "median_km": float(np.median(err)),
        "p90_km": float(np.percentile(err, 90)),
        "within_150km": float((err <= 150).mean()),
        "within_500km": float((err <= 500).mean()),
        "state_acc": float(np.mean(state_ok)),
        "region_acc": float(np.mean(region_ok)),
        **{f"cover{int(lv * 100)}": float(np.mean(hits[lv])) for lv in LEVELS},
    }


def question_order(people, path=None):
    """The greedy ordering from choose.py, filtered to what the set can test.

    A validation set that only covers four questions cannot test a twenty-five
    question quiz, so the curve stops where the evidence stops rather than
    padding itself out with questions nobody in the set answered.
    """
    have = {q for a in people.answers.values() for q, _ in a}
    order = []
    path = path or DATA / "model" / "question_order.csv"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["question"] in have:
                    order.append(r["question"])
    for q in sorted(have, key=lambda s: int(s) if s.isdigit() else 0):
        if q not in order:
            order.append(q)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="ygdp")
    ap.add_argument("--tau", type=float, default=None,
                    help="flat tau; default is the information-weighted schedule")
    ap.add_argument("--eps", type=float, default=None,
                    help="default is the model's fitted eps")
    ap.add_argument("--max-k", type=int, default=25)
    ap.add_argument("--rho", type=float, default=None,
                    help="override the within-person residual correlation; "
                         "rho=0 treats answers as independent. Sweeping this "
                         "brackets the one parameter measured on only two "
                         "syntactic items and unmeasured for lexical ones.")
    ap.add_argument("--out", default=None,
                    help="output basename; default is the validation set name")
    args = ap.parse_args()

    if args.rho is not None:
        infer.RHO = args.rho

    root = DATA / args.set
    if not (root / "answers.csv").exists():
        print(f"no validation set at {root}. See model/people.py for the format.")
        return

    g = Geolocator()
    people = People(root)
    s = people.summary()
    print(f"validation set: {args.set}")
    print(f"  {s['people']} people, {s['questions']} questions, "
          f"median {s['answers_per_person_median']:.0f} answers each")
    print(f"  tau={args.tau or 'auto'} eps={args.eps if args.eps is not None else g.eps} "
          f"rho={infer.RHO}\n")

    order = question_order(people)
    ks = [k for k in range(1, min(args.max_k, len(order)) + 1)]
    print(f"{'k':>3} {'asked':>6} {'median km':>10} {'p90 km':>8} "
          f"{'<150km':>7} {'state':>7} {'region':>7} " +
          " ".join(f"{f'cov{int(l * 100)}':>7}" for l in LEVELS))

    rows = []
    for k in ks:
        r = score_at(g, people, order, k, args.tau, args.eps)
        rows.append(r)
        print(f"{k:>3} {r['asked_median']:>6.0f} {r['median_km']:>10.0f} "
              f"{r['p90_km']:>8.0f} {r['within_150km']:>7.1%} "
              f"{r['state_acc']:>7.1%} {r['region_acc']:>7.1%} " +
              " ".join(f"{r[f'cover{int(l * 100)}']:7.3f}" for l in LEVELS))

    out = DATA / "model" / f"accuracy_{args.out or args.set}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    print("\nquestion order used:")
    print("  " + ", ".join(order[:len(ks)]))


if __name__ == "__main__":
    main()
