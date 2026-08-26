"""Recover how many people the survey actually heard from in each state.

The mirror publishes per-state percentages but never the sample sizes, which
matters more than it sounds: a percentage from Alaska and a percentage from
California are printed identically and are not remotely the same measurement.
Without the denominators there is no way to know which state results deserve
trust, and the model fuses those results into every surface it builds.

The counts are recoverable. Percentages are rounded to two decimals, so for a
state answered by n people every printed value must sit within half a rounding
step of some multiple of 100/n, and the implied counts must total n exactly.
Sweeping n and keeping the candidates that satisfy both conditions leaves, in
the clean cases, precisely the multiples of the true n. So the estimate is the
smallest candidate, accepted only when the rest of the surviving set really is
its multiples; that check is what separates a determined answer from a
coincidence.

Two properties of the result are worth noting because they were not assumed.
Within a state the recovered n barely moves across the 122 questions, which
says the state pages divide by a fixed denominator rather than by the number of
people who answered each question. And the estimates from different questions
agree to within a fraction of a percent, which a spurious fit would not do.

This is also the only unconfounded measure of the survey's geographic reach.
Respondent density read off the dot maps cannot do it: in dense metros the dots
merge, so New York City looks sparse when it is merely saturated.
"""

import csv
from collections import defaultdict

import numpy as np

from common import DATA, out_dir

MAX_N = 4000  # above this the rounding tolerance stops discriminating
MIN_CHOICES = 4
MULTIPLE_FRACTION = 0.85  # of surviving candidates, how many must be multiples


def load():
    pct = defaultdict(dict)
    with open(DATA / "hds" / "state_pct.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pct[(r["state"], r["question"])][r["choice"]] = float(r["pct"])
    return pct


def candidates(values, grid, tol):
    """Every n for which the printed percentages are a consistent rounding."""
    p = np.array([v for v in values if v > 0]) / 100.0
    if len(p) < MIN_CHOICES:
        return None
    k = np.outer(grid, p)
    near = np.abs(k - np.round(k)).max(axis=1) <= tol
    totals = np.round(k).sum(axis=1) == grid
    return grid[near & totals]


def main():
    pct = load()
    grid = np.arange(2, MAX_N + 1)
    tol = grid / 20000.0 + 1e-9  # a 2-dp percentage is accurate to +/-0.005

    per_state = defaultdict(list)
    tested = defaultdict(int)
    rows = []
    for (state, question), choices in sorted(pct.items()):
        fit = candidates(list(choices.values()), grid, tol)
        if fit is None or len(fit) == 0:
            continue
        tested[state] += 1
        base = int(fit[0])
        if np.mean(fit % base == 0) >= MULTIPLE_FRACTION:
            per_state[state].append(base)
            rows.append({"state": state, "question": question, "n": base})

    out = out_dir("hds")
    with open(out / "state_n_by_question.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["state", "question", "n"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["state"], int(r["question"]))):
            w.writerow(r)

    summary = []
    for state, ns in per_state.items():
        a = np.array(ns)
        summary.append({
            "state": state,
            "n": int(np.median(a)),
            "n_p25": int(np.percentile(a, 25)),
            "n_p75": int(np.percentile(a, 75)),
            "determined": len(a),
            "tested": tested[state],
        })
    summary.sort(key=lambda r: -r["n"])

    with open(out / "state_n.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["state", "n", "n_p25", "n_p75",
                                          "determined", "tested"])
        w.writeheader()
        w.writerows(summary)

    total = sum(r["n"] for r in summary)
    print(f"{len(summary)} states determined, {total:,} respondents in total "
          f"(the survey reports 30,788)\n")
    print(f"{'state':<7}{'n':>7}{'iqr':>14}{'from':>10}")
    for r in summary[:12]:
        print(f"{r['state']:<7}{r['n']:>7}{r['n_p25']:>8}-{r['n_p75']:<5}"
              f"{r['determined']:>4}/{r['tested']:<5}")
    print("  ...")
    for r in summary[-10:]:
        print(f"{r['state']:<7}{r['n']:>7}{r['n_p25']:>8}-{r['n_p75']:<5}"
              f"{r['determined']:>4}/{r['tested']:<5}")
    print(f"\nwrote {out / 'state_n.csv'} and {out / 'state_n_by_question.csv'}")


if __name__ == "__main__":
    main()
