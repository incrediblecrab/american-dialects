"""Fit tau and eps so that a stated confidence means what it says.

The model's point estimate is only half the claim. Saying "you are from
Pittsburgh" is worth little without saying how sure, and a confidence is only
meaningful if it is right that often. This file is what turns the second half
into a measured quantity instead of a decoration.

Two knobs, fitted together because they trade against each other:

  tau  scales the whole log-likelihood, and so sets how wide the posterior is.
       It exists because answers are not independent given location; a
       Southerner's y'all and fixin' to and crawfish are one fact restated, and
       multiplying them as independent evidence produces a posterior far
       narrower than the evidence supports.

  eps  mixes each answer toward its national marginal, and so sets how much
       damage one answer can do. It exists because real people misclick, say
       both, and move.

Fitted by maximising the log posterior probability of the true cell, summed over
people. That is a strictly proper scoring rule, so it cannot be gamed by simply
widening everything: a posterior spread thin over the whole country scores badly
even though its intervals always contain the truth. Sharpness and calibration
are traded correctly and automatically.

Calibration is then *checked*, not fitted, by asking whether the 50, 80 and 95
percent regions contain the truth that often, and by reliability curves within
demographic strata. Pooled calibration can be perfect while every subgroup is
wrong in opposite directions, which is the failure mode to expect here: the
model has no way to represent age, race, class or mobility, so it will place a
young mobile speaker wherever that kind of person was dense in 2003 and be
confident about it.

Nothing here uses simulated people. Simulated answers are drawn independently
given the cell, which satisfies naive Bayes exactly, so in the simulated world
tau=1 is already correct and there is nothing to fit.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from infer import Geolocator, RHO, tau_for_weights
from people import People
from tensor import haversine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

LEVELS = (0.5, 0.8, 0.95)


def evaluate(g, people, tau, eps, truth_lat="lat", truth_lon="lon", base=None):
    """Score every person once at one setting of the knobs.

    The posterior is sorted once per person and everything is read off that
    single ordering: the credible-region hits at each level, and the level at
    which the region first covers the truth. That last one is the probability
    integral transform, and it is the whole calibration curve in one number per
    person rather than three thresholds.
    """
    n = len(people)
    if n == 0:
        return None
    logp_true = np.empty(n)
    err = np.empty(n)
    pit = np.empty(n)
    hits = {lv: np.empty(n, dtype=bool) for lv in LEVELS}
    conf = np.empty(n)
    ids = []

    for i, (row, answers) in enumerate(people):
        t = tau
        if base is not None:
            bits = g.question_bits
            t = tau_for_weights(
                [bits.get(str(q), 0.0) for q, c in answers
                 if (str(q), str(c)) in g.index], base=base)
        post = g.posterior(answers, tau=t, eps=eps)
        cell = g.t.nearest(row[truth_lat], row[truth_lon])
        logp_true[i] = np.log(max(post[cell], 1e-300))

        order = np.argsort(post)[::-1]
        cum = np.cumsum(post[order])
        rank = int(np.nonzero(order == cell)[0][0])
        pit[i] = cum[rank]
        for lv in LEVELS:
            hits[lv][i] = rank <= np.searchsorted(cum, lv)

        best = int(order[0])
        err[i] = haversine(row[truth_lat], row[truth_lon],
                           g.t.cell_lat[best], g.t.cell_lon[best])
        conf[i] = post[best]
        ids.append(row["person"])

    return {
        "tau": tau if base is None else f"auto({base})", "eps": eps, "n": n,
        "base": base,
        "logp_true": logp_true, "err": err, "conf": conf, "hits": hits,
        "pit": pit, "ids": ids,
        "score": float(logp_true.mean()),
        "median_km": float(np.median(err)),
        "p90_km": float(np.percentile(err, 90)),
        "cover": {lv: float(hits[lv].mean()) for lv in LEVELS},
    }


def fit(g, people, taus, epss, verbose=True, auto=False):
    """Grid search. The grid is small and each evaluation is a few seconds.

    With auto=True the swept quantity is TAU_BASE under the k-dependent
    tau_for_weights, not a flat tau. That is the parameterisation that can be carried
    to a longer quiz, because it separates the part of the discount that grows
    with the number of questions from the part that does not.
    """
    if verbose:
        print(f"{'base' if auto else 'tau':>6} {'eps':>6} {'score':>9} "
              f"{'median km':>10} {'p90 km':>8} " +
              " ".join(f"{f'cov{int(l * 100)}':>7}" for l in LEVELS))
    best, results = None, []
    for tau in taus:
        for eps in epss:
            r = evaluate(g, people, None if auto else tau, eps,
                         base=tau if auto else None)
            if r is None:
                continue
            results.append(r)
            if verbose:
                cov = " ".join(f"{r['cover'][l]:7.3f}" for l in LEVELS)
                print(f"{tau:6.2f} {eps:6.3f} {r['score']:9.3f} "
                      f"{r['median_km']:10.0f} {r['p90_km']:8.0f} {cov}")
            if best is None or r["score"] > best["score"]:
                best = r
    return best, results


def reliability(result, bins=10):
    """Does the model's stated confidence match how often it is right?

    Confidence here is the posterior mass of the credible region that just
    contains the truth: for each person, the smallest level at which the region
    covers them. If the model is calibrated those levels are uniform on [0,1],
    so a flat histogram is the target. This is the probability integral
    transform, and it uses every person once rather than throwing away all but
    three thresholds.
    """
    levels = result["pit"]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.searchsorted(edges, levels, side="right") - 1, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins)
    return levels, edges, counts / max(counts.sum(), 1)


def stratified(people, base, field, bins=None):
    """Calibration within demographic strata, where it usually breaks."""
    groups = people.strata(field, bins)
    out = []
    index = {p: i for i, p in enumerate(base["ids"])}
    for name, members in sorted(groups.items()):
        idx = [index[p] for p in members if p in index]
        if len(idx) < 25:
            continue
        idx = np.array(idx)
        out.append({
            "group": name, "n": len(idx),
            "median_km": float(np.median(base["err"][idx])),
            "pit_mean": float(base["pit"][idx].mean()),
            **{f"cover{int(lv * 100)}": float(base["hits"][lv][idx].mean())
               for lv in LEVELS},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true",
                    help="fit TAU_BASE under the information-weighted schedule "
                         "instead of a flat tau")
    ap.add_argument("--set", default="ygdp",
                    help="folder under data/ holding people.csv and answers.csv")
    ap.add_argument("--strata", default="",
                    help="comma-separated fields; append :a,b,c to bin a "
                         "numeric field, e.g. age:25,35,45,60")
    args = ap.parse_args()

    root = DATA / args.set
    if not (root / "people.csv").exists():
        print(f"no validation set at {root}/people.csv")
        print("build one first; see model/people.py for the format")
        return

    people = People(root)
    s = people.summary()
    print(f"validation set: {args.set}")
    for k, v in s.items():
        if k != "question_list":
            print(f"  {k:<28} {v}")
    print(f"  questions used               {','.join(s['question_list'])}\n")

    g = Geolocator()

    print("--- prior only, no answers (the floor any model must beat) ---")
    empty = evaluate(g, _NoAnswers(people), 1.0, 0.0)
    if empty:
        cov = " ".join(f"cov{int(l * 100)}={empty['cover'][l]:.3f}" for l in LEVELS)
        print(f"  score {empty['score']:.3f}  median {empty['median_km']:.0f} km  {cov}\n")

    if args.auto:
        print(f"--- fitting TAU_BASE under tau_for_weights, rho={RHO} ---")
        taus = [0.2, 0.3, 0.41, 0.55, 0.7, 0.85, 1.0]
    else:
        print("--- fitting a flat tau and eps ---")
        taus = [0.15, 0.25, 0.35, 0.5, 0.7, 1.0]
    epss = [0.0, 0.02, 0.05, 0.10, 0.20]
    best, results = fit(g, people, taus, epss, auto=args.auto)
    label = "base" if args.auto else "tau"
    val = best["base"] if args.auto else best["tau"]
    print(f"\nbest {label}={val} eps={best['eps']} score={best['score']:.3f}")
    near = [r for r in results if r["score"] > best["score"] - 0.01]
    if len(near) > 1:
        vals = sorted({(r["base"] if args.auto else r["tau"]) for r in near})
        print(f"within 0.01 nats of the best: {label} in "
              f"[{min(vals)}, {max(vals)}] -- the surface is flat, so treat "
              f"the fitted value as an order of magnitude")
    if empty:
        print(f"gain over prior only: {best['score'] - empty['score']:.3f} "
              f"nats per person")

    print("\n--- calibration check at the fitted setting ---")
    levels, edges, frac = reliability(best)
    print("if calibrated, each decile below holds 0.100 of people")
    for i in range(len(frac)):
        bar = "#" * int(round(frac[i] * 200))
        print(f"  {edges[i]:.1f}-{edges[i + 1]:.1f} {frac[i]:6.3f} {bar}")
    print(f"  mean level {levels.mean():.3f} (0.500 if calibrated)")

    for spec in [f for f in args.strata.split(";") if f]:
        field, _, cuts = spec.partition(":")
        bins = [float(x) for x in cuts.split(",")] if cuts else None
        print(f"\n--- calibration by {field} ---")
        rows = stratified(people, best, field, bins)
        if not rows:
            print("  no group with at least 25 people")
            continue
        print(f"  {'group':<24} {'n':>5} {'median km':>10} {'pit':>6} " +
              " ".join(f"{f'cov{int(l * 100)}':>7}" for l in LEVELS))
        for r in rows:
            print(f"  {r['group'][:24]:<24} {r['n']:>5} {r['median_km']:>10.0f} "
                  f"{r['pit_mean']:6.3f} " +
                  " ".join(f"{r[f'cover{int(l * 100)}']:7.3f}" for l in LEVELS))

    out = DATA / "model" / f"calibration_{args.set}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tau", "base", "eps", "n", "score",
                                          "median_km", "p90_km"] +
                           [f"cover{int(l * 100)}" for l in LEVELS])
        w.writeheader()
        for r in results:
            w.writerow({"tau": r["tau"], "base": r["base"],
                        "eps": r["eps"], "n": r["n"],
                        "score": round(r["score"], 4),
                        "median_km": round(r["median_km"], 1),
                        "p90_km": round(r["p90_km"], 1),
                        **{f"cover{int(l * 100)}": round(r["cover"][l], 4)
                           for l in LEVELS}})
    print(f"\nwrote {out}")


class _NoAnswers:
    """The same people with their answers removed, to measure the prior alone."""

    def __init__(self, people):
        self.rows = people.rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for r in self.rows:
            yield r, []


if __name__ == "__main__":
    main()
