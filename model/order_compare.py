"""Does the regional-signal-to-idiolect ratio pick a better question set than
mutual information does?

model/signal_split.py measures, per question, how many bits the answer carries
about where the person is from and how many it carries about who they are.
model/choose.py currently ranks on the first of those alone. The hypothesis this
file tests is that ranking on the ratio picks a better set -- that a question
which is half idiolect is a worse use of a tap than its information suggests.

The comparison is like-for-like by construction. Every ordering comes out of the
same greedy loop in choose.py, against the same running posterior, on the same
400 personas, with the selection criterion as the single difference; the
criterion enters through `Selector.order(score=...)` and nothing else moves. The
default reproduces data/model/question_order.csv exactly, which is the check
that the machinery is the deployed machinery and not a lookalike.

Scoring is then done the way model/neural.py scores its curve: one population of
simulated people, drawn once, with a calibrated idiolect and 15% movers, and
every ordering asked of those same people. Because the people are shared, the
per-person errors are paired, so the difference between two orderings is tested
as a paired quantity rather than by comparing two independent medians. That
matters more than it sounds: the spread of error across people is enormous
compared with the difference between two orderings, and an unpaired comparison
at this sample size would not be able to see anything.

FOUR SELECTION ARMS, all fixed before any of them was scored:

  mi        the deployed ordering, greedy mutual information. Baseline.
  ratio     greedy on I(A;C) / I(A;U) -- the literal hypothesis. Note that a
            pure ratio is indifferent to magnitude, so it will take a clean
            question carrying a hundredth of a bit over a slightly dirty one
            carrying half a bit. That is a real property of the criterion and
            not a bug in the implementation; it is why the next arm exists.
  purity    greedy on I(A;C) * I(A;C)/(I(A;C) + I(A;U)) -- information, weighted
            by the fraction of it that is about geography. Bounded, so it cannot
            run off after tiny clean questions.
  net       greedy on I(A;C) - I(A;U) -- bits about the place minus bits about
            the person, on the same scale.

AND TWO ARMS THAT ARE EXPECTED TO LOSE, kept separate on purpose:

  mi+rho_q      the deployed ordering, scored with the per-question rho used as
                a per-question DISCOUNT on the evidence. This is a per-question
                version of exactly the design-effect correction that this
                project measured correctly, deployed, and then found to be the
                largest single source of error in the model.
  mi+rho=.177   the same ordering under the pooled discount that finding was
                originally about, so the two failures can be read side by side
                on the same people rather than across two documents.

Neither is wired into anything. The defensible use of a per-question ratio is
choosing which questions to ask. Discounting the answers you get back is the
move that already failed, and a per-question version of it fails the same way.

RESULT, on 8,000 identical simulated people, paired, k = 1 to 30. The hypothesis
is not supported. `ratio` is worse than mutual information at 21 of the 28 quiz
lengths where it picks a different set and better at one, costing a mean of
44 km and 64 km at the deployed fourteen. `net` is worse at 18 of 23, costing a
mean of 19 km. `purity`, the only arm that is even close, is significantly
better at 3 lengths and significantly worse at 2, for a mean of 4 km in its
favour -- which is within noise and not worth a second selection criterion.
Replicated on an independent draw of 8,000 people with the same sign
everywhere it was significant. Mutual information already captures what the
ratio was supposed to add.

The trap arms lose, as expected and by a lot: 818 km at k = 14 against 351, best
at k = 13 and then climbing 124 km by k = 30 -- the same turnover, at the same
place, as the pooled discount it generalises. The reason it does not merely
resemble the pooled failure but reproduces it is measurable: the mean pairwise
idiolect correlation over the deployed question set is 0.159 to 0.181 depending
on length, against 0.177 pooled over all 122. The questions the quiz asks are
not unusually clean on the idiolect axis, so a per-question discount has almost
nothing to be per-question about.

Run:  ./.venv/bin/python model/order_compare.py --n 8000 --kmax 30
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import idiolect as idio
import infer
import signal_split as ss
from calibrate import LEVELS
from choose import Selector
from infer import Geolocator
from neural import LEGACY_RHO
from tensor import haversine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

OUT = DATA / "model"
ORDER_OUT = OUT / "question_order_ratio.csv"
CURVE_OUT = OUT / "order_curve.csv"

SEED = 771131          # the same held-out people the neural curve is scored on
MOVER = 0.15           # one person in six raised somewhere other than recorded
BASELINE = "mi"


def criteria(sig):
    """The greedy objectives. `bits` is the conditional information of the
    question given the running posterior; the idiolect term is marginal, because
    U is independent of C and a posterior over cells carries no belief about the
    respondent's idiolect to condition on. That approximation is stated rather
    than hidden: conditioning would shrink I(A;U) slightly as the quiz goes on.
    """
    iu = {q: max(sig[q]["idiolect_bits"], 1e-9) for q in sig}
    share = {q: sig[q]["geo_share"] for q in sig}
    return {
        "ratio": lambda q, b: b / iu[q],
        "purity": lambda q, b: b * share[q],
        "net": lambda q, b: b - iu[q],
    }


def derive(g, sig, k, personas, stride, out=ORDER_OUT):
    """Every alternative ordering, in one file, with a criterion column."""
    s = Selector(g, cell_stride=stride)
    texts = ss.question_text()
    rows = []
    for name, fn in criteria(sig).items():
        picked, curve, _ = s.order(k=k, m=personas, score=fn, verbose=False)
        print(f"  {name:>7}: {', '.join(picked[:k])}")
        for r in curve:
            q = r["question"]
            rows.append({
                "criterion": name, "n": r["n"], "question": q,
                "bits": r["bits"], "entropy": r["entropy"],
                "mi_bits": sig[q]["mi_bits"],
                "idiolect_bits": sig[q]["idiolect_bits"],
                "ratio": sig[q]["ratio"], "geo_share": sig[q]["geo_share"],
                "selfconsistency_km": r["selfconsistency_km"],
                "selfconsistency_state_acc": r["selfconsistency_state_acc"],
                "text": texts.get(q, ""),
            })
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return {name: [r["question"] for r in rows if r["criterion"] == name]
            for name in criteria(sig)}


# ------------------------------------------------------------------ scoring

def contaminated(g):
    """The whole eps-mixed log-likelihood tensor, once.

    g.loglik() recomputes the mixture per answer per call, which is fine for one
    person and hopeless for a few hundred thousand posteriors.
    """
    LL = np.empty(g.t.logp.shape, dtype=np.float32)
    for i in range(0, LL.shape[0], 32):
        blk = g.t.logp[i:i + 32]
        if g.eps <= 0:
            LL[i:i + 32] = blk
        else:
            LL[i:i + 32] = np.logaddexp(
                np.log1p(-g.eps) + blk,
                np.log(g.eps * g.marginal[i:i + 32])[:, None]).astype(np.float32)
    return LL


def tau_pooled(bits, rho=None, base=None):
    """infer.tau_for_weights, re-implemented here only so the trap arm can vary
    rho without rebinding a module attribute the deployed path reads."""
    rho = infer.RHO if rho is None else rho
    base = infer.TAU_BASE if base is None else base
    w = np.sort(np.asarray([x for x in bits if x > 0], dtype=float))[::-1]
    if w.size == 0:
        return float(base)
    cum = np.cumsum(w)
    k_eff = cum * cum / np.maximum(np.cumsum(w * w), 1e-30)
    total = base * cum / (1.0 + (k_eff - 1.0) * rho)
    return float(np.maximum.accumulate(total)[-1] / cum[-1])


def tau_perquestion(bits, rho_bar, base=None):
    """The trap, implemented honestly so that it gets a fair hearing.

    The pooled design effect uses one rho for every pair of answers. The
    per-question generalisation replaces it with the mean pairwise correlation
    of the questions actually asked, rho_bar = mean sqrt(h_q h_q'), which is the
    right way to average a design effect over heterogeneous units and reduces to
    the pooled form when the subset is representative. Everything else -- Kish's
    effective count, the monotonicity floor -- is unchanged, so the only
    difference from the deployed path is that rho is now a property of the
    question set instead of a global constant.
    """
    return tau_pooled(bits, rho=rho_bar, base=base)


def stats(post, truth, lat, lon, t):
    """Per-person outcomes for a block of posteriors.

    Coverage is computed as the posterior mass strictly above the true cell
    rather than by sorting: credible_cells takes cells in descending order until
    the cumulative mass reaches the level, so the true cell is inside the set
    exactly when the mass above it has not yet reached the level. Same answer,
    no sort, and the sort is what makes the naive version too slow to run at
    this sample size.
    """
    best = post.argmax(1)
    err = haversine(lat[truth], lon[truth], t.cell_lat[best], t.cell_lon[best])
    same = t.state[best] == t.state[truth]
    p_true = post[np.arange(len(truth)), truth]
    above = np.where(post > p_true[:, None], post, 0.0).sum(1)
    cov = {lv: above < lv for lv in LEVELS}
    return err, same, np.log(np.clip(p_true, 1e-300, None)), cov


def summarise(parts, label, k, n):
    err = np.concatenate([p[0] for p in parts])
    same = np.concatenate([p[1] for p in parts])
    logs = np.concatenate([p[2] for p in parts])
    cov = {lv: float(np.concatenate([p[3][lv] for p in parts]).mean())
           for lv in LEVELS}
    return err, {
        "ordering": label, "k": k, "n": n,
        "median_km": float(np.median(err)),
        "mean_km": float(err.mean()),
        "p90_km": float(np.percentile(err, 90)),
        "within_150km": float((err <= 150).mean()),
        "state_acc": float(same.mean()),
        "logscore": float(logs.mean()),
        "calib_err": float(np.mean([abs(cov[lv] - lv) for lv in LEVELS])),
        **{f"cover{int(lv * 100)}": cov[lv] for lv in LEVELS},
    }


def run_ordering(g, LL, order, chosen, home, label, kmax, tau_fn=None,
                 chunk=250, verbose=True):
    """Score one ordering at every k from 1 to kmax. Returns per-person errors.

    The log-likelihood is accumulated incrementally down the ordering, so going
    from k to k+1 costs one gather and one add rather than a rebuild. tau is
    applied to the accumulated sum at each k, which is what infer.posterior
    does; `tau_fn(used) -> float` defaults to the deployed schedule and is the
    only thing the discount arms change.
    """
    t = g.t
    n = len(home)
    lat, lon = t.cell_lat, t.cell_lon
    bitmap = g.question_bits
    if tau_fn is None:
        def tau_fn(used):
            return tau_pooled([bitmap[u] for u in used])
    parts = {k: [] for k in range(1, kmax + 1)}

    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        L = np.zeros((j - i, t.n_cells), dtype=np.float32)
        for k, q in enumerate(order[:kmax], 1):
            L += LL[t.rows[q][chosen[q][i:j]]]
            tau = tau_fn(order[:k])
            lp = g.log_prior[None, :] + tau * L.astype(np.float64)
            lp -= lp.max(1, keepdims=True)
            post = np.exp(lp)
            post /= post.sum(1, keepdims=True)
            parts[k].append(stats(post, home[i:j], lat, lon, t))

    rows, errs = [], {}
    for k in range(1, kmax + 1):
        e, row = summarise(parts[k], label, k, n)
        errs[k] = e
        rows.append(row)
        if verbose:
            print(f"  {label:>7} k={k:2d}  {row['median_km']:6.0f} km  "
                  f"state {row['state_acc']:5.1%}  logS {row['logscore']:7.2f}")
    return rows, errs


def paired(base, alt, boots=2000, seed=12345):
    """Bootstrap CI for the difference in median error, resampling PEOPLE.

    Returns median(alt) - median(base), so a negative number means the
    alternative ordering placed people closer. The two arms answered on the same
    people, so the resample must keep a person in both arms or the pairing is
    thrown away and the interval is far too wide to see anything at this sample
    size.

    Identical arrays mean the two orderings selected the same SET of questions
    at this k -- a Bayesian posterior does not care what order a fixed set of
    answers arrived in -- so there is nothing to resample and the comparison is
    an exact tie.
    """
    same = bool(np.array_equal(base, alt))
    if same:
        return 0.0, 0.0, 0.0, 0.0, 1.0, True
    rng = np.random.default_rng(seed)
    d = float(np.median(alt) - np.median(base))
    n = len(base)
    idx = rng.integers(0, n, size=(boots, n))
    diffs = np.median(alt[idx], axis=1) - np.median(base[idx], axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (d, float(lo), float(hi), float((alt < base).mean()),
            float((alt == base).mean()), False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--personas", type=int, default=400)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--signal", default=None)
    ap.add_argument("--order-out", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-trap", action="store_true")
    args = ap.parse_args()

    g = Geolocator()
    sig = ss.load(Path(args.signal) if args.signal else ss.OUT)
    theta, rho_sim = ss.pool_theta()

    print(f"deriving orderings ({args.personas} personas, stride {args.stride})")
    orders = derive(g, sig, args.kmax, args.personas, args.stride,
                    out=Path(args.order_out) if args.order_out else ORDER_OUT)
    orders[BASELINE] = idio.deployed_questions(args.kmax)
    print(f"  {BASELINE:>7}: {', '.join(orders[BASELINE])}")

    questions = [str(q) for q in g.t.questions]
    print(f"\ndrawing {args.n} people: theta {theta:.3f} "
          f"(pooled rho {rho_sim:.3f}), {MOVER:.0%} movers, seed {args.seed}")
    home, _, _, chosen = idio.simulate(g, args.n, questions, theta,
                                       mover=MOVER, seed=args.seed)
    LL = contaminated(g)

    rows, errs = [], {}
    for name in [BASELINE, "ratio", "purity", "net"]:
        r, e = run_ordering(g, LL, orders[name], chosen, home, name,
                            args.kmax, verbose=False)
        rows += r
        errs[name] = e
        print(f"  {name:>8}  " + "  ".join(
            f"k={x['k']}:{x['median_km']:.0f}" for x in r
            if x["k"] in (5, 12, 14, args.kmax)))

    if not args.no_trap:
        bits = g.question_bits
        scale = ss.pair_scale(sig)
        print(f"\npair model: corr(q,q') = {scale:.3f} * sqrt(h_q h_q'); "
              f"mean over all 122 questions = {ss.pair_rho(sig, list(sig), scale):.3f}, "
              f"over the deployed first {args.kmax} = "
              f"{ss.pair_rho(sig, orders[BASELINE], scale):.3f}")
        traps = [
            ("mi+rho_q",
             lambda used: tau_perquestion([bits[u] for u in used],
                                          ss.pair_rho(sig, used, scale))),
            (f"mi+rho={LEGACY_RHO:g}",
             lambda used: tau_pooled([bits[u] for u in used],
                                     rho=LEGACY_RHO)),
        ]
        for name, fn in traps:
            r, e = run_ordering(g, LL, orders[BASELINE], chosen, home, name,
                                args.kmax, tau_fn=fn, verbose=False)
            rows += r
            errs[name] = e
            print(f"  {name:>8}  " + "  ".join(
                f"k={x['k']}:{x['median_km']:.0f}" for x in r
                if x["k"] in (5, 12, 14, args.kmax)))

    path = Path(args.out) if args.out else CURVE_OUT

    print(f"\npaired against {BASELINE}, n={args.n}, median km "
          f"(negative = the alternative places people closer)")
    arms = [a for a in errs if a != BASELINE]
    print(f"{'k':>3} " + "".join(f"{a:>36}" for a in arms))
    for row in rows:
        if row["ordering"] == BASELINE:
            row.update(d_vs_mi=0.0, d_lo=0.0, d_hi=0.0, win_rate=0.0,
                       tie_rate=1.0, same_set=1)
    for k in range(1, args.kmax + 1):
        cells = []
        for a in arms:
            d, lo, hi, win, tie, same = paired(errs[BASELINE][k], errs[a][k])
            r = next(x for x in rows if x["ordering"] == a and x["k"] == k)
            r.update(d_vs_mi=d, d_lo=lo, d_hi=hi, win_rate=win,
                     tie_rate=tie, same_set=int(same))
            cells.append("same question set" if same else
                         f"{d:+7.1f} [{lo:+6.1f},{hi:+6.1f}] w{win:4.0%}")
        print(f"{k:>3} " + "".join(f"{c:>36}" for c in cells))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
