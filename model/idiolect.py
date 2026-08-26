"""Simulate people who have an idiolect, and price the discount against them.

model/nullcheck.py builds respondents who obey the model exactly: each answer is
drawn independently given the true cell, so the within-person residual
correlation in that world is zero by construction. Sweeping --rho in
model/validate.py against those people therefore does not ask "what if people
are correlated". It asks "what does it cost to insure against a risk that is
provably absent", and the answer is always "more than nothing", monotonically.
That sweep brackets the price of the insurance. It cannot price the risk.

This file supplies the missing half. It gives each simulated person a latent
idiolect u ~ N(0,1) and tilts their answer distribution toward or away from
nationally marked variants in proportion to it:

    log P_i(a | cell) = log P(a | cell) + theta * u_i * z_a

where z_a is the markedness of choice a -- its national surprisal, centred and
scaled to unit variance within each question, so theta means the same thing
whether the question has two choices or nine. A person with u > 0 is broadly
regional: they say yinz and bubbler and hoagie and pop. A person with u < 0 is
broadly standard: soda, you guys, sub, water fountain. That axis is the one that
the design effect exists to discount, and crucially it is orthogonal to
geography -- the location term is untouched, so the model's surfaces remain
exactly correct on average and only the independence assumption is violated.

theta = 0 reproduces nullcheck exactly. Larger theta produces more dependence.
theta is not itself interpretable, so it is calibrated by bisection against the
realised correlation, measured with the same estimator that produced rho = 0.177
from YGDP: markedness scores per person per item, leave-one-out Gaussian kernel
residualisation on geography at 250 km, then the mean off-diagonal pairwise
correlation. Those functions are imported from model/ygdp_validation.py rather
than reimplemented, so the simulated number and the measured number are the same
quantity computed by the same code.

With both halves in hand the decision becomes a 2-D loss surface: the true
correlation on one axis, the deployed one on the other. The diagonal is the
oracle. Reading across a row gives the cost of deploying the wrong value; taking
the worst case down each column gives the minimax choice. That is the actual
question -- how many questions to ask, and what to discount them by -- and it
cannot be answered from either simulation alone.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import infer
import ygdp_validation as yv
from calibrate import LEVELS
from infer import Geolocator, credible_cells
from people import People
from tensor import haversine
from validate import region_of

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

SEED = 20131015
BW_KM = 250.0          # matches the YGDP spatial residualisation
CAL_N = 1500           # population size used when calibrating theta
KS = [5, 8, 12, 16, 20]


def deployed_questions(n=20, path=None):
    path = path or DATA / "model" / "question_order.csv"
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(r["question"])
            if len(out) >= n:
                break
    return out


def national(g, q):
    """Prior-weighted national probability of each choice of question q."""
    rows = g.t.rows[q]
    P = np.exp(np.asarray(g.t.logp[rows], dtype=np.float64))
    m = P @ g.prior
    s = m.sum()
    return m / s if s > 0 else np.full(len(m), 1.0 / len(m))


def markedness(g, q):
    """National surprisal per choice, centred and scaled to unit variance.

    Weighted by the national marginal, so a rare choice does not dominate the
    scale merely by being rare. Returns zeros for degenerate questions.
    """
    m = national(g, q)
    s = -np.log(np.clip(m, 1e-12, None))
    mu = float(m @ s)
    var = float(m @ (s - mu) ** 2)
    if var <= 1e-12:
        return np.zeros_like(s), s
    return (s - mu) / np.sqrt(var), s


def _xy(g):
    lat = np.asarray(g.t.cell_lat, dtype=np.float64)
    lon = np.asarray(g.t.cell_lon, dtype=np.float64)
    lat0 = lat.mean()
    return ((lon - lon.mean()) * 111.0 * np.cos(np.radians(lat0)),
            (lat - lat0) * 111.0)


def smooth_field(x, y, rng, n_modes=40, wavelength_km=500.0):
    """A smooth random scalar field over the grid, unit variance.

    Sum of random sinusoids with a characteristic wavelength, evaluated at each
    cell's projected position. Pixel-recovery error is not white: a dot map read
    slightly wrong is read slightly wrong over a whole neighbourhood, because
    the error comes from blur, overlap and colour bleed that vary smoothly in
    space. A spatially correlated field is therefore the right shape of noise,
    and white noise would be the wrong one -- it would average out over the many
    cells a posterior touches and understate the damage badly.
    """
    out = np.zeros(len(x))
    k = 2 * np.pi / wavelength_km
    for _ in range(n_modes):
        ang = rng.uniform(0, 2 * np.pi)
        scale = np.exp(rng.normal(0, 0.5))          # spread of wavelengths
        fx, fy = k * scale * np.cos(ang), k * scale * np.sin(ang)
        out += np.cos(fx * x + fy * y + rng.uniform(0, 2 * np.pi))
    return out / np.sqrt(n_modes / 2.0)


def make_fields(g, questions, seed=SEED, wavelength_km=500.0):
    """One unit-variance field per choice, drawn once and reused at every
    amplitude so the bisection scales a fixed perturbation rather than
    redrawing a different one at each step."""
    rng = np.random.default_rng(seed)
    x, y = _xy(g)
    return {q: np.stack([smooth_field(x, y, rng, wavelength_km=wavelength_km)
                         for _ in range(len(g.t.rows[q]))])
            for q in questions}


def perturb(g, questions, amp, fields):
    """Surfaces the people are drawn from, wrong by a smooth amount.

    Returns {question: log-probability matrix}. amp=0 returns the model's own
    surfaces, in which case the model is perfectly specified.
    """
    out = {}
    for q in questions:
        rows = g.t.rows[q]
        lp = np.asarray(g.t.logp[rows], dtype=np.float64)
        if amp > 0:
            lp = lp + amp * fields[q]
            lp = lp - lp.max(axis=0, keepdims=True)
            lp = lp - np.log(np.exp(lp).sum(axis=0, keepdims=True))
        out[q] = lp
    return out


def surface_mae(g, questions, surf):
    """Mean absolute change in choice probability, prior-weighted, in points.

    Halved so it reads as total variation distance: the share of respondents in
    a cell whose answer distribution has moved, not double-counted across the
    choice they left and the choice they arrived at.
    """
    tot = 0.0
    for q in questions:
        rows = g.t.rows[q]
        a = np.exp(np.asarray(g.t.logp[rows], dtype=np.float64))
        a /= a.sum(axis=0, keepdims=True)
        b = np.exp(surf[q])
        b /= b.sum(axis=0, keepdims=True)
        tot += float((np.abs(a - b).sum(axis=0) * g.prior).sum())
    return 100.0 * tot / len(questions) / 2.0


def calibrate_amp(g, questions, target_mae, fields, lo=0.0, hi=6.0, iters=20):
    """Bisect the perturbation amplitude to a target surface error in points."""
    if target_mae <= 0:
        return 0.0, 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = surface_mae(g, questions, perturb(g, questions, mid, fields))
        if abs(got - target_mae) < 0.1:
            return mid, got
        if got < target_mae:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, surface_mae(g, questions, perturb(g, questions, mid, fields))


def simulate(g, n, questions, theta, mover=0.0, seed=SEED, surf=None):
    """Draw n people. Returns home cells, speech cells, u, choices.

    `theta` is the idiolect strength: a diffuse nuisance that makes a person
    consistently marked or consistently standard across all questions.

    `mover` is the fraction whose speech comes from an independently drawn cell
    rather than the one they are recorded at -- raised in one place, living in
    another. That is a concentrated nuisance: every answer agrees, and they all
    agree on the wrong place, so the posterior is narrow and wrong. The two
    mechanisms damage a Bayesian update in opposite ways and there is no reason
    a single scalar discount should handle both, which is part of what this
    measures.

    `surf` supplies the surfaces people are actually drawn from. Passing a
    perturbed set makes the model's own surfaces wrong, which is the error
    TAU_BASE exists to absorb and which no other simulation in this project
    contains.
    """
    rng = np.random.default_rng(seed)
    t = g.t
    home = rng.choice(t.n_cells, size=n, p=g.prior)
    speech = home.copy()
    if mover > 0:
        moved = rng.random(n) < mover
        speech[moved] = rng.choice(t.n_cells, size=int(moved.sum()), p=g.prior)
    u = rng.standard_normal(n)
    chosen = {}
    for q in questions:
        rows = t.rows[q]
        src = surf[q] if surf is not None else np.asarray(t.logp[rows],
                                                          dtype=np.float64)
        base = src[:, speech]
        z, _ = markedness(g, q)
        lp = base + theta * z[:, None] * u[None, :]
        lp -= lp.max(axis=0, keepdims=True)
        P = np.exp(lp)
        P /= P.sum(axis=0, keepdims=True)
        cum = np.cumsum(P, axis=0)
        r = rng.random(n)
        idx = (cum < r[None, :]).sum(axis=0)
        chosen[q] = np.clip(idx, 0, len(rows) - 1)
    return home, speech, u, chosen


def measure_rho(g, questions, chosen, lat, lon, bw_km=BW_KM, min_pairs=40):
    """The YGDP estimator, applied to simulated categorical answers.

    Each person-question cell becomes the national surprisal of the answer they
    gave -- the categorical analogue of a nonstandardness rating. Geography is
    then removed by the same kernel smoother at the same bandwidth, and the mean
    off-diagonal correlation of what remains is the within-person dependence.
    """
    n = len(lat)
    P = np.full((n, len(questions)), np.nan)
    for j, q in enumerate(questions):
        _, surp = markedness(g, q)
        P[:, j] = surp[chosen[q]]
    R = yv.spatial_residualize(P, np.asarray(lat), np.asarray(lon), bw_km=bw_km)
    C, _ = yv.pairwise_corr(R, min_pairs=min_pairs)
    return yv.mean_offdiag(C)


def rho_at(g, questions, theta, n=CAL_N, seed=SEED):
    home, _, _, chosen = simulate(g, n, questions, theta, mover=0.0, seed=seed)
    lat = g.t.cell_lat[home]
    lon = g.t.cell_lon[home]
    rho, _ = measure_rho(g, questions, chosen, lat, lon)
    return float(rho)


def calibrate_theta(g, questions, target, lo=0.0, hi=3.0, tol=0.004, iters=18):
    """Bisect theta so the realised correlation matches `target`."""
    if target <= 0:
        return 0.0, 0.0
    r_hi = rho_at(g, questions, hi)
    while r_hi < target and hi < 12.0:
        lo, hi = hi, hi * 2
        r_hi = rho_at(g, questions, hi)
    if r_hi < target:
        return hi, r_hi
    best = (hi, r_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = rho_at(g, questions, mid)
        if abs(r - target) < abs(best[1] - target):
            best = (mid, r)
        if abs(r - target) < tol:
            return mid, r
        if r < target:
            lo = mid
        else:
            hi = mid
    return best


def score(g, people, order, k):
    """Accuracy, calibration and log score after the first k questions.

    Median error and state accuracy say how much was extracted; coverage says
    whether the stated confidence is honest; the log score at the true cell is
    the proper scoring rule that trades the two against each other, and is the
    only one of the three a discount can legitimately improve. Tempering always
    widens the posterior, so it can only hurt the point estimate; the question
    is whether the honesty it buys is worth more than the sharpness it spends,
    and that is what the log score measures. Choosing rho on median error alone
    would answer "deploy zero" in every world, including the ones where the
    model is badly overconfident, which is why that comparison is not the test.
    """
    allowed = set(order[:k])
    err, state_ok, region_ok, logs = [], [], [], []
    hits = {lv: [] for lv in LEVELS}
    for row, answers in people:
        subset = [(q, c) for q, c in answers if q in allowed]
        post = g.posterior(subset, tau="auto", eps=None)
        best = int(np.argmax(post))
        cell = g.t.nearest(row["lat"], row["lon"])
        err.append(haversine(row["lat"], row["lon"],
                             g.t.cell_lat[best], g.t.cell_lon[best]))
        truth_state = str(g.t.state[cell])
        state_ok.append(str(g.t.state[best]) == truth_state)
        region_ok.append(region_of(g.t.state[best]) == region_of(truth_state)
                         and region_of(truth_state) != "")
        logs.append(float(np.log(max(float(post[cell]), 1e-300))))
        for lv in LEVELS:
            hits[lv].append(cell in set(credible_cells(post, lv).tolist()))
    err = np.array(err)
    cov = {lv: float(np.mean(hits[lv])) for lv in LEVELS}
    return {
        "k": k,
        "median_km": float(np.median(err)),
        "p90_km": float(np.percentile(err, 90)),
        "within_150km": float((err <= 150).mean()),
        "state_acc": float(np.mean(state_ok)),
        "region_acc": float(np.mean(region_ok)),
        "logscore": float(np.mean(logs)),
        "calib_err": float(np.mean([abs(cov[lv] - lv) for lv in LEVELS])),
        **{f"cover{int(lv * 100)}": cov[lv] for lv in LEVELS},
    }


def write_population(g, name, cells, chosen, questions):
    root = DATA / name
    root.mkdir(parents=True, exist_ok=True)
    t = g.t
    with open(root / "people.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person", "lat", "lon", "state", "truth_cell"])
        for i, c in enumerate(cells):
            w.writerow([f"i{i:05d}", f"{t.cell_lat[c]:.5f}",
                        f"{t.cell_lon[c]:.5f}", t.state[c], int(c)])
    with open(root / "answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person", "question", "choice"])
        for i in range(len(cells)):
            for q in questions:
                w.writerow([f"i{i:05d}", q, t.choice[t.rows[q][chosen[q][i]]]])
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--questions", type=int, default=20)
    ap.add_argument("--true-rho", default="0,0.05,0.10,0.18,0.30",
                    help="within-person correlations to simulate")
    ap.add_argument("--deployed-rho", default="0,0.05,0.09,0.13,0.177",
                    help="values of RHO the model is run with")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--mover", type=float, default=0.0,
                    help="fraction raised somewhere other than where they are "
                         "recorded; their answers are narrow and wrong")
    ap.add_argument("--surface-mae", type=float, default=0.0,
                    help="make the model's surfaces wrong by this many points "
                         "of mean absolute choice probability, smoothly in "
                         "space; this is the error TAU_BASE exists to absorb")
    ap.add_argument("--base", default=None,
                    help="comma-separated TAU_BASE values to try instead of the "
                         "fitted one")
    ap.add_argument("--out", default="idiolect")
    ap.add_argument("--keep", action="store_true",
                    help="keep the simulated populations on disk")
    args = ap.parse_args()

    targets = [float(x) for x in args.true_rho.split(",") if x != ""]
    deployed = [float(x) for x in args.deployed_rho.split(",") if x != ""]
    ks = [int(x) for x in args.ks.split(",") if x != ""]
    bases = ([float(x) for x in args.base.split(",") if x != ""]
             if args.base else [infer.TAU_BASE])

    g = Geolocator()
    questions = deployed_questions(args.questions)
    print(f"{len(questions)} questions: {', '.join(questions)}")
    print(f"n={args.n}, movers={args.mover:.0%}, "
          f"tau_base={','.join(f'{b:g}' for b in bases)}")

    surf, mae = None, 0.0
    if args.surface_mae > 0:
        fields = make_fields(g, questions)
        amp, mae = calibrate_amp(g, questions, args.surface_mae, fields)
        surf = perturb(g, questions, amp, fields)
        print(f"surfaces perturbed: amplitude {amp:.3f} nats -> "
              f"{mae:.2f} points mean absolute error")
    print()

    print("calibrating idiolect strength against the YGDP estimator")
    print(f"{'target rho':>10} {'theta':>8} {'realised':>9}")
    thetas = []
    for tgt in targets:
        th, got = calibrate_theta(g, questions, tgt)
        thetas.append((tgt, th, got))
        print(f"{tgt:>10.3f} {th:>8.3f} {got:>9.3f}")
    print()

    rows = []
    for tgt, th, _ in thetas:
        home, speech, _, chosen = simulate(g, args.n, questions, th,
                                           mover=args.mover, seed=SEED + 1,
                                           surf=surf)
        lat, lon = g.t.cell_lat[home], g.t.cell_lon[home]
        rho_pop, npairs = measure_rho(g, questions, chosen, lat, lon)
        moved = int((home != speech).sum())
        name = f"{args.out}_true{tgt:g}"
        root = write_population(g, name, home, chosen, questions)
        people = People(root)
        print(f"--- true rho {tgt:g}  (theta {th:.3f}, realised {rho_pop:.3f} "
              f"on {npairs} pairs, {moved} movers, surface MAE {mae:.1f}pp) ---")
        print(f"{'base':>5} {'deployed':>9}   " +
              "  ".join(f"{('k=' + str(k)):>21}" for k in ks))
        print(f"{'':>5} {'':>9}   " +
              "  ".join(f"{'km/state/logS/cal':>21}" for _ in ks))
        for base in bases:
            infer.TAU_BASE = base
            for dep in deployed:
                infer.RHO = dep
                txt = []
                for k in ks:
                    r = score(g, people, questions, k)
                    rows.append({"true_rho": tgt, "realised_rho": rho_pop,
                                 "theta": th, "movers": moved,
                                 "surface_mae": mae, "tau_base": base,
                                 "deployed_rho": dep, **r})
                    txt.append(f"{r['median_km']:>5.0f}/{r['state_acc']:>4.0%}/"
                               f"{r['logscore']:>6.2f}/{r['calib_err']:>4.2f}")
                print(f"{base:>5.2f} {dep:>9.3f}   " +
                      "  ".join(f"{c:>21}" for c in txt))
        print()
        if not args.keep:
            for p in root.iterdir():
                p.unlink()
            root.rmdir()

    out = DATA / "model" / f"{args.out}_surface.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
