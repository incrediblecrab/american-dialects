"""Does the network recognise a mover and widen, or does it just concentrate?

Part of the case for building model/neural.py was that a factorised likelihood
multiplies independent evidence and therefore *concentrates* as answers arrive,
whereas a network is handed the whole answer vector at once and could in
principle recognise a person whose speech matches no single place -- raised in
one region, recorded in another -- and respond by widening rather than settling
confidently on the wrong one. findings.md states that as a capability the
network has "in principle". It was never measured. This file measures it.

The claim decomposes into three questions, all of which need movers and
non-movers held to the same number of questions:

  1. Error.       Does the network degrade less than Bayes on movers?
  2. Width.       Does it widen on movers *specifically*? A model that is
                  uniformly vaguer is not doing anything clever, so the
                  quantity is the ratio of mover width to non-mover width,
                  not the absolute width.
  3. Calibration. Does the stated 80% region contain the true home 80% of the
                  time, separately for each group? This is what separates a
                  confidently wrong model from a usefully uncertain one, and it
                  is invisible in a pooled number.

The network does emit a distribution -- a softmax over 1024 population-weighted
centroids, splatted back over all 50888 cells in proportion to the prior -- so
"widens rather than concentrates" is a property it is capable of having. That
was checked before the experiment was designed, because if the output were a
point estimate the question would already be answered.

WHAT THE SIMULATOR ACTUALLY CONTAINS. model/idiolect.py draws a mover's speech
cell from `g.prior`, which is the same distribution `home` is drawn from, and
draws the idiolect u independently of mover status. So the law of `speech` is
the prior whether or not the person moved, and the answers depend on the person
only through `speech` and `u`. The answer vector is therefore statistically
independent of mover status, and

    P(moved | answers) = MOVER   for every possible answer vector.

The Bayes-optimal posterior over home follows in one line. Writing Q(c) =
prior(c) P(A|c) and Z = sum_c Q(c), and using P(A | moved) = sum_s prior(s)
P(A|s) = Z = P(A | stayed),

    P(home=c | A)  =  [(1-m) Q(c) + m prior(c) Z] / Z
                   =  (1-m) posterior(c) + m prior(c)

-- a mixture with a weight that does not depend on the answers. The correct
response to movers, in this world, is to widen *every* posterior by the same
fixed admixture of prior. Selective widening is not merely hard here, it is
wrong, because there is nothing to select on.

That is a derivation, not a measurement, and this file does not ask the reader
to take it on trust. `probe` trains a classifier on the same encoding the
network sees and reports how well it predicts mover status, against a
label-shuffled negative control and a positive control (predicting the sign of
the latent idiolect, which the answers really do carry). `diagnose` measures
what the models actually do. The derivation predicts the measurements; the
measurements are what is reported.

The mixture is also cheap to test as a deployable change, so it is an arm.

Usage:
    python mover.py diagnose      error, width and coverage, movers vs stayers
    python mover.py probe         is mover status predictable from answers?
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

import idiolect as idio
import infer
import neural as nn_mod
from infer import Geolocator
from tensor import haversine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

OUT = DATA / "model"
SEED = 424242
"""Held-out. Distinct from neural.py's training seed and from its eval seed."""

LEVELS = (0.5, 0.8, 0.95)
KS = [1, 5, 12, 14, 20, 30]
BOOT = 2000


def bins_for(dist_km):
    """Displacement bands, so "mover" is not treated as one homogeneous thing.

    Somebody who moved 80 km is barely distinguishable from a non-mover -- the
    surfaces vary over hundreds of kilometres -- and lumping them in with a
    coast-to-coast move drags every mover statistic toward the non-mover one.
    The bands are fixed rather than quantiles so they mean the same thing at
    every k and for every model.
    """
    edges = [(0, 500), (500, 1500), (1500, 10_000)]
    names = ["mover_0_500km", "mover_500_1500km", "mover_1500km_plus"]
    return [(nm, (dist_km >= lo) & (dist_km < hi))
            for nm, (lo, hi) in zip(names, edges)]


def stats_block(post, home, cell_lat, cell_lon, cell_km2, chunk=200):
    """Per-person error, credible-region size and coverage, from one sort each.

    Everything here is derived from a single descending sort of the posterior,
    because infer.credible_cells and infer.credible_area each re-sort and this
    needs six of them per person. The definitions are copied exactly, not
    approximated: a cell is inside the level-L region iff the mass strictly
    ahead of it in descending order is below L, which is what
    `order[:searchsorted(cum, L) + 1]` selects.
    """
    n, ncell = post.shape
    out = {
        "err_km": np.empty(n),
        "logscore": np.empty(n),
        "entropy_bits": np.empty(n),
        **{f"area{int(lv * 100)}": np.empty(n) for lv in LEVELS},
        **{f"cover{int(lv * 100)}": np.empty(n, dtype=bool) for lv in LEVELS},
    }
    idx = np.arange(ncell)
    for i in range(0, n, chunk):
        P = post[i:i + chunk]
        m = P.shape[0]
        rowid = np.arange(m)
        order = np.argsort(-P, axis=1, kind="stable")
        ps = np.take_along_axis(P, order, 1)
        cum = np.cumsum(ps, axis=1)
        acum = np.cumsum(cell_km2[order], axis=1)

        h = home[i:i + chunk]
        best = order[:, 0]
        out["err_km"][i:i + m] = haversine(
            cell_lat[h], cell_lon[h], cell_lat[best], cell_lon[best])
        p_true = P[rowid, h]
        out["logscore"][i:i + m] = np.log(np.clip(p_true, 1e-300, None))
        q = np.clip(P, 1e-300, None)
        out["entropy_bits"][i:i + m] = -(P * np.log2(q)).sum(1)

        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.broadcast_to(idx, order.shape), 1)
        r_true = rank[rowid, h]
        ahead = cum[rowid, r_true] - ps[rowid, r_true]
        for lv in LEVELS:
            tag = int(lv * 100)
            i0 = (cum < lv).sum(1)
            out[f"area{tag}"][i:i + m] = acum[rowid, i0]
            out[f"cover{tag}"][i:i + m] = ahead < lv
    return out


def summarise(st, mask, model, k, group):
    n = int(mask.sum())
    row = {"model": model, "k": k, "group": group, "n": n}
    if n == 0:
        return None
    e = st["err_km"][mask]
    row["median_km"] = float(np.median(e))
    row["p90_km"] = float(np.percentile(e, 90))
    row["within_150km"] = float((e <= 150).mean())
    row["logscore"] = float(st["logscore"][mask].mean())
    row["entropy_bits"] = float(np.median(st["entropy_bits"][mask]))
    for lv in LEVELS:
        tag = int(lv * 100)
        row[f"area{tag}_km2"] = float(np.median(st[f"area{tag}"][mask]))
        row[f"cover{tag}"] = float(st[f"cover{tag}"][mask].mean())
    return row


def boot_ratio(a, b, stat=np.median, reps=BOOT, seed=7):
    """Bootstrap CI for stat(a)/stat(b), resampling each group independently.

    Movers and stayers are disjoint samples of different sizes, so the ratio's
    uncertainty is dominated by whichever is smaller; resampling both is the
    only way that shows up in the interval.
    """
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    ra = stat(a[rng.integers(0, len(a), (reps, len(a)))], axis=1)
    rb = stat(b[rng.integers(0, len(b), (reps, len(b)))], axis=1)
    r = ra / np.clip(rb, 1e-30, None)
    return (float(stat(a) / stat(b)),
            float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def wilson(k, n, z=1.96):
    """Wilson interval, which behaves at the coverage rates movers produce.

    A normal-approximation interval on a proportion near 0.1 with a few hundred
    trials can reach below zero, and mover coverage lands exactly there.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


# ------------------------------------------------------------------ the models

def load_net(dev):
    import torch

    ck = torch.load(OUT / "neural_net.pt", map_location=dev, weights_only=False)
    net = nn_mod.build_net(ck["n_in"], ck["n_out"], ck["width"],
                           ck["depth"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def net_posteriors(net, A_t, mask, tab, n_opt, lab, w_cell, dev, chunk=1024):
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for i in range(0, A_t.shape[0], chunk):
            x = nn_mod.encode(A_t[i:i + chunk], mask[i:i + chunk], tab, n_opt)
            out.append(F.softmax(net(x), 1).cpu().numpy())
    pc = np.concatenate(out).astype(np.float64)
    pc /= pc.sum(1, keepdims=True)
    return pc[:, lab] * w_cell[None, :]


def bayes_posteriors(g, A, qpos, use, rho, n):
    t = g.t
    keep = infer.RHO
    infer.RHO = rho
    try:
        P = np.empty((n, t.n_cells))
        for i in range(n):
            ans = [(q, t.choice[t.rows[q][A[i, qpos[q]]]]) for q in use]
            P[i] = g.posterior(ans)
    finally:
        infer.RHO = keep
    return P


def to_cluster(P, lab, w_cell, n_clusters):
    """Push a fine-grid posterior through the network's 1024-class bottleneck.

    Without this arm, any width difference between the network and Bayes is
    confounded: the network's output is piecewise-constant within a cluster by
    construction, which changes credible-region areas whatever it has learned.
    This is Bayes wearing the network's representation, so the gap between the
    two is what the network actually learned and nothing else.
    """
    pc = np.zeros((P.shape[0], n_clusters))
    np.add.at(pc.T, lab, P.T)
    pc /= pc.sum(1, keepdims=True)
    return pc[:, lab] * w_cell[None, :]


def prior_stats(prior, home, cell_lat, cell_lon, cell_km2):
    """The same statistics for a model that ignores the answers entirely.

    This is the floor, and it is not a rhetorical one. A mover's answers carry
    information about where they learned to speak and none about where they are
    recorded, so the best any model can do for them is the population prior.
    Any model whose mover error is *worse* than this is being actively misled
    by the answers rather than merely failing to profit from them, and that is
    a distinction worth being able to state numerically.

    Computed in closed form because the posterior is identical for every
    person, so materialising it as (n, 50888) would be 2.4 GB of one repeated
    row.
    """
    order = np.argsort(-prior, kind="stable")
    ps = prior[order]
    cum = np.cumsum(ps)
    acum = np.cumsum(cell_km2[order])
    rank = np.empty(len(prior), dtype=np.int64)
    rank[order] = np.arange(len(prior))
    r_true = rank[home]
    ahead = cum[r_true] - ps[r_true]
    best = order[0]
    out = {
        "err_km": haversine(cell_lat[home], cell_lon[home],
                            cell_lat[best], cell_lon[best]),
        "logscore": np.log(np.clip(prior[home], 1e-300, None)),
        "entropy_bits": np.full(len(home),
                                -(prior * np.log2(np.clip(prior, 1e-300,
                                                          None))).sum()),
    }
    for lv in LEVELS:
        tag = int(lv * 100)
        i0 = int((cum < lv).sum())
        out[f"area{tag}"] = np.full(len(home), acum[i0])
        out[f"cover{tag}"] = ahead < lv
    return out


# -------------------------------------------------------------------- diagnose

def diagnose(args):
    import torch

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = np.load(OUT / "neural_pool.npz", allow_pickle=True)
    cent, lab = d["centroids"], d["labels"]
    questions = [str(q) for q in d["questions"]]
    theta = float(d["theta"])
    n_clusters = cent.shape[0]

    net, ck = load_net(dev)
    g = Geolocator()
    t = g.t
    tab = torch.tensor(nn_mod.option_table(g, questions), device=dev)
    n_opt = t.logp.shape[0]
    csum = np.zeros(n_clusters)
    np.add.at(csum, lab, g.prior)
    w_cell = g.prior / np.clip(csum[lab], 1e-30, None)
    cell_km2 = g.cell_km2

    order = idio.deployed_questions(max(args.ks))
    qpos = {q: j for j, q in enumerate(questions)}

    n = args.n
    print(f"net from epoch {ck.get('epoch', '?')} (val {ck.get('val', 0):.4f}), "
          f"device {dev}")
    print(f"{n} held-out people, seed {args.seed}, "
          f"mover rate {nn_mod.MOVER:.0%}, theta {theta:.3f}")
    A, home, speech = nn_mod.make_pool(g, n, questions, theta, args.seed)
    moved = home != speech
    disp = np.where(moved, haversine(t.cell_lat[home], t.cell_lon[home],
                                     t.cell_lat[speech], t.cell_lon[speech]),
                    0.0)
    print(f"movers {moved.sum()} ({moved.mean():.1%}), "
          f"median displacement {np.median(disp[moved]):.0f} km, "
          f"p10 {np.percentile(disp[moved], 10):.0f} km")

    groups = [("all", np.ones(n, dtype=bool)), ("stayer", ~moved),
              ("mover", moved)] + bins_for(np.where(moved, disp, -1.0))
    A_t = torch.tensor(A, device=dev)

    rows, ratios = [], []

    def record(name, k, st):
        for gname, gmask in groups:
            r = summarise(st, gmask, name, k, gname)
            if r:
                rows.append(r)
        for lv in LEVELS:
            tag = int(lv * 100)
            for gname, gmask in [("stayer", ~moved), ("mover", moved)]:
                lo, hi = wilson(int(st[f"cover{tag}"][gmask].sum()),
                                int(gmask.sum()))
                for rr in rows:
                    if (rr["model"] == name and rr["k"] == k
                            and rr["group"] == gname):
                        rr[f"cover{tag}_lo"] = lo
                        rr[f"cover{tag}_hi"] = hi
        for field, label in [("area80", "width80"), ("err_km", "error"),
                             ("entropy_bits", "entropy")]:
            pt, lo, hi = boot_ratio(st[field][moved], st[field][~moved])
            row = {"model": name, "k": k, "quantity": label,
                   "ratio_mover_over_stayer": pt, "lo95": lo, "hi95": hi,
                   "n_mover": int(moved.sum()),
                   "n_stayer": int((~moved).sum()),
                   "auc": "", "auc_lo95": "", "auc_hi95": ""}
            if label != "error" and np.ptp(st[field]) > 0:
                # Can a mover be spotted from how wide the model went? The
                # ratio of medians is one summary of that and a blunt one;
                # this is the whole distribution, and 0.5 means the width says
                # nothing at all.
                a, alo, ahi = auc_ci(moved.astype(float), st[field])
                row.update(auc=a, auc_lo95=alo, auc_hi95=ahi)
            ratios.append(row)

    record("prior", 0, prior_stats(g.prior, home, t.cell_lat, t.cell_lon,
                                   cell_km2))

    for k in args.ks:
        use = order[:k]
        mask = np.zeros((n, len(questions)), dtype=np.float32)
        for q in use:
            mask[:, qpos[q]] = 1.0
        mt = torch.tensor(mask, device=dev)

        t0 = time.time()
        Pn = net_posteriors(net, A_t, mt, tab, n_opt, lab, w_cell, dev)
        arms = {"net": Pn,
                "net+mix": (1.0 - nn_mod.MOVER) * Pn
                + nn_mod.MOVER * g.prior[None, :]}
        P0 = bayes_posteriors(g, A, qpos, use, 0.0, n)
        arms["bayes(rho=0)"] = P0
        arms["bayes(rho=0)@cluster"] = to_cluster(P0, lab, w_cell, n_clusters)
        arms["bayes(rho=0)+mix"] = ((1.0 - nn_mod.MOVER) * P0
                                    + nn_mod.MOVER * g.prior[None, :])
        if args.legacy:
            arms[f"bayes(rho={nn_mod.LEGACY_RHO:g})"] = bayes_posteriors(
                g, A, qpos, use, nn_mod.LEGACY_RHO, n)

        for name, P in arms.items():
            record(name, k, stats_block(P, home, t.cell_lat, t.cell_lon,
                                        cell_km2))
        del P0, Pn, arms

        for name in ["net", "bayes(rho=0)"]:
            s = next(r for r in rows if r["model"] == name and r["k"] == k
                     and r["group"] == "stayer")
            m = next(r for r in rows if r["model"] == name and r["k"] == k
                     and r["group"] == "mover")
            w = next(x for x in ratios if x["model"] == name and x["k"] == k
                     and x["quantity"] == "width80")
            print(f"  k={k:2d} {name:14s} stayer {s['median_km']:5.0f} km "
                  f"cov80 {s['cover80']:.3f} | mover {m['median_km']:5.0f} km "
                  f"cov80 {m['cover80']:.3f} | width x{w['ratio_mover_over_stayer']:.2f} "
                  f"[{w['lo95']:.2f},{w['hi95']:.2f}]")
        print(f"     ({time.time() - t0:.0f}s)")

    keys = ["model", "k", "group", "n", "median_km", "p90_km", "within_150km",
            "logscore", "entropy_bits"]
    for lv in LEVELS:
        tag = int(lv * 100)
        keys += [f"area{tag}_km2", f"cover{tag}", f"cover{tag}_lo",
                 f"cover{tag}_hi"]
    path = OUT / f"mover_split{args.tag}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({kk: r.get(kk, "") for kk in keys})
    print(f"wrote {path} ({len(rows)} rows)")

    path = OUT / f"mover_ratio{args.tag}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ratios[0].keys()))
        w.writeheader()
        w.writerows(ratios)
    print(f"wrote {path} ({len(ratios)} rows)")
    report(rows, ratios)


def report(rows, ratios):
    ref = {r["group"]: r for r in rows if r["model"] == "prior"}
    print("\n--- the floor: a model that ignores the answers (the prior) ---")
    for gname in ("stayer", "mover", "all"):
        r = ref.get(gname)
        if r:
            print(f"  {gname:8s} n={r['n']:5d}  median {r['median_km']:5.0f} km"
                  f"  area80 {r['area80_km2']/1e6:6.3f} Mkm2"
                  f"  cover80 {r['cover80']:.3f}")

    print("\n--- does anything widen on movers specifically? "
          "(median 80% area, mover / stayer) ---")
    models = sorted({r["model"] for r in ratios if r["k"] > 0})
    ks = sorted({r["k"] for r in ratios if r["k"] > 0})
    print(f"{'model':24s}" + "".join(f"{('k=' + str(k)):>18}" for k in ks))
    for m in models:
        cells = []
        for k in ks:
            r = next((x for x in ratios if x["model"] == m and x["k"] == k
                      and x["quantity"] == "width80"), None)
            cells.append(f"{r['ratio_mover_over_stayer']:.2f} "
                         f"[{r['lo95']:.2f},{r['hi95']:.2f}]" if r else "-")
        print(f"{m:24s}" + "".join(f"{c:>18}" for c in cells))

    print("\n--- median km error, mover / stayer ---")
    for m in models:
        cells = []
        for k in ks:
            s = next((r for r in rows if r["model"] == m and r["k"] == k
                      and r["group"] == "stayer"), None)
            v = next((r for r in rows if r["model"] == m and r["k"] == k
                      and r["group"] == "mover"), None)
            cells.append(f"{s['median_km']:.0f}/{v['median_km']:.0f}"
                         if s else "-")
        print(f"{m:24s}" + "".join(f"{c:>18}" for c in cells))

    print("\n--- 80% coverage of the true home, by group ---")
    for m in models:
        line = []
        for k in ks:
            s = next((r for r in rows if r["model"] == m and r["k"] == k
                      and r["group"] == "stayer"), None)
            v = next((r for r in rows if r["model"] == m and r["k"] == k
                      and r["group"] == "mover"), None)
            a = next((r for r in rows if r["model"] == m and r["k"] == k
                      and r["group"] == "all"), None)
            line.append(f"{s['cover80']:.2f}/{v['cover80']:.2f}/{a['cover80']:.2f}"
                        if s else "-")
        print(f"{m:24s}" + "".join(f"{c:>18}" for c in line)
              + "   (stayer/mover/pooled)")

    print("\n--- can a mover be identified from the width alone? "
          "(AUC of the 80% area) ---")
    for m in models:
        cells = []
        for k in ks:
            r = next((x for x in ratios if x["model"] == m and x["k"] == k
                      and x["quantity"] == "width80"), None)
            cells.append(f"{r['auc']:.3f} [{r['auc_lo95']:.3f},"
                         f"{r['auc_hi95']:.3f}]" if r and r["auc"] != "" else "-")
        print(f"{m:24s}" + "".join(f"{c:>22}" for c in cells))


# ----------------------------------------------------------------------- probe

def answer_independence(g, questions, chosen, moved, order):
    """A model-free version of the same question, with no learning involved.

    If mover status is independent of the answers then, question by question,
    movers and stayers draw from the same choice distribution. A chi-square
    test per question checks that directly. It is weaker than the probe -- it
    cannot see joint structure across questions, which is exactly where a
    "these answers do not agree with each other" signature would live -- but it
    is free of any possibility that a network simply failed to train, so the
    two failures would have to be independent to both be false negatives.
    """
    from scipy.stats import chi2_contingency

    out = []
    for q in order:
        c = chosen[q]
        nlev = int(c.max()) + 1
        tabl = np.zeros((2, nlev), dtype=np.int64)
        for lev in range(nlev):
            tabl[0, lev] = int(((c == lev) & ~moved).sum())
            tabl[1, lev] = int(((c == lev) & moved).sum())
        keep = tabl.sum(0) > 0
        tabl = tabl[:, keep]
        if tabl.shape[1] < 2:
            continue
        _, p, _, _ = chi2_contingency(tabl)
        out.append((q, float(p)))
    return out


def probe(args):
    """Can mover status be predicted from the answers at all?

    Three tasks on identical inputs. `moved` is the question. `moved_shuffled`
    is a negative control -- the same labels, permuted, so whatever AUC that
    reaches is what this training setup scores on pure noise. `idiolect_sign`
    is a positive control: the sign of the latent u, which really is written
    into the answers, so if the probe cannot find that either then a null on
    `moved` would mean nothing.

    The probe is given the same 802-bit encoding the geolocation network sees,
    so a null here is a statement about that input and not about some reduced
    view of it. Inputs are encoded per batch rather than materialised, because
    the whole matrix at this sample size is most of a gigabyte and there is no
    reason to hold it.
    """
    import torch
    import torch.nn.functional as F

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = np.load(OUT / "neural_pool.npz", allow_pickle=True)
    questions = [str(q) for q in d["questions"]]
    theta = float(d["theta"])
    g = Geolocator()
    tab = torch.tensor(nn_mod.option_table(g, questions), device=dev)
    n_opt = g.t.logp.shape[0]
    qpos = {q: j for j, q in enumerate(questions)}
    order = idio.deployed_questions(max(args.ks))

    n = args.n
    print(f"{n} people ({args.holdout} held out), device {dev}")
    rng = np.random.default_rng(args.seed + 5)
    home, speech, u, chosen = idio.simulate(
        g, n, questions, theta, mover=nn_mod.MOVER, seed=args.seed + 5)
    A = np.zeros((n, len(questions)), dtype=np.int16)
    for j, q in enumerate(questions):
        A[:, j] = chosen[q]
    moved = home != speech
    print(f"movers {moved.sum()} ({moved.mean():.1%})")

    ps = answer_independence(g, questions, chosen, moved, order)
    below = sum(1 for _, p in ps if p < 0.05)
    print(f"\nper-question chi-square, movers vs stayers, "
          f"{len(ps)} deployed questions:")
    print(f"  {below} of {len(ps)} at p < 0.05 (expected {0.05*len(ps):.1f} "
          f"by chance), smallest p = {min(p for _, p in ps):.3f}, "
          f"median p = {np.median([p for _, p in ps]):.3f}")

    tasks = {
        "moved": moved.astype(np.float32),
        "moved_shuffled": rng.permutation(moved.astype(np.float32)),
        "idiolect_sign": (u > 0).astype(np.float32),
    }
    A_t = torch.tensor(A, device=dev)
    n_te = args.holdout
    n_tr = n - n_te
    n_in = n_opt + len(questions)

    out = []
    for k in args.ks:
        use = order[:k]
        mask = np.zeros(len(questions), dtype=np.float32)
        for q in use:
            mask[qpos[q]] = 1.0
        mrow = torch.tensor(mask, device=dev)

        def enc(sel):
            return nn_mod.encode(A_t[sel], mrow.expand(len(sel), -1), tab,
                                 n_opt)

        for name, y in tasks.items():
            yt = torch.tensor(y, device=dev)
            torch.manual_seed(0)
            head = nn_mod.build_net(n_in, 1, width=args.width,
                                    depth=args.depth).to(dev)
            opt = torch.optim.AdamW(head.parameters(), lr=3e-4,
                                    weight_decay=0.01)
            pos = float(yt[:n_tr].mean())
            with torch.no_grad():
                head[-1].bias.fill_(float(np.log(pos / (1 - pos)))
                                    if 0 < pos < 1 else 0.0)
            for _ in range(args.epochs):
                perm = torch.randperm(n_tr, device=dev)
                head.train()
                for i in range(0, n_tr - args.batch + 1, args.batch):
                    sel = perm[i:i + args.batch]
                    loss = F.binary_cross_entropy_with_logits(
                        head(enc(sel)).squeeze(1), yt[sel])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            head.eval()
            with torch.no_grad():
                p = torch.sigmoid(torch.cat([
                    head(enc(torch.arange(i, min(i + 8192, n), device=dev))
                         ).squeeze(1)
                    for i in range(n_tr, n, 8192)])).cpu().numpy()
            yv = y[n_tr:]
            auc, lo, hi = auc_ci(yv, p)
            base = float(yv.mean())
            row = {"task": name, "k": k, "n_train": n_tr, "n_test": n_te,
                   "base_rate": base, "auc": auc, "auc_lo95": lo,
                   "auc_hi95": hi,
                   "brier": float(np.mean((p - yv) ** 2)),
                   "brier_base": float(np.mean((base - yv) ** 2)),
                   "acc": float(((p > 0.5) == (yv > 0.5)).mean()),
                   "acc_majority": float(max(base, 1 - base))}
            out.append(row)
            print(f"  k={k:2d}  {name:15s} AUC {auc:.4f} "
                  f"[{lo:.4f},{hi:.4f}]  brier {row['brier']:.5f} "
                  f"(base {row['brier_base']:.5f})")

    path = OUT / "mover_probe.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path} ({len(out)} rows)")


def auc_ci(y, p, reps=2000, seed=11):
    """AUC with a bootstrap interval, resampling test people.

    The interval is the point of this function. A probe reporting 0.503 with no
    interval cannot be distinguished from a probe reporting 0.503 that would
    have read 0.55 on another draw, and the whole argument here rests on being
    able to say that a null is a null.
    """
    y = np.asarray(y) > 0.5
    p = np.asarray(p, dtype=np.float64)

    def one(yy, pp):
        npos, nneg = int(yy.sum()), int((~yy).sum())
        if npos == 0 or nneg == 0:
            return float("nan")
        r = np.empty(len(pp))
        r[np.argsort(pp, kind="stable")] = np.arange(1, len(pp) + 1)
        return (r[yy].sum() - npos * (npos + 1) / 2) / (npos * nneg)

    rng = np.random.default_rng(seed)
    boots = np.array([one(y[i], p[i]) for i in
                      rng.integers(0, len(y), (reps, len(y)))])
    return (one(y, p), float(np.nanpercentile(boots, 2.5)),
            float(np.nanpercentile(boots, 97.5)))


# ------------------------------------------------------------------- the fix

def mix(args):
    """What the fixed admixture of prior is worth, in a proper score.

    The derivation says the exact posterior over home is (1-m) posterior +
    m prior with m the mover rate. Coverage says that helps movers a lot at 95%
    and only a little at 80%, which is what a mixture with weight 0.15 must do.
    Coverage is not a proper score, though, so on its own it cannot say whether
    the trade is worth making. This computes the log score at the true home
    cell across a grid of mixture weights, on the same held-out people, and
    bootstraps the paired difference against m = 0.

    No sorting is involved -- only the posterior mass at one cell per person --
    so this runs in a fraction of the time `diagnose` takes and can afford a
    whole sweep. The optimum is also a measurement in its own right: if the
    weight that maximises the score lands near the simulated mover rate, then
    the quantity being fitted is the thing it is named after.
    """
    import torch

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = np.load(OUT / "neural_pool.npz", allow_pickle=True)
    lab, cent = d["labels"], d["centroids"]
    questions = [str(q) for q in d["questions"]]
    theta = float(d["theta"])
    net, _ = load_net(dev)
    g = Geolocator()
    t = g.t
    tab = torch.tensor(nn_mod.option_table(g, questions), device=dev)
    n_opt = t.logp.shape[0]
    csum = np.zeros(cent.shape[0])
    np.add.at(csum, lab, g.prior)
    w_cell = g.prior / np.clip(csum[lab], 1e-30, None)

    order = idio.deployed_questions(max(args.ks))
    qpos = {q: j for j, q in enumerate(questions)}
    n = args.n
    A, home, speech = nn_mod.make_pool(g, n, questions, theta, args.seed)
    moved = home != speech
    print(f"{n} held-out people, seed {args.seed}, {moved.sum()} movers "
          f"({moved.mean():.1%}); grid {args.grid[0]}..{args.grid[1]}")
    A_t = torch.tensor(A, device=dev)
    ph = g.prior[home]
    grid = np.linspace(args.grid[0], args.grid[1], args.steps)
    rng = np.random.default_rng(3)
    out = []

    for k in args.ks:
        use = order[:k]
        mask = np.zeros((n, len(questions)), dtype=np.float32)
        for q in use:
            mask[:, qpos[q]] = 1.0
        mt = torch.tensor(mask, device=dev)
        Pn = net_posteriors(net, A_t, mt, tab, n_opt, lab, w_cell, dev)
        p_net = Pn[np.arange(n), home].copy()
        del Pn
        P0 = bayes_posteriors(g, A, qpos, use, 0.0, n)
        p_bay = P0[np.arange(n), home].copy()
        del P0

        for name, p0 in [("bayes(rho=0)", p_bay), ("net", p_net)]:
            base = np.log(np.clip(p0, 1e-300, None))
            best = (-1e18, 0.0)
            for m in grid:
                s = np.log(np.clip((1 - m) * p0 + m * ph, 1e-300, None))
                if s.mean() > best[0]:
                    best = (float(s.mean()), float(m))
            s_dep = np.log(np.clip((1 - nn_mod.MOVER) * p0
                                   + nn_mod.MOVER * ph, 1e-300, None))
            diff = s_dep - base
            idx = rng.integers(0, n, (BOOT, n))
            bd = diff[idx].mean(1)
            row = {"model": name, "k": k, "n": n,
                   "n_mover": int(moved.sum()),
                   "logscore_m0": float(base.mean()),
                   "logscore_at_MOVER": float(s_dep.mean()),
                   "gain_nats": float(diff.mean()),
                   "gain_lo95": float(np.percentile(bd, 2.5)),
                   "gain_hi95": float(np.percentile(bd, 97.5)),
                   "gain_stayer": float(diff[~moved].mean()),
                   "gain_mover": float(diff[moved].mean()),
                   "best_m": best[1], "logscore_best": best[0]}
            out.append(row)
            print(f"  k={k:2d} {name:14s} m=0 {row['logscore_m0']:8.4f} -> "
                  f"m={nn_mod.MOVER:.2f} {row['logscore_at_MOVER']:8.4f}  "
                  f"gain {row['gain_nats']:+.4f} "
                  f"[{row['gain_lo95']:+.4f},{row['gain_hi95']:+.4f}] nats "
                  f"(stayer {row['gain_stayer']:+.4f}, "
                  f"mover {row['gain_mover']:+.4f}); "
                  f"best m {row['best_m']:.3f}")

    path = OUT / "mover_mix.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path} ({len(out)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("diagnose", help="movers vs stayers, every model")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--ks", type=int, nargs="+", default=KS)
    p.add_argument("--legacy", action="store_true",
                   help="also score the retired rho=.177 discount, which is "
                        "the scalar-temper arm the claim is set against")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", default="",
                   help="suffix for the output CSVs, so an independent-seed "
                        "replication does not overwrite the headline run")
    p.set_defaults(fn=diagnose)

    p = sub.add_parser("probe", help="is mover status in the answers at all?")
    p.add_argument("--n", type=int, default=300_000)
    p.add_argument("--holdout", type=int, default=60_000)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 14, 30])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(fn=probe)

    p = sub.add_parser("mix", help="what the prior admixture is worth")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 14, 30])
    p.add_argument("--grid", type=float, nargs=2, default=[0.0, 0.60])
    p.add_argument("--steps", type=int, default=121)
    p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(fn=mix)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
