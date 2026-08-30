"""A discriminative neural model of P(hometown | answers), and its backtest.

Everything else in this project is one generative model used backwards. The
surfaces give P(answer | cell), the prior gives P(cell), and infer.posterior
multiplies them. That is naive Bayes, and it is wrong in three ways we have
already measured:

  1. It assumes answers are conditionally independent given location. They are
     not: the within-person residual correlation is rho = 0.177 (YGDP, +-0.11).
     Independence makes the model count the same evidence repeatedly.
  2. TAU_BASE and RHO exist to patch (1) by raising the likelihood to a power.
     That is a survey design effect applied to a likelihood, which is a category
     error, and model/idiolect.py showed rho = 0 beating rho = 0.177 in 253 of
     253 matched comparisons.
  3. It cannot represent a mover. Somebody raised in Alabama and recorded in
     Oregon gives twelve answers that agree with each other and disagree with
     the label. Naive Bayes returns a narrow posterior centred on Alabama. A
     scalar temper widens every posterior equally, so it cannot tell "unsure"
     from "confidently misled" -- the two failure modes need opposite responses.

A discriminative model has none of those problems, because it never factorises.
It is handed the whole answer vector and asked directly for a distribution over
locations, so correlation between answers is something it can learn rather than
something it must assume away, and "this pattern of answers usually belongs to
somebody who has moved" is representable.

The honest limit. There is no real dataset of people with known hometowns who
answered many dialect questions -- HDS and Cambridge publish only aggregates,
and YGDP's 1450 located respondents answered five questions of which four are
one construction. So the training data here is drawn from the generative model
in model/idiolect.py, and the net can only learn what that process contains. It
cannot discover facts about American English that the surfaces do not already
hold. What it can do is stop making an independence assumption that the process
violates, which is exactly the gap that RHO was invented to paper over. The
backtest below measures that gap and nothing more, and it is reported against
the Bayes model on identical held-out people so the comparison is paired.

Representation. The grid has 50888 cells, too many to be softmax classes and
far finer than the ~250 km errors anyone achieves, so the target is a
population-weighted k-means clustering of the grid (default 1024 clusters,
~90 km apart). Predictions are splatted back onto the fine grid in proportion
to the population prior within each cluster, which makes the net's output a
distribution over the same 50888 cells the Bayes model uses. Every metric is
then computed by the same code on the same objects, and neither side is handed
a resolution advantage.

Usage:
    python neural.py prep                 build clusters and the training pool
    python neural.py train                train the network
    python neural.py eval                 backtest against the Bayes model
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import infer
import idiolect as idio
from infer import Geolocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

SEED = 20131015
OUT = DATA / "model"

# The generative process the net is trained against. These are the values
# model/idiolect.py settled on as the honest population: correlation at the
# measured level, one person in six raised somewhere other than where they are
# recorded, and surfaces that are themselves wrong by a few points.
RHO_TRUE = 0.18
MOVER = 0.15

LEGACY_RHO = 0.177
"""The design-effect discount this project deployed until it was disproved.

The curve keeps it as a negative control, because it is the evidence for the
central finding: a model carrying this discount bottoms out around twelve
questions and then gets steadily WORSE as it is told more, while both correctly
specified models keep improving. That comparison is the point of the figure.

Pinned to a literal rather than read from `infer.RHO`. It used to be read from
there, which was fine only while the two happened to be equal; the moment RHO
moves to 0 that arm would silently become a duplicate of the rho=0 arm while
still carrying the label ".177". A control that renames itself when the thing
it is controlling for changes is not a control.
"""


def _arm(rho):
    """Curve label for a Bayes arm, e.g. 0.177 -> 'bayes(rho=.177)'.

    Derived from the value so a label can never disagree with the rho it was
    actually computed at. The leading zero is dropped to match the names
    already written into neural_curve.csv and quoted in the documents.
    """
    s = f"{rho:g}"
    return f"bayes(rho={s[1:] if s.startswith('0.') else s})"


def xy(t, cells):
    """Flat kilometre coordinates, good enough for clustering CONUS."""
    lat = np.asarray(t.cell_lat[cells], dtype=np.float64)
    lon = np.asarray(t.cell_lon[cells], dtype=np.float64)
    return np.stack([lon * 111.0 * np.cos(np.radians(39.0)), lat * 111.0], 1)


def build_clusters(g, n_clusters=1024, seed=SEED):
    """Population-weighted k-means over the grid.

    Weighted by the prior rather than uniform, so cluster resolution follows
    where people actually are: dense around the northeast corridor, coarse in
    the Great Basin. That matches where the metric cares about precision.
    """
    from scipy.cluster.vq import kmeans2

    t = g.t
    rng = np.random.default_rng(seed)
    draw = rng.choice(t.n_cells, size=200_000, p=g.prior)
    pts = xy(t, draw)
    cent, _ = kmeans2(pts, n_clusters, minit="++", seed=seed, iter=40)

    allpts = xy(t, np.arange(t.n_cells))
    lab = np.empty(t.n_cells, dtype=np.int32)
    for i in range(0, t.n_cells, 4096):
        d = ((allpts[i:i + 4096, None, :] - cent[None]) ** 2).sum(-1)
        lab[i:i + 4096] = d.argmin(1)
    return cent, lab


def make_pool(g, n, questions, theta, seed):
    """Draw n people from the generative process; return answers and homes."""
    home, speech, u, chosen = idio.simulate(
        g, n, questions, theta, mover=MOVER, seed=seed)
    A = np.zeros((n, len(questions)), dtype=np.int16)
    for j, q in enumerate(questions):
        A[:, j] = chosen[q]
    return A, home.astype(np.int32), speech.astype(np.int32)


def prep(args):
    g = Geolocator()
    t = g.t
    questions = [str(q) for q in t.questions]
    print(f"{len(questions)} questions, {t.n_cells} cells")

    t0 = time.time()
    cent, lab = build_clusters(g, args.clusters)
    print(f"clustered into {args.clusters} in {time.time()-t0:.0f}s")
    d = np.sqrt(((xy(t, np.arange(t.n_cells)) - cent[lab]) ** 2).sum(1))
    print(f"  cell to centroid: median {np.median(d):.0f} km, "
          f"p90 {np.percentile(d,90):.0f} km")

    t0 = time.time()
    theta, rho = idio.calibrate_theta(g, questions[:20], RHO_TRUE)
    print(f"theta {theta:.3f} -> realised rho {rho:.3f} "
          f"({time.time()-t0:.0f}s)")

    t0 = time.time()
    A, home, speech = make_pool(g, args.n, questions, theta, SEED)
    print(f"pool {A.shape} in {time.time()-t0:.0f}s")

    path = OUT / f"neural_pool.npz"
    np.savez_compressed(
        path, answers=A, home=home, speech=speech, centroids=cent,
        labels=lab, questions=np.array(questions), theta=theta, rho=rho)
    print(f"wrote {path} ({path.stat().st_size/1e6:.0f} MB)")


# ---------------------------------------------------------------- the network

def option_table(g, questions):
    """(n_questions, max_choices) map from (question, choice) to tensor row.

    t.rows[q] lists the rows of the 680-row likelihood tensor belonging to
    question q. Flattening that into a rectangular lookup lets a whole batch of
    answer vectors be turned into multi-hot input with one gather.
    """
    t = g.t
    width = max(len(t.rows[q]) for q in questions)
    tab = np.zeros((len(questions), width), dtype=np.int64)
    for j, q in enumerate(questions):
        r = t.rows[q]
        tab[j, :len(r)] = r
        tab[j, len(r):] = r[-1]
    return tab


def build_net(n_in, n_out, width=1024, depth=3):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.f = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w),
                                   nn.GELU(), nn.Linear(w, w))

        def forward(self, x):
            return x + self.f(x)

    layers = [nn.Linear(n_in, width)]
    layers += [Block(width) for _ in range(depth)]
    layers += [nn.LayerNorm(width), nn.Linear(width, n_out)]
    return nn.Sequential(*layers)


def sample_masks(bits, B, kmax, rng, device):
    """Which questions each training example gets to see.

    A quiz does not ask random questions -- it asks the informative ones first,
    so a net trained only on uniform subsets would be evaluated off its training
    distribution. Gumbel top-k with an exponent drawn per example interpolates
    between the two regimes: alpha = 0 is a uniform subset, alpha = 3 is close to
    always taking the highest-information questions. Training across the range
    means one network serves any selection policy and any k.
    """
    import torch

    Q = bits.shape[0]
    alpha = torch.randint(0, 4, (B, 1), device=device).float()
    gum = -torch.log(-torch.log(torch.rand(B, Q, device=device) + 1e-9) + 1e-9)
    keys = alpha * bits[None, :] + gum
    order = keys.argsort(dim=1, descending=True)
    k = torch.randint(1, kmax + 1, (B, 1), device=device)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(Q, device=device).expand(B, Q))
    return (rank < k).float()


def encode(A, mask, tab, n_opt):
    """Answer indices plus a mask -> multi-hot over options, plus the mask."""
    import torch

    idx = torch.gather(tab.expand(A.shape[0], -1, -1), 2,
                       A.long().unsqueeze(-1)).squeeze(-1)
    x = torch.zeros(A.shape[0], n_opt, device=A.device)
    x.scatter_(1, idx, mask)
    return torch.cat([x, mask], 1)


def train(args):
    import torch
    import torch.nn.functional as F

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = np.load(OUT / "neural_pool.npz", allow_pickle=True)
    A_all = d["answers"]
    home_all = d["home"]
    cent = d["centroids"]
    lab = d["labels"]
    questions = [str(q) for q in d["questions"]]
    print(f"pool {A_all.shape}, {cent.shape[0]} clusters, device {dev}")

    g = Geolocator()
    tab = torch.tensor(option_table(g, questions), device=dev)
    n_opt = g.t.logp.shape[0]
    bits = torch.tensor(
        [g.question_bits.get(q, 0.0) for q in questions],
        dtype=torch.float32, device=dev)
    bits = (bits - bits.mean()) / (bits.std() + 1e-6)

    cell_xy = torch.tensor(xy(g.t, np.arange(g.t.n_cells)),
                           dtype=torch.float32, device=dev)
    cent_t = torch.tensor(cent, dtype=torch.float32, device=dev)

    n_val = 20_000
    A_va = torch.tensor(A_all[-n_val:], device=dev)
    h_va = torch.tensor(home_all[-n_val:].astype(np.int64), device=dev)
    A_tr = torch.tensor(A_all[:-n_val], device=dev)
    h_tr = torch.tensor(home_all[:-n_val].astype(np.int64), device=dev)
    n_tr = A_tr.shape[0]
    theta = float(d["theta"])

    net = build_net(n_opt + len(questions), cent.shape[0],
                    width=args.width, depth=args.depth).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    print(f"{npar/1e6:.1f}M parameters, refresh every {args.refresh} epoch(s)")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (n_tr // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    rng = np.random.default_rng(SEED)
    sigma = args.sigma

    def soft_target(h):
        d2 = ((cell_xy[h][:, None, :] - cent_t[None]) ** 2).sum(-1)
        return F.softmax(-d2 / (2 * sigma ** 2), dim=1)

    best = (1e9, -1)
    t0 = time.time()
    for ep in range(args.epochs):
        # Fresh people, not just fresh masks. A frozen pool is a finite sample
        # however large, and the first run of this file overfit it from epoch 36
        # onward: training loss kept falling while held-out loss rose. Drawing a
        # new pool costs three seconds against a ten second epoch and removes
        # the failure mode rather than regularising around it.
        if args.refresh and ep > 0 and ep % args.refresh == 0:
            A_np, h_np, _ = make_pool(g, n_tr, questions, theta,
                                      SEED + 1000 * ep)
            A_tr = torch.tensor(A_np, device=dev)
            h_tr = torch.tensor(h_np.astype(np.int64), device=dev)

        perm = torch.randperm(n_tr, device=dev)
        net.train()
        tot = 0.0
        nb = 0
        for i in range(0, n_tr - args.batch + 1, args.batch):
            sel = perm[i:i + args.batch]
            A = A_tr[sel]
            mask = sample_masks(bits, A.shape[0], args.kmax, rng, dev)
            x = encode(A, mask, tab, n_opt)
            logits = net(x)
            loss = -(soft_target(h_tr[sel]) *
                     F.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1

        net.eval()
        with torch.no_grad():
            vrng = np.random.default_rng(12345)     # same masks every epoch
            torch.manual_seed(12345)
            vl, vn = 0.0, 0
            for i in range(0, A_va.shape[0], 4096):
                A = A_va[i:i + 4096]
                mask = sample_masks(bits, A.shape[0], args.kmax, vrng, dev)
                logits = net(encode(A, mask, tab, n_opt))
                vl += float(-(soft_target(h_va[i:i + 4096]) *
                              F.log_softmax(logits, 1)).sum(1).mean())
                vn += 1
        val = vl / vn
        star = ""
        if val < best[0]:
            best = (val, ep)
            star = "  *"
            torch.save({"state": {k: v.detach().cpu().clone()
                                  for k, v in net.state_dict().items()},
                        "width": args.width, "depth": args.depth,
                        "n_in": n_opt + len(questions), "n_out": cent.shape[0],
                        "questions": questions, "sigma": sigma, "epoch": ep,
                        "val": val}, OUT / "neural_net.pt")
        print(f"  epoch {ep+1:3d}/{args.epochs}  train {tot/nb:.4f}  "
              f"val {val:.4f}  {time.time()-t0:.0f}s{star}")

    print(f"best val {best[0]:.4f} at epoch {best[1]+1}; "
          f"kept that checkpoint at {OUT / 'neural_net.pt'}")


# ----------------------------------------------------------------- backtesting

def person_stats(g, post, cells):
    """Per-person outcomes for a block of posteriors. Chunkable.

    Returned as raw per-person arrays rather than summaries so that a caller can
    accumulate over chunks and only aggregate at the end; a (n, 50888) float64
    block is 400 MB per thousand people, which is the whole reason this is not
    computed in one go.
    """
    from infer import credible_cells
    from tensor import haversine
    from calibrate import LEVELS

    t = g.t
    best = post.argmax(1)
    err = haversine(t.cell_lat[cells], t.cell_lon[cells],
                    t.cell_lat[best], t.cell_lon[best])
    same = np.array([str(t.state[b]) == str(t.state[c])
                     for b, c in zip(best, cells)])
    logs = np.log(np.clip(post[np.arange(len(cells)), cells], 1e-300, None))
    cov = {lv: np.array([cells[i] in set(credible_cells(post[i], lv).tolist())
                         for i in range(len(cells))]) for lv in LEVELS}
    return err, same, logs, cov


def aggregate(parts, label, k):
    from calibrate import LEVELS

    err = np.concatenate([p[0] for p in parts])
    same = np.concatenate([p[1] for p in parts])
    logs = np.concatenate([p[2] for p in parts])
    cov = {lv: float(np.concatenate([p[3][lv] for p in parts]).mean())
           for lv in LEVELS}
    return {
        "model": label, "k": k, "n": len(err),
        "median_km": float(np.median(err)),
        "p90_km": float(np.percentile(err, 90)),
        "within_150km": float((err <= 150).mean()),
        "state_acc": float(same.mean()),
        "logscore": float(logs.mean()),
        "calib_err": float(np.mean([abs(cov[lv] - lv) for lv in LEVELS])),
        **{f"cover{int(lv*100)}": cov[lv] for lv in LEVELS},
    }


def metrics(g, post, cells, label, k):
    return aggregate([person_stats(g, post, cells)], label, k)


def splat(g, pc, lab, w_cell):
    """Cluster probabilities -> a distribution over all 50888 grid cells.

    Mass inside a cluster is divided in proportion to the population prior,
    which is the only defensible split given the net was never told anything
    finer. The result lives on the same support as the Bayes posterior, so the
    two can be scored by the same function without either being handicapped.
    """
    return pc[:, lab] * w_cell[None, :]


def evaluate(args):
    import torch
    import torch.nn.functional as F

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = np.load(OUT / "neural_pool.npz", allow_pickle=True)
    cent, lab = d["centroids"], d["labels"]
    questions = [str(q) for q in d["questions"]]
    theta = float(d["theta"])

    ck = torch.load(OUT / "neural_net.pt", map_location=dev, weights_only=False)
    net = build_net(ck["n_in"], ck["n_out"], ck["width"], ck["depth"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()

    g = Geolocator()
    t = g.t
    tab = torch.tensor(option_table(g, questions), device=dev)
    n_opt = t.logp.shape[0]

    csum = np.zeros(cent.shape[0])
    np.add.at(csum, lab, g.prior)
    w_cell = g.prior / np.clip(csum[lab], 1e-30, None)

    order = idio.deployed_questions(max(args.ks))
    qpos = {q: j for j, q in enumerate(questions)}

    print(f"held-out people: {args.n}, seed {args.seed} (train seed {SEED})")
    A, home, speech = make_pool(g, args.n, questions, theta, args.seed)
    A_t = torch.tensor(A, device=dev)

    rows = []
    for k in args.ks:
        use = order[:k]
        mask = np.zeros((args.n, len(questions)), dtype=np.float32)
        for q in use:
            mask[:, qpos[q]] = 1.0
        with torch.no_grad():
            out = []
            mt = torch.tensor(mask, device=dev)
            for i in range(0, args.n, 4096):
                x = encode(A_t[i:i + 4096], mt[i:i + 4096], tab, n_opt)
                out.append(F.softmax(net(x), 1).cpu().numpy())
        pc = np.concatenate(out).astype(np.float64)
        pc /= pc.sum(1, keepdims=True)
        rows.append(metrics(g, splat(g, pc, lab, w_cell), home, "net", k))

        for rho, name in [(infer.RHO, "bayes(rho=.177)"), (0.0, "bayes(rho=0)")]:
            keep = infer.RHO
            infer.RHO = rho
            try:
                P = np.empty((args.n, t.n_cells))
                for i in range(args.n):
                    ans = [(q, t.choice[t.rows[q][A[i, qpos[q]]]]) for q in use]
                    P[i] = g.posterior(ans)
            finally:
                infer.RHO = keep
            rows.append(metrics(g, P, home, name, k))
            if rho == 0.0:
                # The same posterior forced through the net's representation:
                # summed into clusters, then splatted back by prior. Any gap
                # between this and bayes(rho=0) is the price of discretising to
                # 1024 classes, and any gap between this and the net is what
                # the network actually learned. Without this arm the two are
                # confounded and the comparison says nothing.
                pcb = np.zeros((args.n, cent.shape[0]))
                np.add.at(pcb.T, lab, P.T)
                pcb /= pcb.sum(1, keepdims=True)
                rows.append(metrics(g, splat(g, pcb, lab, w_cell), home,
                                    "bayes(rho=0)@cluster", k))

        for r in rows[-4:]:
            print(f"  k={r['k']:2d}  {r['model']:16s} "
                  f"{r['median_km']:6.0f} km  state {r['state_acc']*100:5.1f}%  "
                  f"log {r['logscore']:8.3f}  cover80 {r['cover80']:.3f}")

    import csv as _csv
    path = OUT / "neural_backtest.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def curve(args):
    """Accuracy against number of questions, for both models, same people.

    The k = 12 in the site was inherited from the NYT quiz format, not derived.
    This measures where the returns actually stop, separately for the Bayes
    model and the network, because they fail differently: the tempered Bayes
    posterior turns over once the discount outgrows the evidence, and the net
    has no discount to turn over but is capped by its 1024-cluster output.
    There is no reason those two should stop paying at the same place.
    """
    import torch
    import torch.nn.functional as F

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = np.load(OUT / "neural_pool.npz")
    cent, lab = d["centroids"], d["labels"]
    questions = [str(q) for q in d["questions"]]
    theta = float(d["theta"])

    ck = torch.load(OUT / "neural_net.pt", map_location=dev, weights_only=False)
    net = build_net(ck["n_in"], ck["n_out"], ck["width"], ck["depth"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    print(f"net from epoch {ck.get('epoch', '?')} (val {ck.get('val', 0):.4f})")

    g = Geolocator()
    t = g.t
    tab = torch.tensor(option_table(g, questions), device=dev)
    n_opt = t.logp.shape[0]
    csum = np.zeros(cent.shape[0])
    np.add.at(csum, lab, g.prior)
    w_cell = g.prior / np.clip(csum[lab], 1e-30, None)

    order = idio.deployed_questions(args.kmax, path=args.order)
    qpos = {q: j for j, q in enumerate(questions)}
    A, home, _ = make_pool(g, args.n, questions, theta, args.seed)
    A_t = torch.tensor(A, device=dev)
    C = args.chunk

    rows = []
    for k in range(1, args.kmax + 1):
        use = order[:k]
        mask = np.zeros((args.n, len(questions)), dtype=np.float32)
        for q in use:
            mask[:, qpos[q]] = 1.0
        mt = torch.tensor(mask, device=dev)

        parts = []
        with torch.no_grad():
            for i in range(0, args.n, C):
                x = encode(A_t[i:i + C], mt[i:i + C], tab, n_opt)
                pc = F.softmax(net(x), 1).cpu().numpy().astype(np.float64)
                pc /= pc.sum(1, keepdims=True)
                parts.append(person_stats(g, splat(g, pc, lab, w_cell),
                                          home[i:i + C]))
        rows.append(aggregate(parts, "net", k))

        for rho, name in [(0.0, _arm(0.0)), (LEGACY_RHO, _arm(LEGACY_RHO))]:
            keep = infer.RHO
            infer.RHO = rho
            try:
                parts = []
                for i in range(0, args.n, C):
                    m = min(C, args.n - i)
                    P = np.empty((m, t.n_cells))
                    for j in range(m):
                        ans = [(q, t.choice[t.rows[q][A[i + j, qpos[q]]]])
                               for q in use]
                        P[j] = g.posterior(ans)
                    parts.append(person_stats(g, P, home[i:i + m]))
            finally:
                infer.RHO = keep
            rows.append(aggregate(parts, name, k))

        line = "  ".join(
            f"{r['model'].split('(')[0][:5]}:{r['median_km']:5.0f}km/"
            f"{r['state_acc']*100:4.1f}%" for r in rows[-3:])
        print(f"  k={k:2d}  {line}")

    import csv as _csv
    path = Path(args.out) if args.out else OUT / "neural_curve.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")
    report_curve(rows)


def report_curve(rows):
    """Where the returns stop, per model."""
    for name in ["net", _arm(0.0), _arm(LEGACY_RHO)]:
        r = sorted([x for x in rows if x["model"] == name],
                   key=lambda x: x["k"])
        if not r:
            continue
        km = [x["median_km"] for x in r]
        best = min(km)
        total = km[0] - best
        print(f"\n{name}: {km[0]:.0f} km at k=1 -> {best:.0f} km best "
              f"(k={r[km.index(best)]['k']})")
        print("   marginal km saved per question:")
        for a in range(0, len(km) - 1, 4):
            b = min(a + 4, len(km) - 1)
            print(f"     k={a+1:2d}-{b+1:2d}  {(km[a]-km[b])/(b-a):6.1f} km/q")
        for frac in (0.5, 0.75, 0.9, 0.95):
            hit = next((r[i]["k"] for i in range(len(km))
                        if km[0] - km[i] >= frac * total), None)
            print(f"     {int(frac*100)}% of total reduction by k={hit}")
        for lam in (10, 20, 30, 50):
            kstar = min(range(len(km)), key=lambda i: km[i] + lam * (i + 1))
            print(f"     at {lam:2d} km/question, optimal k = {kstar+1}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep", help="build clusters and training pool")
    p.add_argument("--clusters", type=int, default=1024)
    p.add_argument("--n", type=int, default=400_000)
    p.set_defaults(fn=prep)

    p = sub.add_parser("train", help="train the network")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--kmax", type=int, default=40)
    p.add_argument("--sigma", type=float, default=100.0)
    p.add_argument("--refresh", type=int, default=1,
                   help="redraw the training pool every N epochs; 0 to freeze")
    p.set_defaults(fn=train)

    p = sub.add_parser("eval", help="backtest against the Bayes model")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=771131)
    p.add_argument("--ks", type=int, nargs="+",
                   default=[1, 3, 5, 8, 12, 16, 20])
    p.set_defaults(fn=evaluate)

    p = sub.add_parser("curve", help="accuracy vs number of questions")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--seed", type=int, default=771131)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--order", type=str, default=None,
                   help="question ordering CSV; default is the deployed one. "
                        "Lets a candidate ordering be scored before it is "
                        "deployed, so the ordering, RHO and the quiz length "
                        "can be changed in one step instead of three.")
    p.add_argument("--out", type=str, default=None,
                   help="write here instead of data/model/neural_curve.csv")
    p.set_defaults(fn=curve)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
