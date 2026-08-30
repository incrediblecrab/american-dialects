"""Choose which questions to ask, and in what order.

The measure is mutual information: how many bits does knowing the answer to
this question buy about where the person is from, given what is already known.

    I(A ; C) = H( sum_c w_c P(a|c) ) - sum_c w_c H( P(.|c) )

The first term is the uncertainty about their answer, the second is how
uncertain the answer stays once the location is fixed. The difference is the
part of the answer that is actually about geography rather than about noise.

Conditioning on "what is already known" is doing real work here, and it is the
reason to prefer adaptive selection. Naive Bayes cannot see that y'all,
fixin' to and might could are three ways of asking one question, so scoring
questions against a fixed prior would happily pick all three. Scored against the
running posterior they cannot: once y'all has moved the mass south, the other
two are near-constant over everything still plausible, and their information
collapses. Redundancy is invisible to the model but visible to the posterior.

Two orderings come out of this. The fixed one is a fair quiz to hand to anyone;
the adaptive one branches on each answer and is strictly better, at the cost of
having to be run interactively.

The simulated respondents here are drawn from the model's own surfaces. That is
legitimate for *designing* a quiz, which is a question about the model's
beliefs. It is worse than merely inadmissible for measuring accuracy: each
simulated answer is drawn independently given the true cell, so the synthetic
world satisfies the conditional-independence assumption exactly. In that world
tau=1 is correct and the model is perfectly calibrated by construction. So this
simulation cannot detect the single flaw most likely to break the model, and its
error and accuracy columns are labelled self-consistency for that reason. Real
numbers come from validate.py and people whose locations are known.
"""

import argparse

import numpy as np

from infer import Geolocator, tau_for

SEED = 20130801  # the month the NYT quiz went up


class Selector:
    def __init__(self, g, cell_stride=1):
        self.g = g
        self.t = g.t
        self.cells = np.arange(0, self.t.n_cells, cell_stride)
        self.prior = g.prior[self.cells]
        self.prior = self.prior / self.prior.sum()
        self.p, self.cond_h = {}, {}
        for q in self.t.questions:
            rows = self.t.rows[q]
            p = np.exp(self.t.logp[rows][:, self.cells]).astype(np.float32)
            p /= np.maximum(p.sum(axis=0, keepdims=True), 1e-12)
            self.p[q] = p
            self.cond_h[q] = -(p * np.log2(np.maximum(p, 1e-12))).sum(axis=0)

    def information(self, w, questions=None):
        """Bits about location per question, for a batch of posteriors.

        w has shape (n_posteriors, n_cells).
        """
        out = {}
        for q in (questions or self.t.questions):
            marg = w @ self.p[q].T
            marg = np.maximum(marg, 1e-12)
            marg /= marg.sum(axis=1, keepdims=True)
            h_marg = -(marg * np.log2(marg)).sum(axis=1)
            out[q] = h_marg - (w @ self.cond_h[q])
        return out

    def personas(self, m, rng):
        """Simulated respondents: a true home cell, drawn by population."""
        return rng.choice(len(self.cells), size=m, p=self.prior)

    def order(self, k=25, m=400, tau=None, rho=None, verbose=True, score=None):
        """Greedy fixed ordering, scored on simulated respondents.

        The running posterior is discounted with the same k-dependent tau the
        deployed model uses, because selection sees redundancy only through the
        posterior; an undiscounted posterior collapses early and then reports
        every remaining question as uninformative for the wrong reason.

        `rho` overrides the module-level RHO for the discount schedule only.
        This exists because the ordering and the discount are entangled: the
        deployed ordering was derived under RHO = 0.177, and findings.md shows
        that discount is itself the largest single source of error. Re-deriving
        the ordering under the discount it disproves would bake the error in
        again, so the two have to be able to move independently.

        `score(question, bits) -> float` replaces bits as the greedy objective.
        The default is bits itself, so the deployed ordering is unaffected. It
        exists so that an alternative selection criterion can be tried against
        an ordering derived by identical machinery on identical personas, with
        the criterion as the only difference; model/order_compare.py uses it to
        rank on the regional-signal-to-idiolect ratio. The reported `bits` is
        always the information, whatever the ordering was chosen on, so the two
        files stay readable side by side.
        """
        rng = np.random.default_rng(SEED)
        truth = self.personas(m, rng)
        w = np.tile(self.prior.astype(np.float32), (m, 1))
        remaining = list(self.t.questions)
        picked, curve = [], []

        h0 = float(np.mean(-(w * np.log2(np.maximum(w, 1e-12))).sum(axis=1)))
        if verbose:
            print("columns below the entropy are self-consistency, "
                  "not accuracy: the personas obey the model exactly\n")
            print(f"{'#':>3} {'q':>5} {'bits gained':>12} {'entropy':>9} "
                  f"{'sc km':>10} {'sc state %':>12}")
        for step in range(1, k + 1):
            info = self.information(w, remaining)
            bits = {q: float(info[q].mean()) for q in remaining}
            best = max(remaining,
                       key=lambda q: score(q, bits[q]) if score else bits[q])
            gained = bits[best]

            p = self.p[best]
            step_tau = (tau_for(step, rho=rho) / tau_for(step - 1, rho=rho)
                        if tau is None else tau)
            for i in range(m):
                probs = p[:, truth[i]].astype(np.float64)
                probs = probs / probs.sum()
                a = rng.choice(len(probs), p=probs)
                w[i] *= np.maximum(p[a], 1e-12) ** step_tau
                w[i] /= w[i].sum()

            remaining.remove(best)
            picked.append(best)
            h = float(np.mean(-(w * np.log2(np.maximum(w, 1e-12))).sum(axis=1)))
            km = self.median_error(w, truth)
            acc = self.state_accuracy(w, truth)
            curve.append({"n": step, "question": best, "bits": gained,
                          "entropy": h, "selfconsistency_km": km,
                          "selfconsistency_state_acc": acc})
            if verbose:
                print(f"{step:>3} {best:>5} {gained:>12.3f} {h:>9.2f} "
                      f"{km:>10.0f} {acc:>12.1%}")
        return picked, curve, h0

    def median_error(self, w, truth):
        """Self-consistency, not accuracy. See the module docstring."""
        from tensor import haversine
        best = w.argmax(axis=1)
        cells = self.cells
        d = haversine(self.t.cell_lat[cells[truth]], self.t.cell_lon[cells[truth]],
                      self.t.cell_lat[cells[best]], self.t.cell_lon[cells[best]])
        return float(np.median(d))

    def state_accuracy(self, w, truth):
        """Self-consistency, not accuracy. See the module docstring."""
        states = self.t.state[self.cells]
        best = w.argmax(axis=1)
        return float(np.mean(states[best] == states[truth]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=25)
    ap.add_argument("--personas", type=int, default=400)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tau", type=float, default=None,
                    help="flat tau; default is the k-dependent tau_for(k)")
    ap.add_argument("--rho", type=float, default=None,
                    help="override RHO in the discount schedule; --rho 0 turns "
                         "the design-effect discount off")
    ap.add_argument("--out", type=str, default=None,
                    help="write here instead of data/model/question_order.csv")
    args = ap.parse_args()

    g = Geolocator()
    s = Selector(g, cell_stride=args.stride)
    import infer
    print(f"{len(s.cells)} cells, {len(s.t.questions)} questions, "
          f"{args.personas} simulated respondents, "
          f"tau={args.tau if args.tau else 'auto'}, "
          f"rho={infer.RHO if args.rho is None else args.rho}\n")
    picked, curve, h0 = s.order(k=args.questions, m=args.personas, tau=args.tau,
                                rho=args.rho)

    import csv
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
    from common import DATA
    out = Path(args.out) if args.out else DATA / "model" / "question_order.csv"
    texts = {}
    with open(DATA / "hds" / "questions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            texts[r["question"]] = r["text"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "question", "bits", "entropy",
                                          "selfconsistency_km",
                                          "selfconsistency_state_acc", "text"])
        w.writeheader()
        for row in curve:
            row = dict(row, text=texts.get(row["question"], ""))
            w.writerow(row)
    print(f"\nstarting entropy {h0:.2f} bits; wrote {out}")


if __name__ == "__main__":
    main()
