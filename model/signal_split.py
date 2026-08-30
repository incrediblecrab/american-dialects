"""Split each question's answer variability into the part that is about place
and the part that is about the person.

rho is measured pooled: one within-person residual correlation across all
questions, 0.177 on the YGDP respondents, reproduced in simulation by
model/idiolect.py. Pooling is a statement about the population, not about any
particular question, and per question the split is obviously not uniform. Asking
what someone calls a sweetened carbonated beverage is asking where they are
from. Asking whether they would say "the car needs washed" is asking rather more
about them.

The two quantities here are both mutual informations of the answer with a latent
variable, so they are on the same scale and directly comparable:

    I(A ; C)  bits the answer carries about the grid cell
    I(A ; U)  bits the answer carries about the person's idiolect

C is the location, drawn from the population prior. U is the latent
broadly-marked / broadly-standard axis that model/idiolect.py defines and
calibrates: a person with u > 0 says yinz and bubbler and pop, a person with
u < 0 says you guys and water fountain and soda. Given both, the answer
distribution is the surface tilted along markedness,

    P(a | c, u)  ∝  P(a | c) exp(theta * u * z_a)

with z the national surprisal of choice a, centred and scaled to unit variance
within the question so that theta means the same thing whether the question has
two choices or nine. theta is not a free parameter here: it is the value
model/idiolect.py's bisection settled on to make the realised pooled correlation
equal the measured one, and it is read from the pool the network was trained on
so that the measurement and the population it is tested against are the same
world.

C and U are independent by construction -- the location term is untouched by the
tilt -- so the two informations are not competing for the same variance in any
mechanical way, and their ratio is a real property of the question.

Three things follow, and only the first two are used anywhere.

  * `ratio` = I(A;C) / I(A;U) is the regional-signal-to-idiolect ratio. It says
    how much of what a question extracts is about geography rather than about
    the respondent. High ratio: the question is a good use of a tap. Low ratio:
    the respondent spends a tap and the model learns mostly about them.

  * `geo_share` = I(A;C) / (I(A;C) + I(A;U)) is the same thing bounded to (0,1),
    which is what a selection rule wants, because a pure ratio is indifferent to
    magnitude and will happily prefer a question carrying a hundredth of a bit
    cleanly over one carrying half a bit at 90% purity.

  * `rho_q` = the per-question version of the pooled rho, for completeness. Under
    the single-factor tilt the residual correlation between two questions is
    sqrt(h_q h_q') where h is the idiolect share of the NON-geographic variance,
    so the pooled number is the mean of that over pairs and the per-question one
    is the mean over that question's row. It reproduces the pooled figure by
    construction and is reported so the decomposition can be checked against
    the number it came from.

`rho_q` is deliberately NOT used to discount anything. That is the move this
project has already tested and disproved at the pooled level, and there is no
reason a per-question version of a discount that made the model worse would make
it better. It is measured, written down, tested once as a clearly labelled
losing arm in model/order_compare.py, and otherwise left alone. What a
per-question ratio is legitimately for is deciding WHICH QUESTIONS TO ASK. That
is model/order_compare.py's business, and it touches the evidence not at all.

The ranking is the part to read first, because an estimator that called
obviously regional questions personal would be broken and it would show here
before it showed anywhere else. It does not. The top is what a dialectologist
would write down unprompted -- what "the City" means, the grey crustacean that
rolls into a ball, the sweetened carbonated beverage, the night before
Halloween, the long sandwich, the thing you drink water from at school. The
bottom is the how-do-YOU-say-it items with no isogloss behind them: the "s" in
chromosome, the -sp- in thespian, amphitheater, citizen, coupon, poem, umbrella.
Split by question class, the 31 single-word pronunciation prompts have a median
ratio of 0.47 and the 91 lexical and syntactic questions 1.29.

The check that makes that convincing is the exception rather than the rule.
If the estimator were merely rediscovering "pronunciation questions are noisy"
it would put every phonetic item at the bottom. It does not: cot/caught ranks
14th of 122, Mary/merry/marry 20th and aunt 21st, which are precisely the three
pronunciation variables in the Harvard set that are real American isoglosses.
The estimator separates them from the rest of the phonetics without being told
they are different.

Run:  ./.venv/bin/python model/signal_split.py
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from idiolect import markedness
from infer import Geolocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

OUT = DATA / "model" / "question_signal.csv"
POOL = DATA / "model" / "neural_pool.npz"
N_NODES = 25          # Gauss-Hermite nodes for the u integral
POOLED_RHO = 0.177    # the measured within-person correlation, YGDP, n=349


def gauss_hermite(n=N_NODES):
    """Nodes and weights for E[f(u)] with u ~ N(0,1)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


def pool_theta(path=POOL):
    """The idiolect strength the simulator was calibrated to.

    Read rather than recomputed. model/idiolect.py bisects theta against the
    YGDP estimator and model/neural.py stores the result alongside the people it
    generated; reading it back guarantees the decomposition describes the same
    world the orderings are then scored in. A recomputation would drift with the
    bisection seed and silently describe a slightly different population.
    """
    d = np.load(path)
    return float(d["theta"]), float(d["rho"])


def split_question(g, q, theta, nodes, weights):
    """I(A;C), I(A;U) and the variance decomposition for one question.

    Everything is exact given the tilt model: the integral over u is quadrature
    and the sum over cells is the full prior, so there is no sampling noise in
    any column of the output.
    """
    rows = g.t.rows[q]
    prior = g.prior
    P = np.exp(np.asarray(g.t.logp[rows], dtype=np.float64))
    P /= np.maximum(P.sum(axis=0, keepdims=True), 1e-300)
    z, surp = markedness(g, q)

    pa_c = np.zeros_like(P)               # P(a | c), marginal over u
    pa_u = np.zeros((len(nodes), len(rows)))
    v_noise = 0.0                         # E_cu Var(s | c,u)
    mu_cu = np.zeros((len(nodes), P.shape[1]))

    for i, (u, w) in enumerate(zip(nodes, weights)):
        lp = np.log(np.maximum(P, 1e-300)) + theta * u * z[:, None]
        lp -= lp.max(axis=0, keepdims=True)
        Pu = np.exp(lp)
        Pu /= Pu.sum(axis=0, keepdims=True)
        pa_c += w * Pu
        pa_u[i] = Pu @ prior
        m = surp @ Pu
        mu_cu[i] = m
        v_noise += w * float(((Pu * (surp[:, None] - m[None, :]) ** 2).sum(0)
                              * prior).sum())

    pa = weights @ pa_u
    pa = np.maximum(pa, 1e-300)

    i_c = float((prior[None, :] * pa_c
                 * np.log2(np.maximum(pa_c, 1e-300) / pa[:, None])).sum())
    i_u = float((weights[:, None] * pa_u
                 * np.log2(np.maximum(pa_u, 1e-300) / pa[None, :])).sum())

    h_a = float(-(pa * np.log2(pa)).sum())
    h_a_c = float((prior[None, :] * -pa_c
                   * np.log2(np.maximum(pa_c, 1e-300))).sum())

    mu_c = weights @ mu_cu                      # E_u[s | c]
    mu = float(mu_c @ prior)
    v_geo = float(((mu_c - mu) ** 2) @ prior)
    v_idio = float((weights[:, None] * (mu_cu - mu_c[None, :]) ** 2).sum(0)
                   @ prior)

    return {
        "question": q, "choices": len(rows),
        "mi_bits": i_c, "idiolect_bits": i_u,
        "entropy_bits": h_a, "cond_entropy_bits": h_a_c,
        "var_geo": v_geo, "var_idiolect": v_idio, "var_noise": v_noise,
    }


def measure(g, theta, questions=None, n_nodes=N_NODES, pooled_rho=POOLED_RHO):
    """Per-question split for every question, plus the rho row means."""
    nodes, weights = gauss_hermite(n_nodes)
    rows = [split_question(g, q, theta, nodes, weights)
            for q in (questions or g.t.questions)]

    # h is the idiolect share of the non-geographic variance: the part of what
    # is left after location that is the shared person axis rather than
    # independent noise. Under one factor, corr(q,q') = sqrt(h_q h_q').
    h = np.array([r["var_idiolect"] / max(r["var_idiolect"] + r["var_noise"],
                                          1e-300) for r in rows])
    s = np.sqrt(h)
    k = len(rows)
    implied = float((s.sum() ** 2 - (s ** 2).sum()) / (k * (k - 1)))
    scale = pooled_rho / implied if implied > 0 else 1.0

    for i, r in enumerate(rows):
        others = (s.sum() - s[i]) / (k - 1)
        r["rho_q"] = float(s[i] * others * scale)
        r["idiolect_share"] = float(h[i])
        r["ratio"] = (r["mi_bits"] / r["idiolect_bits"]
                      if r["idiolect_bits"] > 1e-12 else float("inf"))
        r["geo_share"] = (r["mi_bits"] / (r["mi_bits"] + r["idiolect_bits"])
                          if r["mi_bits"] + r["idiolect_bits"] > 1e-12 else 0.0)
    return rows, implied, scale


def pair_scale(sig, pooled_rho=POOLED_RHO):
    """Recover the calibration constant from a loaded table.

    The single-factor model says corr(q, q') = scale * sqrt(h_q h_q'), and scale
    is fixed by making the mean over all pairs equal the measured pooled rho.
    Recomputed here rather than stored so that a subset's mean pairwise
    correlation and the per-question rho in the CSV can never come from
    different constants.
    """
    s = np.sqrt(np.array([r["idiolect_share"] for r in sig.values()]))
    k = len(s)
    implied = float((s.sum() ** 2 - (s ** 2).sum()) / (k * (k - 1)))
    return pooled_rho / implied if implied > 0 else 1.0


def pair_rho(sig, questions, scale=None):
    """Mean pairwise residual correlation over a subset of questions.

    This is what a design effect needs when the units are not exchangeable: not
    one global rho but the average correlation among the questions actually
    asked. It reduces to the pooled figure when the subset is representative,
    which is itself worth knowing.
    """
    scale = pair_scale(sig) if scale is None else scale
    s = np.sqrt(np.array([sig[q]["idiolect_share"] for q in questions]))
    k = len(s)
    if k < 2:
        return 0.0
    return float(scale * (s.sum() ** 2 - (s ** 2).sum()) / (k * (k - 1)))


def question_text():
    out = {}
    with open(DATA / "hds" / "questions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["question"]] = r["text"]
    return out


FIELDS = ["question", "choices", "mi_bits", "idiolect_bits", "ratio",
          "geo_share", "rho_q", "idiolect_share", "entropy_bits",
          "cond_entropy_bits", "var_geo", "var_idiolect", "var_noise", "text"]


def load(path=OUT):
    """Read the measured split back, keyed by question."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["question"]] = {
                k: (float(v) if k not in ("question", "text", "choices")
                    else (int(v) if k == "choices" else v))
                for k, v in r.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=N_NODES)
    ap.add_argument("--theta", type=float, default=None,
                    help="override the idiolect strength read from the pool")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = Geolocator()
    if args.theta is None:
        theta, rho = pool_theta()
        print(f"idiolect strength theta = {theta:.5f} "
              f"(realised pooled rho {rho:.3f} in the simulator)")
    else:
        theta, rho = args.theta, float("nan")
        print(f"idiolect strength theta = {theta:.5f} (given)")

    rows, implied, scale = measure(g, theta, n_nodes=args.nodes)
    texts = question_text()
    for r in rows:
        r["text"] = texts.get(r["question"], "")

    print(f"single-factor implied pooled rho = {implied:.3f}; "
          f"scaled by {scale:.3f} to the measured {POOLED_RHO}")

    bits = np.array([r["mi_bits"] for r in rows])
    idio = np.array([r["idiolect_bits"] for r in rows])
    print(f"\n{len(rows)} questions: "
          f"I(A;C) median {np.median(bits):.3f} bits, max {bits.max():.3f}; "
          f"I(A;U) median {np.median(idio):.4f} bits, max {idio.max():.4f}")

    order = sorted(rows, key=lambda r: -r["ratio"])
    print("\nmost geographic (highest regional-signal-to-idiolect ratio)")
    print(f"{'q':>4} {'ratio':>7} {'I(A;C)':>7} {'I(A;U)':>7} {'rho_q':>6}  text")
    for r in order[:12]:
        print(f"{r['question']:>4} {r['ratio']:>7.1f} {r['mi_bits']:>7.3f} "
              f"{r['idiolect_bits']:>7.4f} {r['rho_q']:>6.3f}  "
              f"{r['text'][:70]}")
    print("\nmost personal (lowest ratio)")
    for r in order[-12:]:
        print(f"{r['question']:>4} {r['ratio']:>7.1f} {r['mi_bits']:>7.3f} "
              f"{r['idiolect_bits']:>7.4f} {r['rho_q']:>6.3f}  "
              f"{r['text'][:70]}")

    path = Path(args.out) if args.out else OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r["ratio"]):
            w.writerow({k: r[k] for k in FIELDS})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
