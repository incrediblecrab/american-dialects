"""Verify that the prose still matches the data, and that deployed constants live in one place.

The numbers in README.md, findings.md and eli5.md were computed once and typed
in by hand. Nothing stopped them drifting from the files they came from, and
one of them had: the README claimed 12 rows for a crosswalk that has 14.

Worse than a stale row count is a stale recommendation. `RHO`, `TAU_BASE` and
`N_QUESTIONS` are described in prose in four documents. The day one of them is
changed, all four are silently false and the repository has no way to notice.

So this script is the single place that binds a number in the data to the words
in the report. It recomputes each headline claim from the tracked CSVs, checks
the documents still assert it, and checks the documents' description of what is
deployed against what the code actually says. It loads no model and needs no
derived array, so it runs in about a second.

    ./.venv/bin/python check.py           only the checks that can run
    ./.venv/bin/python check.py -v        show every check, not just failures

A claim that cannot be recomputed because a derived file is gitignored is
reported as SKIP, never as a pass.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DOCS = ["README.md", "findings.md", "eli5.md"]

PASS, FAIL, SKIP = "pass", "fail", "skip"
results = []


def record(status, name, detail=""):
    results.append((status, name, detail))


def text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def rows(path):
    with open(DATA / path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------- deployed constants

CONSTANTS = ("RHO", "TAU_BASE", "N_QUESTIONS")


def deployed():
    """Read the deployed constants out of infer.py without importing the model."""
    src = (ROOT / "model" / "infer.py").read_text(encoding="utf-8")
    out = {}
    for name in CONSTANTS:
        m = re.search(rf"^{name} = ([0-9.]+)$", src, re.M)
        if m:
            out[name] = float(m.group(1))
    return out


def check_constants_defined_once(const):
    """Each deployed constant must be assigned at module level in exactly one file."""
    for name in CONSTANTS:
        if name not in const:
            record(FAIL, f"{name} is defined", "no module-level assignment in model/infer.py")
            continue
        hits = []
        for py in sorted(ROOT.glob("*/*.py")):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if re.match(rf"^{name} = ", line):
                    hits.append(f"{py.relative_to(ROOT)}:{i}")
        if hits == [f"model/infer.py:{_lineno('model/infer.py', name)}"]:
            record(PASS, f"{name} defined in exactly one place", hits[0])
        else:
            record(FAIL, f"{name} defined in exactly one place", f"found {len(hits)}: {hits}")


def _lineno(rel, name):
    for i, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
        if re.match(rf"^{name} = ", line):
            return i
    return -1


def check_quiz_length_not_duplicated(const):
    """The quiz length must not survive as a bare literal anywhere it is used.

    It used to appear four times: quiz.py's argparse default, server.py's
    /api/start fallback, and twice in app.js. The browser now takes the length
    from the server's reply, so the number exists once.
    """
    n = const.get("N_QUESTIONS")
    if n is None:
        record(SKIP, "quiz length is not hardcoded", "N_QUESTIONS not found")
        return
    lit = str(int(n))
    offenders = []
    for rel, pats in [("model/quiz.py", [rf"default={lit}\b"]),
                      ("site/server.py", [rf'data\.get\("n"\) or {lit}\b']),
                      ("site/static/app.js", [rf"\bn:\s*{lit}\b", rf"n:\s*{lit}\s*\}}"])]:
        src = (ROOT / rel).read_text(encoding="utf-8")
        for p in pats:
            if re.search(p, src):
                offenders.append(f"{rel} ~ /{p}/")
    if offenders:
        record(FAIL, "quiz length is not hardcoded", f"literal {lit} still in {offenders}")
    else:
        record(PASS, "quiz length is not hardcoded",
               f"quiz.py, server.py and app.js all derive from N_QUESTIONS = {lit}")


def check_docs_describe_deployed(const):
    """What the prose says is deployed must be what infer.py says is deployed."""
    rho, n = const.get("RHO"), const.get("N_QUESTIONS")
    if rho is not None:
        want = f"RHO = {rho:.3f}".rstrip("0")
        blob = "\n".join(text(d) for d in DOCS)
        for phrase in [f"`RHO = {rho}` (deployed)", f"runs the Bayes model with `RHO = {rho}`"]:
            if phrase not in blob:
                record(FAIL, "docs describe the deployed RHO",
                       f"expected to find {phrase!r}; if RHO changed, the docs did not")
                break
        else:
            record(PASS, "docs describe the deployed RHO", f"RHO = {rho} in code and in prose")
        del want
    if n is not None:
        n_int = int(n)
        word = {12: "twelve", 13: "thirteen", 14: "fourteen"}.get(n_int, str(n_int))
        readme = text("README.md")
        if f"asks a hardcoded {word}" in readme:
            record(PASS, "README describes the deployed quiz length",
                   f"N_QUESTIONS = {n_int}, README says 'a hardcoded {word}'")
        else:
            record(FAIL, "README describes the deployed quiz length",
                   f"N_QUESTIONS = {n_int} but README does not say 'asks a hardcoded {word}'")


# --------------------------------------------------------------- README inventory table

def check_readme_table():
    """Every plain row count in the README's 'what's here' table, against the file.

    The table is the claim; the filesystem is the truth. Rows whose count is not
    a plain integer (parameter counts, grid shapes) are checked elsewhere or not
    at all, and rows whose file is a gitignored derived array are skipped.
    """
    pat = re.compile(r"^\|\s*`(data/[^`]+\.csv)`\s*\|\s*([\d,]+)\s*\|")
    seen = 0
    for line in text("README.md").splitlines():
        m = pat.match(line)
        if not m:
            continue
        rel, claimed = m.group(1), int(m.group(2).replace(",", ""))
        path = ROOT / rel
        if not path.exists():
            record(SKIP, f"{rel} row count", "file not present (derived or gitignored)")
            continue
        with open(path, encoding="utf-8", newline="") as f:
            actual = sum(1 for _ in f) - 1
        seen += 1
        if actual == claimed:
            record(PASS, f"{rel} row count", f"{actual:,}")
        else:
            record(FAIL, f"{rel} row count", f"README says {claimed:,}, file has {actual:,}")
    if not seen:
        record(FAIL, "README inventory table", "no row-count rows matched; table format changed?")


# --------------------------------------------------------------- recovered geography

def check_pixel_recovery():
    """r, MAE and modal agreement of the recovered surfaces against published state percentages.

    The modal comparison has a subtlety that cost an hour to rediscover, so it
    is written down here. The truth winner must be taken over the FULL
    state_pct.csv, not over validation.csv. validation.csv only carries choices
    that had a plot colour, and 17 choices did not, so a state-question whose
    true winner has no map would otherwise be scored against the wrong winner.
    Taking truth from validation.csv gives 88.2%; the honest number is 87.6%.
    """
    val = rows("hds/geo/validation.csv")
    a = np.array([float(r["pct_published"]) for r in val])
    b = np.array([float(r["pct_from_map"]) for r in val])
    claim(len(val), 28513, "validation.csv comparison count", ["README.md"], "{:,}")
    claim(float(np.corrcoef(a, b)[0, 1]), 0.955, "pixel recovery, pearson r", DOCS, "{:.3f}")
    claim(float(np.abs(a - b).mean()), 5.6, "pixel recovery, mean absolute error",
          ["README.md", "findings.md"], "{:.1f}")

    truth = {(r["state"], r["question"], r["choice"]): float(r["pct"])
             for r in rows("hds/state_pct.csv")}
    derived = {(r["state"], r["question"], r["choice"]): float(r["pct_from_map"]) for r in val}
    tw, dw = _winners(truth), _winners(derived)
    common = [k for k in tw if k in dw]
    agree = sum(1 for k in common if tw[k] == dw[k])
    claim(100.0 * agree / len(common), 87.6, "pixel recovery, modal answer agreement",
          ["README.md"], "{:.1f}")


def _winners(d):
    best = {}
    for (s, q, c), v in d.items():
        if best.get((s, q), (None, -1.0))[1] < v:
            best[(s, q)] = (c, v)
    return {k: v[0] for k, v in best.items()}


# --------------------------------------------------------------- external tuning target

def check_popvssoda():
    """Pop vs. Soda totals, and the off-by-one that makes two different totals both correct.

    The source reports SUMCOUNT = 294,080 but its four category columns sum to
    294,079, because Lawrence County, Ohio has SUMCOUNT = 71 against 70
    categorised responses. tune.py scores the categories, so 294,079 is the
    number the model was fitted on and 294,080 is the collection's own total.
    Both appear in the README and both are right.
    """
    r = rows("popvssoda/counties.csv")
    cats = ["SUMPOP", "SUMSODA", "SUMCOKE", "SUMOTHER"]
    parts = sum(sum(int(x[c] or 0) for c in cats) for x in r)
    total = sum(int(x["SUMCOUNT"] or 0) for x in r)
    nonzero = sum(1 for x in r if int(x["SUMCOUNT"] or 0) > 0)
    claim(parts, 294079, "pop vs soda, categorised responses (what tune.py scores)",
          DOCS, "{:,}")
    claim(total, 294080, "pop vs soda, source's own total", ["README.md"], "{:,}")
    claim(nonzero, 3076, "pop vs soda, counties with responses", ["findings.md"], "{:,}")
    if total - parts != 1:
        record(FAIL, "pop vs soda, the documented off-by-one",
               f"expected the two totals to differ by exactly 1, they differ by {total - parts}")
    else:
        record(PASS, "pop vs soda, the documented off-by-one", "294,080 - 294,079 = 1, as documented")


def check_tuning():
    """Log-loss at the frozen settings, not at the grid minimum.

    tensor.py freezes sigma=8, gamma=1.5, alpha=0.02, box=9, rake=True, and the
    reported 0.7234 is that row. The grid actually dips to 0.71826 at rake=0,
    which is NOT the deployed configuration and must not be quoted: raking is
    kept because it fixes the level that the dot maps cannot supply.
    """
    t = rows("model/tuning.csv")
    frozen = [x for x in t
              if x["rake"] == "1" and float(x["sigma"]) == 8.0
              and float(x["gamma"]) == 1.5 and float(x["alpha"]) == 0.02]
    if len(frozen) != 1:
        record(FAIL, "tuning, frozen settings row",
               f"expected exactly one row at sigma=8 gamma=1.5 alpha=0.02 rake=1, found {len(frozen)}")
        return
    claim(float(frozen[0]["logloss"]), 0.7234, "log-loss at the frozen settings",
          ["README.md", "findings.md"], "{:.4f}")
    base = [x for x in t if float(x["sigma"]) == 2.0 and float(x["gamma"]) == 1.0
            and float(x["alpha"]) == 0.0 and x["rake"] == "0"]
    del base  # the 1.1608 no-geography baseline is computed in tune.py, not stored in the grid


# --------------------------------------------------------------- the three-model curve

CURVE = {
    ("net", 5): 654, ("bayes(rho=0)", 5): 685, ("bayes(rho=.177)", 5): 897,
    ("net", 12): 343, ("bayes(rho=0)", 12): 444, ("bayes(rho=.177)", 12): 760,
    ("net", 14): 281, ("bayes(rho=0)", 14): 384, ("bayes(rho=.177)", 14): 789,
    ("net", 20): 199, ("bayes(rho=0)", 20): 264, ("bayes(rho=.177)", 20): 847,
}


def check_neural_curve():
    """Every median error quoted in the README and findings tables."""
    by = {(r["model"], int(r["k"])): r for r in rows("model/neural_curve.csv")}
    for (model, k), want in sorted(CURVE.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        r = by.get((model, k))
        if r is None:
            record(SKIP, f"curve {model} k={k}", "row missing from neural_curve.csv")
            continue
        got = round(float(r["median_km"]))
        docs = [d for d in DOCS if str(want) in text(d)]
        if got != want:
            record(FAIL, f"curve {model} k={k}", f"docs say {want} km, csv has {got} km")
        elif not docs:
            record(FAIL, f"curve {model} k={k}", f"{want} km computed but quoted in no document")
        else:
            record(PASS, f"curve {model} k={k}", f"{got} km, in {', '.join(docs)}")

    d12 = by.get(("bayes(rho=.177)", 12))
    d20 = by.get(("bayes(rho=.177)", 20))
    if d12 and d20:
        claim(float(d20["median_km"]) - float(d12["median_km"]), 88,
              "deployed model's degradation from k=12 to k=20", ["README.md", "findings.md"], "{:.0f}")
    n20, b20 = by.get(("net", 20)), by.get(("bayes(rho=0)", 20))
    if n20 and b20:
        claim(100 * float(n20["state_acc"]), 46.5, "net modal-state accuracy at k=20",
              ["README.md", "findings.md"], "{:.1f}")
        claim(100 * float(b20["state_acc"]), 48.0, "bayes modal-state accuracy at k=20",
              ["README.md", "findings.md"], "{:.1f}")


# --------------------------------------------------------------- claim plumbing

def claim(computed, quoted, name, docs, fmt):
    """Assert a computed value rounds to the quoted one AND still appears in the prose."""
    shown = fmt.format(computed)
    want = fmt.format(quoted)
    if shown != want:
        record(FAIL, name, f"docs say {want}, data gives {shown}")
        return
    missing = [d for d in docs if want not in text(d)]
    if missing:
        record(FAIL, name, f"value {want} is correct but missing from {', '.join(missing)}")
    else:
        record(PASS, name, f"{want} in {', '.join(docs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true", help="show passing checks too")
    args = ap.parse_args()

    const = deployed()
    check_constants_defined_once(const)
    check_quiz_length_not_duplicated(const)
    check_docs_describe_deployed(const)
    check_readme_table()
    check_pixel_recovery()
    check_popvssoda()
    check_tuning()
    check_neural_curve()

    width = max(len(n) for _, n, _ in results)
    for status, name, detail in results:
        if status == PASS and not args.verbose:
            continue
        mark = {PASS: "ok  ", FAIL: "FAIL", SKIP: "skip"}[status]
        print(f"{mark}  {name:<{width}}  {detail}")

    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_skip = sum(1 for s, _, _ in results if s == SKIP)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nA failure means the prose and the data disagree. Fix whichever is wrong;\n"
              "do not edit this file to make it pass.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
