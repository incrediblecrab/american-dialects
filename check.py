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
import json
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
            LITERAL[name] = m.group(1)
    return out


LITERAL = {}
"""The constants exactly as infer.py spells them.

The prose quotes source, so the phrase checks have to compare against the
source text and not against a round-trip through float. RHO = 0 written as
`RHO = 0.0` is the same number and the wrong sentence.
"""


def legacy_rho():
    """The discount that used to be deployed, read out of neural.py.

    RHO is now zero, so the two arms of the site's Act III can no longer be
    "RHO and zero" -- they would be the same picture. The historical value has
    to be named somewhere, and neural.py already names it because the curve
    reports that arm. Read rather than re-typed, for the usual reason.
    """
    src = (ROOT / "model" / "neural.py").read_text(encoding="utf-8")
    m = re.search(r"^LEGACY_RHO = ([0-9.]+)$", src, re.M)
    return float(m.group(1)) if m else None


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
        lit = LITERAL.get("RHO", rho)
        for phrase in [f"`RHO = {lit}` (deployed)", f"runs the Bayes model with `RHO = {lit}`"]:
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
    claim(len(val), 28513, "validation.csv comparison count", ["README.md"], "{:,}", copies=3)
    claim(float(np.corrcoef(a, b)[0, 1]), 0.955, "pixel recovery, pearson r", DOCS, "{:.3f}",
          copies=6)
    # The unit is inside the format string on purpose. Bare "5.6" also matches a
    # slope in findings.md quoted in km per question, which has nothing to do
    # with this error and which was silently absorbed into an earlier copies=2.
    # A pin that counts a coincidence is not pinning the figure it names.
    claim(float(np.abs(a - b).mean()), 5.6, "pixel recovery, mean absolute error",
          ["README.md"], "{:.1f} points", copies=1)

    truth = {(r["state"], r["question"], r["choice"]): float(r["pct"])
             for r in rows("hds/state_pct.csv")}
    derived = {(r["state"], r["question"], r["choice"]): float(r["pct_from_map"]) for r in val}
    tw, dw = _winners(truth), _winners(derived)
    common = [k for k in tw if k in dw]
    agree = sum(1 for k in common if tw[k] == dw[k])
    claim(100.0 * agree / len(common), 87.6, "pixel recovery, modal answer agreement",
          ["README.md"], "{:.1f}", copies=1)


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
          DOCS, "{:,}", copies=6)
    claim(total, 294080, "pop vs soda, source's own total", ["README.md"], "{:,}", copies=2)
    claim(nonzero, 3076, "pop vs soda, counties with responses", ["findings.md"], "{:,}",
          copies=1)
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
          ["README.md", "findings.md"], "{:.4f}", copies=3)
    base = [x for x in t if float(x["sigma"]) == 2.0 and float(x["gamma"]) == 1.0
            and float(x["alpha"]) == 0.0 and x["rake"] == "0"]
    del base  # the 1.1608 no-geography baseline is computed in tune.py, not stored in the grid


# --------------------------------------------------------------- the three-model curve

CURVE = {
    ("net", 5): 654, ("bayes(rho=0)", 5): 685, ("bayes(rho=.177)", 5): 897,
    ("net", 12): 347, ("bayes(rho=0)", 12): 431,
    ("net", 14): 291, ("bayes(rho=0)", 14): 361, ("bayes(rho=.177)", 14): 807,
    ("net", 20): 199, ("bayes(rho=0)", 20): 264,
    ("net", 30): 148, ("bayes(rho=0)", 30): 176, ("bayes(rho=.177)", 30): 903,
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

    peak = min(range(1, 31), key=lambda k: float(by[("bayes(rho=.177)", k)]["median_km"])
               if ("bayes(rho=.177)", k) in by else 1e9)
    dpk, d30 = by.get(("bayes(rho=.177)", peak)), by.get(("bayes(rho=.177)", 30))
    if dpk and d30:
        claim(peak, 13, "the question count where the deployed model bottoms out",
              ["README.md", "findings.md"], "{:.0f}", copies=5)
        claim(float(d30["median_km"]) - float(dpk["median_km"]), 109,
              "deployed model's degradation from its own optimum to k=30",
              ["README.md", "findings.md"], "{:.0f}", copies=4)
    n20, b20 = by.get(("net", 20)), by.get(("bayes(rho=0)", 20))
    if n20 and b20:
        claim(100 * float(n20["state_acc"]), 46.5, "net modal-state accuracy at k=20",
              ["README.md", "findings.md"], "{:.1f}", copies=2)
        claim(100 * float(b20["state_acc"]), 48.0, "bayes modal-state accuracy at k=20",
              ["README.md", "findings.md"], "{:.1f}", copies=2)


# --------------------------------------------------------------- claim plumbing

def occurrences(want, s):
    """Count copies of a formatted number, as a number rather than as a substring.

    Substring matching silently accepts a drifted copy that merely extends a
    correct one: "0.955" is inside "0.9551", so a figure could rot into a
    different figure without failing. The boundaries below require that the
    match is not part of a longer number on either side.
    """
    return len(re.findall(rf"(?<![\d.,]){re.escape(want)}(?!\d|,\d|\.\d)", s))


def claim(computed, quoted, name, docs, fmt, copies=None):
    """Assert a computed value rounds to the quoted one AND that every printed copy agrees.

    `copies` pins how many times the figure is quoted across `docs`. Presence
    alone is too weak: these documents repeat their headline numbers, and a
    check that passes on finding one correct copy cannot see a second copy that
    has drifted away from it. This project has already been bitten by exactly
    that -- the README's worked example turned out to be a second, separately
    computed one that no tracked code produced. Pinning the count means editing
    prose around a validated figure has to be deliberate, which is the same
    bargain the rest of this file already makes.
    """
    shown = fmt.format(computed)
    want = fmt.format(quoted)
    if shown != want:
        record(FAIL, name, f"docs say {want}, data gives {shown}")
        return
    counts = {d: occurrences(want, text(d)) for d in docs}
    missing = [d for d in docs if counts[d] == 0]
    if missing:
        record(FAIL, name, f"value {want} is correct but missing from {', '.join(missing)}")
        return
    n = sum(counts.values())
    if copies is not None and n != copies:
        where = ", ".join(f"{d}x{counts[d]}" for d in docs)
        record(FAIL, name, f"{want} is quoted {n} times across the docs, expected {copies} ({where})")
        return
    record(PASS, name, f"{want} in {', '.join(docs)}" + (f", {n} copies" if copies else ""))


# ------------------------------------------------------------------- the published site

GENERATED = ROOT / "web" / "src" / "content" / "generated.json"
SITE_SRC = ROOT / "web" / "src"


def _flatten(o, prefix=""):
    """Every scalar in the generated JSON, keyed by its path."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _flatten(v, f"{prefix}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield prefix, o


def _check_generated_inventory_is_complete(g):
    """Every data file the README documents must appear in the site's inventory.

    Act V ends by printing `content.inventory.length` as the number of data
    files standing behind the page, so that sentence is a claim about the
    repository made by a file the repository does not regenerate automatically.
    Adding a CSV and its README row while forgetting to re-run the exporter
    leaves the site quietly undercounting its own evidence, which is how it
    read 59 when the README already documented four more mover files. Nothing
    else caught it: the headline-figure comparison above only inspects figures
    it knows the name of, and an inventory that is merely short is still
    internally consistent. Containment is one-directional on purpose -- the
    exporter legitimately carries files the README's row-count table omits,
    such as binary arrays and gzipped cells.
    """
    pat = re.compile(r"^\|\s*`(data/[^`]+\.csv)`\s*\|\s*[\d,]+\s*\|")
    documented = [m.group(1) for m in
                  (pat.match(line) for line in text("README.md").splitlines()) if m]
    exported = {e["path"] for e in g.get("inventory", [])}
    missing = [r for r in documented if r not in exported]
    if not documented:
        record(FAIL, "site inventory covers the README", "no rows matched; table format changed?")
    elif missing:
        record(FAIL, "site inventory covers the README",
               f"{len(missing)} documented file(s) absent from generated.json "
               f"({', '.join(missing[:3])}); re-run model/export_web.py")
    else:
        record(PASS, "site inventory covers the README",
               f"all {len(documented)} documented CSVs present in the site's {len(exported)}")


def check_generated_json():
    """The site's numbers are the data's numbers.

    web/src/content/generated.json is a second surface for figures that already
    live in four documents, which is exactly the drift this script exists to
    prevent. It is written by model/export_web.py from the tracked CSVs, so the
    check is not that the JSON is plausible but that it still equals what those
    CSVs say today. If the exporter has not been re-run since the data changed,
    this fails and the site is stale.
    """
    if not GENERATED.exists():
        record(SKIP, "site, generated.json exists",
               "web/src/content/generated.json not built; run model/export_web.py")
        return None

    g = json.loads(GENERATED.read_text(encoding="utf-8"))

    val = rows("hds/geo/validation.csv")
    a = np.array([float(r["pct_published"]) for r in val])
    b = np.array([float(r["pct_from_map"]) for r in val])
    cats = ["SUMPOP", "SUMSODA", "SUMCOKE", "SUMOTHER"]
    pvs = rows("popvssoda/counties.csv")

    want = {
        "recovery.comparisons": len(val),
        "recovery.r": round(float(np.corrcoef(a, b)[0, 1]), 3),
        "recovery.mae": round(float(np.abs(a - b).mean()), 1),
        "popVsSoda.categorised": sum(sum(int(x[c] or 0) for c in cats) for x in pvs),
        "popVsSoda.sourceTotal": sum(int(x["SUMCOUNT"] or 0) for x in pvs),
        "popVsSoda.counties": sum(1 for x in pvs if int(x["SUMCOUNT"] or 0) > 0),
    }
    const = deployed()
    want["constants.rho"] = float(const["RHO"])
    want["constants.legacyRho"] = legacy_rho()
    want["constants.tauBase"] = float(const["TAU_BASE"])
    want["constants.nQuestions"] = int(const["N_QUESTIONS"])

    flat = dict(_flatten(g))
    bad = [f"{k}: json {flat.get(k)!r}, data {v!r}"
           for k, v in want.items() if flat.get(k) != v]
    if bad:
        record(FAIL, "site, generated.json agrees with the data", "; ".join(bad[:3]))
    else:
        record(PASS, "site, generated.json agrees with the data",
               f"{len(want)} headline figures match; re-run model/export_web.py if they stop")

    _check_generated_inventory_is_complete(g)

    curve = {(r["model"], int(r["k"])): round(float(r["median_km"]), 1)
             for r in rows("model/neural_curve.csv")}
    arms = set(m for m, _ in curve)
    named = list(g.get("models", {}).values())
    missing = [m for m in named if m not in arms]
    if not named:
        record(FAIL, "site, every named curve arm exists",
               "generated.json has no models block; re-run model/export_web.py")
    elif missing:
        record(FAIL, "site, every named curve arm exists",
               f"{', '.join(missing)} named but absent from neural_curve.csv "
               f"(csv has {', '.join(sorted(arms))})")
    else:
        record(PASS, "site, every named curve arm exists",
               f"{len(named)} arms named, all present")
    off = [f"{r['model']} k={r['k']}" for r in g["curve"]
           if curve.get((r["model"], r["k"])) != r["medianKm"]]
    if off:
        record(FAIL, "site, the accuracy curve matches neural_curve.csv",
               f"{len(off)} rows differ, first {off[0]}")
    else:
        record(PASS, "site, the accuracy curve matches neural_curve.csv",
               f"{len(g['curve'])} rows")

    ordered = [r["question"] for r in rows("model/question_order.csv")]
    shipped = [q["id"] for q in g["questions"]]
    diff = next((i for i, q in enumerate(ordered)
                 if i >= len(shipped) or shipped[i] != q), None)
    if diff is not None:
        record(FAIL, "site, the question ordering matches question_order.csv",
               f"first difference at position {diff + 1}: csv says {ordered[diff]}, "
               f"json says {shipped[diff] if diff < len(shipped) else 'nothing'}")
    else:
        record(PASS, "site, the question ordering matches question_order.csv",
               f"{len(shipped)} questions published, {len(ordered)} in the CSV")

    check_docs_quote_the_fixture(g)
    check_docs_quote_the_quantisation(g)
    check_docs_quote_the_isoglosses(g)
    return flat


def check_docs_quote_the_fixture(g):
    """The worked example in the prose must be the worked example the site runs.

    This check exists because the gap it closes was real. The README quoted a
    1,315,932 km2 / 333,927 km2 pair from an adaptively selected twelve-answer
    game, while the site ran a fixed-ordering fixture that no document
    described. Two different worked examples supporting one claim is precisely
    the drift this script is for, and nothing caught it because the figures had
    no computed counterpart to disagree with.
    """
    try:
        v = g["fixture"]["variants"]
        pairs = [(v["deployed"]["area80Km2"], "the corrected 80% area", 2),
                 (v["discounted"]["area80Km2"], "the discounted 80% area", 2)]
    except KeyError as e:
        record(FAIL, "docs quote the site's worked example", f"fixture is missing {e}")
        return
    for area, name, n in pairs:
        claim(area, area, name, ["README.md", "findings.md"], "{:,.0f}", copies=n)


# Numbers a component may legitimately contain: canvas geometry, tick marks,
# radians, array sizes. The rule below therefore protects only figures that are
# distinctive enough that their appearance in a source file is almost certainly
# a quotation rather than a coincidence -- three or more significant digits, or
# a count in the thousands. Small round numbers like the question count are
# guarded separately by check_quiz_length_not_duplicated.
def _distinctive(v):
    if isinstance(v, int) or float(v).is_integer():
        return abs(v) >= 1000
    digits = f"{v!r}".lstrip("-0.").replace(".", "").lstrip("0")
    return len(digits) >= 3


# Positions and indices, which are not claims about anything. A question's
# rank in the ordering and a curve row's k are structure; the medians and
# correlations beside them are the findings.
NOT_A_FIGURE = ("n", "k", "id")
NOT_A_FIGURE_PATH = ("fixture.cell", "topCells", "mapCell", "mapLat", "mapLon")

BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def check_site_quotes_no_typed_numbers(flat):
    """No component may retype a figure that generated.json already carries.

    This is the half of the single-source rule that a generator cannot enforce
    on its own. Emitting r = 0.955 into JSON accomplishes nothing if a section
    also writes 0.955 into its prose, because the second copy is exactly as
    free to go stale as the four in the documents were. The rule is therefore
    mechanical: if a number is in generated.json, the site must read it from
    there, not spell it.

    Astro files are scanned on the same terms as the components. The page is
    now part markup and part island, and a figure typed into a .astro template
    reaches a reader exactly as a figure typed into a .tsx one does, so a scan
    that stopped at TypeScript would have left the whole static half of the
    page unguarded.
    """
    if flat is None:
        record(SKIP, "site, no component retypes a generated figure",
               "generated.json not built")
        return

    figures = {v for k, v in flat.items()
               if _distinctive(v)
               and k.rsplit(".", 1)[-1] not in NOT_A_FIGURE
               and not any(s in k for s in NOT_A_FIGURE_PATH)}
    files = sorted(p for p in SITE_SRC.rglob("*")
                   if p.suffix in (".ts", ".tsx", ".astro"))
    if not files:
        record(SKIP, "site, no component retypes a generated figure", "web/src has no sources")
        return

    hits = []
    for p in files:
        src = BLOCK_COMMENT.sub("", p.read_text(encoding="utf-8"))
        for n, line in enumerate(src.splitlines(), 1):
            code = line.split("//", 1)[0]
            for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?", code):
                v = float(m.group())
                if (int(v) if v.is_integer() else v) in figures:
                    hits.append(f"{p.relative_to(ROOT)}:{n} has {m.group()}")

    if hits:
        record(FAIL, "site, no component retypes a generated figure",
               f"{len(hits)} literal(s), first {hits[0]}")
    else:
        record(PASS, "site, no component retypes a generated figure",
               f"{len(files)} sources scanned against {len(figures)} generated figures")


def check_docs_quote_the_quantisation(g):
    """The int8 payload cost is measured, so the prose must quote the measurement.

    Like the fixture above, this cannot be recomputed here: it is produced by
    model/export_web.py from likelihood.npz, which is gitignored because it is
    too large to track. So the guarantee available is narrower than elsewhere in
    this file -- it asserts that the prose and the artefact the site actually
    ships agree, not that either is right. That is still worth having, because
    the failure it prevents is the one this project has already made once.
    """
    try:
        q = g["quantisation"]
    except KeyError:
        record(FAIL, "docs quote the quantisation cost", "generated.json has no quantisation block")
        return
    # The units are part of the match on purpose. A bare "61.2" is also the
    # coverage figure in an unrelated table further up, and a presence check
    # that a coincidence can satisfy is not a check.
    for key, quoted, fmt, copies in (
            ("games", q["games"], "{:.0f} simulated games", 1),
            ("identicalPct", q["identicalPct"], "{:.1f}% of them", 1),
            ("maxKm", q["maxKm"], "{:.1f} km", 1),
            ("worstRatio", q["worstRatio"], "{:.4f}", 1)):
        claim(q[key], quoted, f"quantisation, {key}", ["findings.md"], fmt, copies=copies)


def check_docs_quote_the_isoglosses(g):
    """The published isogloss widths are a result, so the prose must quote the artefact.

    Same narrower guarantee as the quantisation block above: the widths are
    measured by model/export_web.py from the quantised surfaces the browser
    actually downloads, and cannot be recomputed here. What is enforced is that
    the two extremes the argument rests on -- the sharpest boundary and the one
    barely worth calling a boundary -- are the ones the site ships, and that the
    prose has not quietly swapped which word holds which end.
    """
    try:
        cs = g["isoglosses"]["contrasts"]
    except KeyError:
        record(FAIL, "docs quote the isogloss widths", "generated.json has no isoglosses block")
        return
    lo = min(cs, key=lambda c: c["widthKm"])
    hi = max(cs, key=lambda c: c["widthKm"])
    for c, name, copies in ((lo, "sharpest", 2), (hi, "broadest", 2)):
        claim(c["widthKm"], c["widthKm"], f"isogloss width, {name} ({c['id']})",
              ["README.md", "findings.md"], "{:.0f} km", copies=copies)
        label = c["a"]["label"]
        if label not in text("findings.md"):
            record(FAIL, f"isogloss {name} is named in the prose",
                   f"findings.md never mentions {label}")
        else:
            record(PASS, f"isogloss {name} is named in the prose", label)


def check_the_quiet_ink_is_actually_readable():
    """tokens.css claims --ink-3 clears 4.5:1 on both grounds. Verify the claim.

    The comment beside the token asserts a contrast ratio, which makes it a
    number about the project stated in prose, and every other such number in
    this repository is checked. A one-hex nudge to --paper by somebody
    adjusting the look of the page would otherwise drop the captions and two
    long paragraphs of Act V below AA without anything saying so.

    Only the three text inks are checked, against both paper tones. The rules
    and the de-emphasised labels are deliberately below AA and are argued for
    in the design notes; this asserts what is claimed, not a blanket policy
    nobody agreed to. The page ships one scheme, so this also fails if a dark
    block reappears without its contrast being argued for here.
    """
    css = text("web/src/styles/tokens.css")
    if "prefers-color-scheme" in css:
        record(FAIL, "the quiet ink is readable",
               "tokens.css declares a colour scheme block; this check covers one scheme")
        return

    def hexes(block):
        return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", block))

    def luminance(h):
        ch = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        ch = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]

    def ratio(a, b):
        hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    AA = 4.5
    worst, bad = (None, 99.0), []
    tok = hexes(css)
    missing = [k for k in ("--paper", "--paper-sunk", "--ink", "--ink-2", "--ink-3")
               if k not in tok]
    if missing:
        record(FAIL, "the quiet ink is readable", f"missing {', '.join(missing)}")
        return
    pairs = 0
    for ink in ("--ink", "--ink-2", "--ink-3"):
        for ground in ("--paper", "--paper-sunk"):
            r = ratio(tok[ink], tok[ground])
            pairs += 1
            if r < worst[1]:
                worst = (f"{ink} on {ground}", r)
            if r < AA:
                bad.append(f"{ink} {tok[ink]} on {ground} {tok[ground]} = {r:.2f}")
    if bad:
        record(FAIL, "the quiet ink is readable", "; ".join(bad))
    else:
        record(PASS, "the quiet ink is readable",
               f"all {pairs} pairs clear {AA}:1, tightest {worst[0]} at {worst[1]:.2f}")


def check_readme_states_the_check_count():
    """The README advertises how many checks this file runs. Keep that honest too.

    It was wrong when this was written -- the README said 48 while the suite had
    grown to 58 -- which is a small drift with an outsized moral, because the
    stale sentence was in the paragraph explaining that stale sentences are what
    this script exists to prevent. A count is a number about the project like
    any other, so it gets checked like any other.
    """
    n = len(results) + 1                       # +1 for the check being defined here
    m = re.search(r"check\.py\s+#\s+(\d+) checks", text("README.md"))
    if not m:
        record(FAIL, "README states the check count", "no '# N checks' line found")
    elif int(m.group(1)) != n:
        record(FAIL, "README states the check count",
               f"README says {m.group(1)} checks, this run performed {n}")
    else:
        record(PASS, "README states the check count", f"{n} checks")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show passing checks too")
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
    check_site_quotes_no_typed_numbers(check_generated_json())
    check_the_quiet_ink_is_actually_readable()
    check_readme_states_the_check_count()

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
