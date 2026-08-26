"""Two follow-up questions about the Harvard/Cambridge agreement.

`cambridge_validate.py` compares the two independently recovered surfaces on the
25 question pairs whose answer texts match at high confidence. That is the right
headline, but it leaves two things unanswered that decide how much the model can
actually claim.

First, coverage of the deployed question set. The geolocator asks a specific
greedy-mutual-information ordering of twenty questions and recommends stopping
at twelve. The agreement analysis is the only independent-population check these
surfaces have, so what matters is not how many questions were corroborated but
WHICH. Twenty-five corroborated questions are worth little if none of them are
the twelve being asked.

Second, whether the corroboration survives outside the easy lexical items. The
high-confidence tier is lexical almost by construction, because lexical answers
are the ones with distinct texts that match across surveys and the ones that get
the cleanly recoverable primary colours. The 382 medium-confidence rows reach
109 questions, most of them phonetic. If the margin over chance collapses there,
the honest claim is narrower: the lexical surfaces are corroborated, not the
surfaces.

Writes data/cambridge/hds_agreement_extended.md. Deliberately does not touch
hds_agreement.md, which cambridge_validate.py owns and rewrites wholesale.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402
from validate_geo import rasterise_states  # noqa: E402
from cambridge_validate import (  # noqa: E402
    load_grid, answer_index_maps, shares,
)

DEPLOYED = ["105", "74", "95", "99", "106", "110", "79", "64", "118", "103",
            "73", "58", "76", "66", "50", "65", "84", "59", "83", "60"]
ASK = 12

PRIMARY = {"#ff0000", "#00ff00", "#0000ff"}


def question_class(q):
    """Harvard's own three blocks, which are contiguous by question number.

    q1-q48 are pronunciation items, q49-q57 the syntax block, q58-q122 lexical.
    """
    n = int(q)
    if n <= 48:
        return "phonetic"
    if n <= 57:
        return "syntactic"
    return "lexical"


def compare(cid, hq, pairs, cmb_den, cmb_idx, hds_den, hds_idx,
            cmb_txt, hds_txt, cmb_color, land):
    """Agreement for one question pair, identical in method to the headline."""
    cl, hl, prim = [], [], []
    for cch, hch in pairs:
        ct = cmb_txt.get(cid, {}).get(cch)
        ht = hds_txt.get(hq, {}).get(hch)
        if ct is None or ht is None:
            continue
        if cid in cmb_idx and ct in cmb_idx[cid] \
           and hq in hds_idx and ht in hds_idx[hq]:
            cl.append(ct)
            hl.append(ht)
            prim.append(cmb_color[cid].get(ct) in PRIMARY)
    if len(cl) < 2:
        return None, "fewer than two answers recoverable in both surveys"

    csh, ctot = shares(cmb_den, cmb_idx, cid, cl, land)
    hsh, htot = shares(hds_den, hds_idx, hq, hl, land)
    ok = (ctot > 0) & (htot > 0)
    if ok.sum() < 20:
        return None, "fewer than 20 cells with density in both surveys"
    csh, hsh = csh[:, ok], hsh[:, ok]
    w = np.minimum(ctot[ok], htot[ok])

    rs, maes = [], []
    for j in range(len(cl)):
        a, b = hsh[j], csh[j]
        if a.std() > 0 and b.std() > 0:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
        maes.append(float(np.abs(a - b).mean()))

    cm, hm = csh.argmax(0), hsh.argmax(0)
    agree = cm == hm
    chance = sum((hm == j).mean() * (cm == j).mean() for j in range(csh.shape[0]))

    pmodal = float("nan")
    pj = [j for j, p in enumerate(prim) if p]
    if len(pj) >= 2:
        pc, ph = csh[pj], hsh[pj]
        pmodal = float((pc.argmax(0) == ph.argmax(0)).mean())

    return {
        "answers": len(cl), "cells": int(ok.sum()),
        "r": float(np.mean(rs)) if rs else float("nan"),
        "mae": float(np.mean(maes)),
        "modal": float(agree.mean()), "chance": float(chance),
        "modal_primary": pmodal,
        "shares_h": hsh, "shares_c": csh, "w": w, "agree": agree,
    }, None


def main():
    hds_den, hds_idx, lats, lons = load_grid(DATA / "hds" / "geo" / "grid.npz")
    cmb_den, cmb_idx, _, _ = load_grid(DATA / "cambridge" / "geo" / "grid.npz")
    hds_txt, cmb_txt = answer_index_maps()

    cmb_color = defaultdict(dict)
    for r in csv.DictReader(open(DATA / "cambridge" / "answers.csv",
                                 encoding="utf-8")):
        ch = "abcdefghijklmnop"[int(r["choice_index"]) - 1]
        cmb_color[r["id"]][ch] = r["color_hex"]

    land = rasterise_states(lats, lons) != ""

    tiers = defaultdict(lambda: defaultdict(list))
    notes = {}
    for r in csv.DictReader(open(DATA / "cambridge" / "hds_crosswalk.csv",
                                 encoding="utf-8")):
        conf = r["confidence"]
        if conf in ("high", "medium"):
            tiers[conf][(r["cambridge_id"], r["hds_question"])].append(
                (r["cambridge_choice"], r["hds_choice"]))
        notes.setdefault((r["hds_question"], conf), r.get("note", ""))

    def run(tier):
        out = {}
        for (cid, hq), pairs in tiers[tier].items():
            res, why = compare(cid, hq, pairs, cmb_den, cmb_idx,
                               hds_den, hds_idx, cmb_txt, hds_txt,
                               cmb_color, land)
            out[hq] = (res, why, cid)
        return out

    high, med = run("high"), run("medium")

    L = ["# Harvard vs Cambridge: coverage of the deployed questions, and the "
         "medium tier\n",
         "Companion to `hds_agreement.md`, which owns the headline "
         "high-confidence numbers and is left untouched. Method is identical: "
         "same grid, same renormalisation over mapped answers, same "
         "chance baseline computed from the two modal maps' own marginals.\n"]

    # ---- Task 1 ---------------------------------------------------------
    L.append("## Coverage of the twenty deployed questions\n")
    L.append("In the order the geolocator asks them. The recommendation is to "
             f"stop at {ASK}, so the first {ASK} rows are the ones that "
             "decide what can be claimed.\n")
    L.append("| # | HDS | tier | answers | cells | mean r | MAE (pp) | modal | "
             "chance | margin | modal (primary) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")

    covered = margins = 0
    first12 = []
    for i, q in enumerate(DEPLOYED, 1):
        res, why, cid = high.get(q, (None, None, None))
        tier = "high"
        if res is None:
            res, why, cid = med.get(q, (None, "no crosswalk row", None))
            tier = "medium" if res is not None else "none"
        star = "**" if i <= ASK else ""
        if res is None:
            L.append(f"| {star}{i}{star} | q{q} | — | — | — | — | — | — | — | "
                     f"— | — | {why} |")
            continue
        m = res["modal"] - res["chance"]
        if i <= ASK:
            first12.append(m)
        if tier == "high":
            covered += 1
        margins += 1
        pm = "—" if np.isnan(res["modal_primary"]) else f"{res['modal_primary']:.0%}"
        L.append(f"| {star}{i}{star} | q{q} | {tier} | {res['answers']} | "
                 f"{res['cells']} | {res['r']:.2f} | {res['mae']:.1f} | "
                 f"{res['modal']:.0%} | {res['chance']:.0%} | "
                 f"{m:+.0%} | {pm} |")

    L.append("")
    L.append(f"Of the twenty deployed questions, **{covered} are in the "
             f"high-confidence tier**. Mean margin over chance across the "
             f"first {ASK}: **{np.mean(first12):+.1%}** "
             f"(min {np.min(first12):+.0%}, max {np.max(first12):+.0%}).\n")

    # ---- Task 2 ---------------------------------------------------------
    L.append("## The medium tier, by question class\n")
    L.append("Never pooled with the high tier. Harvard's question numbering is "
             "contiguous by type, so the split is q1-48 pronunciation, q49-57 "
             "syntax, q58-122 lexical.\n")
    L.append("| tier | class | questions | cell comparisons | pooled r | "
             "modal | chance | margin |")
    L.append("|---|---|---|---|---|---|---|---|")

    def summarise(tier_name, results, cls=None):
        qs = [(q, r) for q, (r, _, _) in results.items()
              if r is not None and (cls is None or question_class(q) == cls)]
        if not qs:
            return None
        a = np.concatenate([r["shares_h"].ravel() for _, r in qs])
        b = np.concatenate([r["shares_c"].ravel() for _, r in qs])
        n = sum(r["agree"].size for _, r in qs)
        modal = sum(float(r["agree"].sum()) for _, r in qs) / n
        chance = sum(r["chance"] * r["agree"].size for _, r in qs) / n
        r_pool = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 \
            else float("nan")
        L.append(f"| {tier_name} | {cls or 'all'} | {len(qs)} | {a.size:,} | "
                 f"{r_pool:.3f} | {modal:.0%} | {chance:.0%} | "
                 f"{modal - chance:+.0%} |")
        return modal - chance

    summarise("high", high)
    out = {}
    for cls in ("lexical", "phonetic", "syntactic"):
        out[cls] = summarise("medium", med, cls)
    summarise("medium", med)

    L.append("")
    lex, pho = out.get("lexical"), out.get("phonetic")
    if lex is not None and pho is not None:
        L.append(
            f"The margin over chance is **{lex:+.0%} on medium-tier lexical "
            f"items** and **{pho:+.0%} on phonetic items**. "
            + ("The corroboration does not survive outside the lexical items, "
               "so the defensible claim is that the LEXICAL surfaces are "
               "corroborated, not the surfaces in general. That is the right "
               "scope anyway: the deployed question set is lexical throughout."
               if pho < 0.5 * lex else
               "The margin holds across both classes, so the corroboration is "
               "not an artefact of easy lexical items."))
    L.append("")

    path = DATA / "cambridge" / "hds_agreement_extended.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
