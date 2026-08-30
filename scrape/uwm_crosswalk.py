"""Build the UWM→HDS question crosswalk.

Matches UWM question text to HDS question text using normalised
difflib.SequenceMatcher similarity, augmented by a containment bonus for the
many HDS entries that are single-word or short-phrase keyword descriptions
(e.g. HDS q4 "caramel" → UWM q102 "How do you pronounce caramel?").

Output: data/uwm/hds_crosswalk.csv
Columns: uwm_question, hds_question, ratio, confidence, method, uwm_text, hds_text
"""

import csv
import difflib
import re
from pathlib import Path

from common import DATA

STOPWORDS = frozenset(
    "a an the do does did how what which would you say your use call"
    " is are was were be been have has had do does that this"
    " i me my we our they them their it its of in on to for"
    " with at from as by or and if not no yes s"
    " one place any something someone".split()
)

# Manually verified false positives from the containment algorithm:
# single-word HDS keywords that happened to appear in unrelated UWM questions.
KNOWN_FALSE = {
    "22": "HDS 'poem' matched UWM q123 because 'poem' appears in an example "
          "sentence; UWM q123 asks about 'used to could' not about 'poem'.",
    "23": "HDS 'really' matched UWM q343 because 'really' appears as an adverb; "
          "UWM q343 asks about a rubber ball.",
    "36": "HDS 'the c in grocery' matched UWM q346 ('supermarket vs grocery "
          "store'); 'grocery' is shared but the questions differ.",
    "40": "HDS 'quarter' matched UWM q391 ('running/fitness activity'); 'quarter' "
          "appears in context but the questions are unrelated.",
    "95": "HDS 'What is the City?' matched UWM q336 ('passageway between buildings "
          "in the city'); 'city' is shared but in a different context.",
    "97": "HDS 'Which of these terms do you prefer?' matched UWM q436 via generic "
          "phrase; the two questions are about unrelated topics.",
    "98": "HDS 'Which of these terms do you prefer?' matched UWM q436 via same "
          "generic phrase; unrelated questions.",
    "28": "HDS q28 'Do you pronounce cot and caught the same?' matched UWM q25 "
          "'Do you pronounce which and witch the same?' via structural template; "
          "different phoneme contrasts (cot/caught vs which/witch).",
}


def norm(text):
    """Lowercase, strip punctuation, collapse whitespace, remove stopwords."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t and t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def _contained(short_norm, long_norm):
    """True if every token in short_norm appears somewhere in long_norm."""
    if not short_norm:
        return False
    st = set(short_norm.split())
    lt = set(long_norm.split())
    return st <= lt


def confidence_band(ratio, hds_q):
    if hds_q in KNOWN_FALSE:
        return "reject"
    if ratio >= 0.99:
        return "verbatim"
    if ratio >= 0.85:
        return "high"
    if ratio >= 0.70:
        return "medium"
    return "low"


def match_hds_to_uwm(hds_rows, uwm_rows):
    """For each HDS question, find the best-matching UWM question."""
    uwm_norm = [(r["question"], r["text"], norm(r["text"])) for r in uwm_rows]
    results = []
    for h in hds_rows:
        hq = h["question"]
        ht = h["text"]
        hn = norm(ht)
        if not hn:
            continue

        best_ratio = -1.0
        best_uq = None
        best_ut = ""
        best_method = "sm"

        for uq, ut, un in uwm_norm:
            if not un:
                continue
            r = _ratio(hn, un)
            contained = _contained(hn, un)
            eff = max(r, 0.82 if contained else 0.0)
            if eff > best_ratio:
                best_ratio = eff
                best_uq = uq
                best_ut = ut
                best_method = "contain" if (contained and 0.82 > r) else "sm"

        if best_uq is not None:
            results.append({
                "hds_question": hq,
                "uwm_question": best_uq,
                "ratio": round(best_ratio, 4),
                "confidence": confidence_band(best_ratio, hq),
                "method": best_method,
                "hds_text": ht,
                "uwm_text": best_ut,
            })
    return results


def main():
    d = DATA / "uwm"
    d.mkdir(parents=True, exist_ok=True)

    hds_rows = list(csv.DictReader(
        open(DATA / "hds" / "questions.csv", encoding="utf-8")))
    uwm_rows = list(csv.DictReader(
        open(d / "questions.csv", encoding="utf-8")))

    print(f"HDS: {len(hds_rows)} questions")
    print(f"UWM: {len(uwm_rows)} questions")

    matches = match_hds_to_uwm(hds_rows, uwm_rows)
    matches.sort(key=lambda r: -r["ratio"])

    for thresh in [0.99, 0.95, 0.85, 0.70]:
        raw = sum(1 for m in matches if m["ratio"] >= thresh)
        clean = sum(1 for m in matches if m["ratio"] >= thresh
                    and m["confidence"] != "reject")
        print(f"  ≥{thresh:.2f}: {clean} valid matches "
              f"({raw} before removing {len(KNOWN_FALSE)} known false positives)")

    out_path = d / "hds_crosswalk.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["uwm_question", "hds_question", "ratio",
                               "confidence", "method", "uwm_text", "hds_text"])
        w.writeheader()
        for m in sorted(matches, key=lambda r: (int(r["hds_question"]), -r["ratio"])):
            w.writerow(m)

    print(f"\nwrote {out_path}: {len(matches)} rows")


if __name__ == "__main__":
    main()
