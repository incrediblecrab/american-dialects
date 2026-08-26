"""Map Cambridge survey questions/answers onto Harvard Dialect Survey ones.

The single most important structural fact about the Cambridge survey is that its
question IDs 241-362 are a near-verbatim re-run of the *entire* Harvard Dialect
Survey (HDS q1-q122), in the same order: Cambridge id == HDS question + 240
(C241 "aunt" == HDS q1, ... C362 "especially" == HDS q122; C240 "Pop or soda?"
is an extra). These share wording and answer sets with HDS but were answered by
a different, later population, which is exactly the independent replication we
want. Crucially they are all *single-select*, unlike the survey's two featured
multi-select items (C1 sandwich, C2 carbonated beverage).

The survey also has a smaller "featured" block (low IDs, e.g. C1, C2, C7, C13,
C20, ...) that re-asks some of the same lexical items with reworded answer sets
(and, for C1/C2, "select all that apply"). We map those too but at lower
confidence, and flag the multi-select ones prominently.

We match answers by TEXT, not by choice index/letter, because the two surveys
order and colour their choices differently (HDS uses ~11 X11 colour names; the
Cambridge tiles use only 7 fixed colours, with the three primaries reserved for
the top-3 answers). Output: data/cambridge/hds_crosswalk.csv with columns
cambridge_id, cambridge_choice, hds_question, hds_choice, confidence, note.
"""

import csv
import re
from collections import defaultdict

from common import DATA


def norm(s):
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)          # drop parentheticals
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Manual (cambridge_answer_norm -> hds_answer_norm) fixes where text differs but
# the meaning is the same. Keyed loosely on the normalised Cambridge text.
ALIAS = {
    "you guys": "you guys", "y all ya ll": "y all", "y all": "y all",
    "crawdad crawdaddy": "crawdad", "crawdad": "crawdad",
    "roly poly rollie pollie rolly polly": "roly poly", "roly poly": "roly poly",
    "pillbug": "pill bug", "pill bug": "pill bug",
    "i use lightning bug and firefly interchangeably":
        "i use lightning bug and firefly interchangeably",
    "milkshake shake": "milkshake shake",
    "median strip": "median strip", "median": "median",
    "kitty corner": "kitty corner", "catty corner": "catty corner",
    "catercorner": "catercorner",
    "the devil is beating his wife the devil is whipping his wife":
        "the devil is beating his wife",
    "the devil is beating his wife": "the devil is beating his wife",
    "goose bumps": "goose bumps", "goose pimples": "goose pimples",
    "trash can": "trash can", "garbage can": "garbage can",
}

# Meta / non-lexical Cambridge answers that carry no comparable HDS choice.
DROP = {"other", "", "i have no word for this", "i have no word for this critter",
        "i have no term for this", "i have no idea", "i have no special word or "
        "phrase for this", "i have no term or expression for this",
        "i don t know this creature", "i have no word for this creature",
        "i have no specific term for this", "i have no special term for this"}


def load():
    camb_q = {r["id"]: r for r in csv.DictReader(
        open(DATA / "cambridge" / "questions.csv", encoding="utf-8"))}
    camb_a = defaultdict(list)
    for r in csv.DictReader(open(DATA / "cambridge" / "answers.csv",
                                 encoding="utf-8")):
        camb_a[r["id"]].append(r)
    hds_a = defaultdict(list)
    for r in csv.DictReader(open(DATA / "hds" / "answers.csv", encoding="utf-8")):
        hds_a[r["question"]].append(r)
    return camb_q, camb_a, hds_a


def match_choice(camb_ans, hds_rows):
    """Best HDS answer row for a Cambridge answer, by normalised text."""
    cn = norm(camb_ans)
    cn = ALIAS.get(cn, cn)
    if cn in DROP:
        return None
    # exact / alias
    for h in hds_rows:
        if norm(h["answer"]) == cn:
            return h
    # first-token / substring
    ctok = cn.split()
    for h in hds_rows:
        hn = norm(h["answer"])
        if not hn:
            continue
        if hn == cn or hn.startswith(cn) or cn.startswith(hn):
            return h
        if ctok and hn.split()[:1] == ctok[:1] and len(ctok[0]) > 3:
            return h
    return None


# Curated question-level mappings. Each entry: (cambridge_id, hds_q, confidence,
# note). The verbatim block is generated below; these are the ones we single out
# with tailored confidence/notes, plus the featured-block duplicates.
FEATURED = [
    ("1", "64", "medium", "featured MULTI-SELECT sandwich; % are share-of-"
     "respondents-selecting, not mutually exclusive -- not comparable to HDS "
     "single-select shares. Use verbatim C304 instead for validation."),
    ("2", "105", "medium", "featured MULTI-SELECT carbonated beverage; co-"
     "selected answers co-locate as overlapping dots and alias to secondary "
     "colours. Use verbatim C345 for validation."),
    ("7", "73", "medium", "featured sneakers; answer set reworded vs HDS q73. "
     "Verbatim twin is C313."),
    ("13", "66", "medium", "featured crawfish; verbatim twin C306."),
    ("20", "50", "medium", "featured address-a-group; adds 'all y'all','you "
     "people'; verbatim twin C290."),
    ("21", "74", "medium", "featured roly-poly; verbatim twin C314."),
    ("30", "80", "medium", "featured sunshower; verbatim twin C320."),
    ("31", "82", "medium", "featured eye-matter; verbatim twin C322."),
    ("68", "58", "medium", "featured yard/garage sale; verbatim twin C298."),
    ("69", "75", "medium", "featured shopping cart; verbatim twin C315."),
    ("70", "84", "medium", "featured roundabout/traffic circle; verbatim twin "
     "C324."),
    ("76", "96", "medium", "featured dinner/supper; verbatim twin C336."),
    ("78", "-", "reject", "upholstered seat couch/sofa: no HDS counterpart."),
    ("79", "118", "low", "featured liquor store; HDS q118 is a *drive-through* "
     "liquor store specifically -- only partially comparable."),
    ("84", "119", "medium", "featured take-out; verbatim twin C359."),
    ("240", "105", "medium", "extra 'Pop or soda?' forced-choice; narrower "
     "answer set than HDS q105. Verbatim twin is C345."),
]

# Featured questions with NO usable HDS counterpart (UK-focused or novel).
REJECT = [
    ("67", "aunt-vowel: recoded into 7 fine phonetic distinctions; HDS q1 codes "
     "aunt differently. Verbatim twin C241 is the clean map."),
    ("71", "rhoticity (pronounce r in car/cart): not an HDS question."),
    ("72", "linking/intrusive r in 'sawing': not an HDS question."),
    ("73", "intrusive r in 'blah-ish': not an HDS question."),
    ("74", "intrusive r in 'Shah of': not an HDS question."),
    ("75", "which/witch (wine-whine merger): not an HDS question."),
    ("77", "queue vs line: UK-oriented, verbatim twin C333 is the map."),
    ("80", "restroom/bathroom/toilet: no HDS counterpart."),
    ("81", "emergency/parking brake: no HDS counterpart."),
    ("82", "stick-shift/manual: no HDS counterpart."),
    ("83", "pacifier/binky: no HDS counterpart."),
    ("85", "waterbug/water strider: verbatim twin C342 maps to HDS q102."),
    ("86", "alley/alleyway: no HDS counterpart."),
    ("87", "ATM/cash machine: no HDS counterpart."),
    ("88", "pinkie toe/little toe: no HDS counterpart."),
    ("89", "green onion/scallion: no HDS counterpart."),
    ("90", "scone pronunciation: UK-oriented, no HDS counterpart."),
    ("91", "roll/bun: no HDS counterpart."),
    ("92", "last vowel (TRAP-BATH): UK-oriented, no HDS counterpart."),
    ("93", "ginnel/snicket: UK-only, ~0% US signal."),
    ("94", "chav/townie: UK-only."),
    ("95", "aitch/haitch: UK-oriented."),
    ("96", "underwear/knickers: no HDS counterpart."),
    ("99", "schedule sk/sh: UK-oriented."),
    ("100", "pants/trousers: UK-oriented."),
    ("104", "chip butty: UK-only."),
    ("106", "garbage man/bin man: bin man is UK; no clean HDS counterpart."),
    ("110", "drying rack/clothes horse: no HDS counterpart."),
    ("112", "soccer/football: no HDS counterpart."),
    ("114", "truce/pax/barley: UK-oriented, no HDS counterpart."),
    ("115", "woodlouse/slater: UK-oriented; verbatim twin C314 maps to HDS q74."),
]


def main():
    camb_q, camb_a, hds_a = load()
    rows = []

    # 1) Verbatim block C(240+hq) == HDS q(hq), hq = 1..122.
    # Confidence by whether the item has real, recoverable US geography and
    # whether its dialect-relevant answers land on the recoverable primaries.
    strong_geo = {"50", "58", "60", "62", "63", "64", "65", "66", "72", "73",
                  "74", "75", "76", "80", "84", "94", "96", "97", "103", "104",
                  "105", "110", "117", "119", "120"}
    phon_geo = {"1", "10", "15", "20", "21", "26", "28", "108"}  # subtler
    for hq in range(1, 123):
        cid = str(hq + 240)
        if cid not in camb_a:
            continue
        hrows = hds_a.get(str(hq), [])
        if hq == 105:
            note = ("verbatim HDS q105 (carbonated beverage), single-select; "
                    "soda/pop/coke land on the recoverable RGB primaries.")
            conf = "high"
        elif str(hq) in strong_geo:
            conf = "high"
            note = "verbatim HDS re-run, single-select; strong lexical isogloss."
        elif str(hq) in phon_geo:
            conf = "medium"
            note = ("verbatim HDS re-run; phonetic item with real but subtler "
                    "geography; some variants fall on secondary colours.")
        else:
            conf = "medium"
            note = ("verbatim HDS re-run, single-select; weak/near-uniform "
                    "geography or answers on hard (secondary/black) colours.")
        matched = 0
        for a in camb_a[cid]:
            h = match_choice(a["answer"], hrows)
            if h is None:
                continue
            rows.append((cid, a["answer"], str(hq), h["answer"], conf, note))
            matched += 1
        if matched == 0 and hrows:
            rows.append((cid, "(question maps; no individual choice text "
                         "aligned)", str(hq), "", "low",
                         "verbatim HDS re-run but answer wording diverged."))

    # 2) Featured-block duplicates and extras.
    for cid, hq, conf, note in FEATURED:
        if conf == "reject" or hq == "-":
            rows.append((cid, "(all)", hq, "", "reject", note))
            continue
        hrows = hds_a.get(hq, [])
        any_ch = False
        for a in camb_a.get(cid, []):
            h = match_choice(a["answer"], hrows)
            if h is None:
                continue
            rows.append((cid, a["answer"], hq, h["answer"], conf, note))
            any_ch = True
        if not any_ch:
            rows.append((cid, "(all)", hq, "", conf, note))

    # 3) Explicit rejects (featured questions with no HDS counterpart).
    for cid, note in REJECT:
        rows.append((cid, "(all)", "-", "", "reject", note))

    out = DATA / "cambridge" / "hds_crosswalk.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cambridge_id", "cambridge_choice", "hds_question",
                    "hds_choice", "confidence", "note"])
        w.writerows(rows)

    n_high = sum(1 for r in rows if r[4] == "high")
    n_hq = len({r[0] for r in rows if r[4] == "high"})
    print(f"wrote {out}: {len(rows)} choice rows")
    print(f"  high-confidence choice rows: {n_high} across {n_hq} questions")
    print(f"  reject rows: {sum(1 for r in rows if r[4]=='reject')}")


if __name__ == "__main__":
    main()
