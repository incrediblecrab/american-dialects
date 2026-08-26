"""Scrape the Cambridge Online Survey of World Englishes (Vaux & Jøhndal).

The live results index lists ~180 questions at
    https://tekstlab.uio.no/cambridge_survey/maps
Each question page (maps/<ID>) carries the question text, a colour legend of
answer choices with national percentages, and a Leaflet map whose respondent
dots come from a raster overlay served at maps/<ID>/{x}/{y}/{z}.png (note the
unusual x/y/z path order). This module recovers the two tabular products:

  data/cambridge/questions.csv  - id, text, n_choices, multiselect
  data/cambridge/answers.csv    - id, choice_index, answer, color_hex, pct

Raw HTML is cached under data/raw/cambridge/. The dot tiles are downloaded by
cambridge_geo.py, which recovers the sub-national geography from them.
"""

import csv
import html
import re

from common import fetch, out_dir

INDEX = "https://tekstlab.uio.no/cambridge_survey/maps"
QUESTION = "https://tekstlab.uio.no/cambridge_survey/maps/{}"

# The seven-colour plot palette, always assigned to answers in this order, so a
# question with N choices uses the first N entries. Parsed from each legend too.
PALETTE = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"]

# Phrases that mark a "select all that apply" question, where percentages are
# shares of respondents (each may pick several) rather than a single-choice
# split. Kept strict: bare "more than one" also appears in ordinary questions
# ("a seat for more than one person", "use more than one modal"), so it is not
# a trigger on its own.
MULTISELECT_RE = re.compile(
    r"select all|check all|mark all|tick all|all that apply|choose all|"
    r"select as many|check as many|select any|check any", re.I)


def clean(fragment):
    """Plain text from an HTML fragment: drop tags, unescape entities, collapse."""
    s = re.sub(r"<[^>]+>", "", fragment)
    s = html.unescape(s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_index(page):
    """Return [(id, text)] in listed order from the results index."""
    items = re.findall(r'<li>\s*<a href="(\d+)">(.*?)</a>\s*</li>', page, re.S)
    return [(int(i), clean(t)) for i, t in items]


def parse_question(qid, page):
    """Extract question text and the answer legend from a question page."""
    m = re.search(r"<div class='right'>\s*(.*?)\s*<ul>", page, re.S)
    if not m:
        raise ValueError(f"q{qid}: no question body")
    qtext = clean(m.group(1))

    answers = []
    legend = re.findall(
        r"<span class='legend-box' style='background-color:\s*"
        r"(#[0-9a-fA-F]{6})'>.*?</span>\s*(.*?)\s*</li>", page, re.S)
    for idx, (color, raw) in enumerate(legend, start=1):
        text = clean(raw)
        pm = re.search(r"\(([\d.]+)%\)\s*$", text)
        pct = float(pm.group(1)) if pm else None
        answer = text[:pm.start()].strip() if pm else text
        answers.append({
            "id": qid,
            "choice_index": idx,
            "answer": answer,
            "color_hex": color.lower(),
            "pct": pct,
        })
    if not answers:
        raise ValueError(f"q{qid}: no answers parsed")
    return qtext, answers


def main():
    d = out_dir("cambridge")

    index_page = fetch(INDEX, "cambridge", "index.html")
    listed = parse_index(index_page)
    print(f"index lists {len(listed)} questions; ids {listed[0][0]}..{listed[-1][0]}")

    questions, answers = [], []
    off_palette = []
    for qid, _ in listed:
        page = fetch(QUESTION.format(qid), "cambridge", f"q_{qid}.html")
        qtext, ans = parse_question(qid, page)
        multiselect = int(bool(MULTISELECT_RE.search(qtext)))
        questions.append({
            "id": qid,
            "text": qtext,
            "n_choices": len(ans),
            "multiselect": multiselect,
        })
        answers.extend(ans)
        # invariant: every dot colour is one of the seven palette colours (the
        # per-choice ORDER varies by question, so match on the set, not position)
        for a in ans:
            if a["color_hex"] not in PALETTE:
                off_palette.append((qid, a["color_hex"]))

    with open(d / "questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "text", "n_choices", "multiselect"])
        w.writeheader()
        w.writerows(questions)

    with open(d / "answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "choice_index", "answer", "color_hex", "pct"])
        w.writeheader()
        w.writerows(answers)

    ms = [q["id"] for q in questions if q["multiselect"]]
    nc = [q["n_choices"] for q in questions]
    print(f"questions={len(questions)} answers={len(answers)}")
    print(f"n_choices: min={min(nc)} max={max(nc)} mean={sum(nc)/len(nc):.1f}")
    print(f"off-palette colours: {len(off_palette)} (want 0)")
    print(f"multiselect questions ({len(ms)}): {ms}")


if __name__ == "__main__":
    main()
