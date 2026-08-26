"""Scrape the Harvard Dialect Survey (Vaux & Golder, 2002-2003) static mirror.

Produces three tables:
  data/hds/questions.csv  - 122 questions
  data/hds/answers.csv    - answer choices with national percentages and plot colour
  data/hds/state_pct.csv  - per-state percentage for every question/answer
"""

import csv
import html
import re
import sys

from common import fetch, out_dir

BASE = "http://dialect.redlog.net/staticmaps/"

STATES = """AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS
MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY""".split()

# The survey rendered phonetic symbols as small GIFs; map filename -> IPA character.
IPA = {
    "ah": "ɑ", "ash": "æ", "backwardsa": "ɔ", "ih": "ɪ", "eh": "ɛ",
    "schwa": "ə", "upsidedowna": "ə", "uh": "ʌ", "oo": "ʊ", "ay": "eɪ",
    "theta": "θ", "eth": "ð", "esh": "ʃ", "ezh": "ʒ", "eng": "ŋ",
    "glottalstop": "ʔ", "barredi": "ɨ", "openo": "ɔ", "smallcapi": "ɪ",
}
unknown_ipa = set()


def clean(fragment):
    """Turn an answer-text HTML fragment into plain text, substituting IPA glyphs."""
    def sub_img(m):
        name = m.group(1).rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if name not in IPA:
            unknown_ipa.add(name)
        return IPA.get(name, f"[{name}]")

    s = re.sub(r'<img[^>]*src=["\']?([^"\'\s>]+)[^>]*>', sub_img, fragment, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_question(num, page):
    """Extract question text, respondent count and answer choices from q_N.html."""
    m = re.search(rf"{num}\.\s*<b>(.*?)</b>\s*<br", page, re.S)
    if not m:
        raise ValueError(f"no question text for q{num}")
    qtext = clean(m.group(1))

    m = re.search(r"\((\d+)\s+respondents\)", page)
    respondents = int(m.group(1)) if m else None

    answers = []
    pattern = r'([a-z])\.\s*<font color="([^"]+)">(.*?)</font>\s*\(([\d.]+)%\)'
    for letter, colour, text, pct in re.findall(pattern, page, re.S):
        answers.append({
            "question": num,
            "choice": letter,
            "choice_index": ord(letter) - ord("a") + 1,
            "answer": clean(text),
            "color": colour,
            "pct_national": float(pct),
        })
    if not answers:
        raise ValueError(f"no answers parsed for q{num}")
    return qtext, respondents, answers


def parse_state(page):
    """Extract per-question answer percentages from state_XX.html."""
    rows = []
    blocks = re.split(r'<table cellpadding="0" cellspacing="0" border="0">', page)
    for block in blocks[1:]:
        m = re.search(r'<td colspan="4"><b>(\d+)\.\s*(.*?)</b></td>', block, re.S)
        if not m:
            continue
        qnum = int(m.group(1))
        for letter, text, pct in re.findall(
            r"<td>\s*(?:<b>)?\s*([a-z])\.\s*(.*?)\s*(?:</b>)?\s*</td>"
            r'\s*<td width="20"></td>\s*<td>\s*(?:<b>)?\s*\(([\d.]+)%\)',
            block, re.S,
        ):
            rows.append({
                "question": qnum,
                "choice": letter,
                "answer": clean(text),
                "pct": float(pct),
            })
    return rows


def main():
    d = out_dir("hds")

    index = fetch(BASE.replace("staticmaps/", "") + "maps.html", "hds", "maps.html")
    nums = sorted({int(n) for n in re.findall(r"q_(\d+)\.html", index)})
    print(f"found {len(nums)} questions")

    questions, answers = [], []
    for n in nums:
        page = fetch(f"{BASE}q_{n}.html", "hds", f"q_{n}.html")
        qtext, resp, ans = parse_question(n, page)
        questions.append({"question": n, "text": qtext, "respondents": resp,
                          "n_choices": len(ans)})
        answers.extend(ans)
        print(f"  q{n:>3}  {len(ans)} choices  {resp} respondents  {qtext[:58]}")

    state_rows = []
    for st in STATES:
        page = fetch(f"{BASE}state_{st}.html", "hds", f"state_{st}.html")
        rows = parse_state(page)
        for r in rows:
            r["state"] = st
        state_rows.extend(rows)
        print(f"  {st}: {len(rows)} answer rows")

    # q_N.html only lists the choices that got a plot colour (max 12), but the state
    # pages carry every choice. Merge so answers.csv is the complete choice list.
    national = {(a["question"], a["choice"]): a for a in answers}
    text_from_state = {}
    for r in state_rows:
        text_from_state.setdefault((r["question"], r["choice"]), r["answer"])

    merged = []
    for (q, letter), text in sorted(text_from_state.items(),
                                   key=lambda kv: (kv[0][0], kv[0][1])):
        nat = national.get((q, letter))
        merged.append({
            "question": q,
            "choice": letter,
            "choice_index": ord(letter) - ord("a") + 1,
            "answer": nat["answer"] if nat else text,
            "color": nat["color"] if nat else "",
            "pct_national": nat["pct_national"] if nat else "",
            "has_map": int(nat is not None),
        })
    dropped = sorted(set(national) - set(text_from_state))
    if dropped:
        print(f"note: {len(dropped)} national choices absent from state pages: {dropped[:5]}")
    answers = merged

    with open(d / "questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["question", "text", "respondents", "n_choices"])
        w.writeheader()
        w.writerows(questions)

    with open(d / "answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["question", "choice", "choice_index", "answer",
                               "color", "pct_national", "has_map"])
        w.writeheader()
        w.writerows(answers)

    with open(d / "state_pct.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["state", "question", "choice", "answer", "pct"])
        w.writeheader()
        w.writerows(state_rows)

    print(f"\nquestions={len(questions)} answers={len(answers)} state_rows={len(state_rows)}")
    if unknown_ipa:
        print(f"UNMAPPED phonetic glyphs: {sorted(unknown_ipa)}", file=sys.stderr)


if __name__ == "__main__":
    main()
