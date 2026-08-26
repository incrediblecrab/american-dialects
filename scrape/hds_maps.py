"""Download every Harvard Dialect Survey dot map.

The survey plotted each respondent at their ZIP centroid. The raw ZIP-level
records were never published, but they survive as pixels in these GIFs:
q_N.gif is the composite map, q_N_M.gif is the map for choice M alone.
"""

import csv

from common import DATA, fetch

BASE = "http://dialect.redlog.net/staticmaps/"


def main():
    with open(DATA / "hds" / "answers.csv", encoding="utf-8") as f:

        answers = [a for a in csv.DictReader(f) if a["has_map"] == "1"]

    questions = sorted({int(a["question"]) for a in answers})
    targets = [(f"q_{q}.gif", f"q_{q}.gif") for q in questions]
    targets += [(f"q_{a['question']}_{a['choice_index']}.gif",
                 f"q_{a['question']}_{a['choice_index']}.gif") for a in answers]

    print(f"downloading {len(targets)} maps ({len(questions)} composite, {len(answers)} per-choice)")
    failed = []
    for i, (name, fn) in enumerate(targets, 1):
        try:
            data = fetch(BASE + name, "hds_maps", fn, binary=True)
            if not data.startswith(b"GIF"):
                failed.append((name, "not a GIF"))
        except Exception as e:
            failed.append((name, str(e)[:80]))
        if i % 100 == 0:
            print(f"  {i}/{len(targets)}")

    print(f"done. failed: {len(failed)}")
    for n, e in failed[:20]:
        print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
