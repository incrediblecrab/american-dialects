"""The party trick, and the instrument that will eventually justify it.

Ask questions, narrow the map, name a place, state a confidence. Adaptive by
default: each question is chosen for how much it would tell us about *this*
person given what they have already said, so the quiz stops asking about
carbonated drinks once it knows the answer and starts asking about something
else.

There is a second purpose, and it is the more important one. Every session is
appended to data/quiz/log.csv along with, if the player will say, where they
actually grew up. That file is the only route to an honest confidence claim.
The model's stated confidence currently rests on an extrapolation: the discount
tau was fitted where two questions could be tested and is being carried to
twenty, and nothing public can check it, because no public dataset has real
people with known hometowns answering many dialect questions. Fifty logged
games would settle it. Run:

    ../.venv/bin/python calibrate.py --set quiz

once the log has enough rows, and the number stops being a guess.
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from choose import Selector
from infer import N_QUESTIONS, Geolocator, Places

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

LOG = DATA / "quiz" / "log.csv"


def load_questions():
    """Question text and answer text, so the quiz reads like English."""
    qs, ans = {}, {}
    with open(DATA / "hds" / "questions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qs[r["question"]] = r["text"]
    with open(DATA / "hds" / "answers.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ans.setdefault(r["question"], {})[r["choice"]] = r["answer"]
    return qs, ans


def wrap(text, width=76, indent=""):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(indent + line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(indent + line)
    return "\n".join(out)


def describe(g, post, places, k=3):
    """What the model would say out loud right now.

    Metro areas rather than incorporated towns, because a player from Pittsburgh
    hears "Bethel Park" as a miss even though it is a suburb of the right answer.
    """
    ranked = places.areas(g.t, post, k)
    states = [(s, v) for s, v in
              sorted(((str(s), float(post[g.t.state == s].sum()))
                      for s in np.unique(g.t.state)), key=lambda kv: -kv[1])[:k]]
    return ranked, states


def play(g, sel, places, qs, ans, n_questions, adaptive=True, order=None):
    asked, answers = [], []
    w = g.prior.astype(np.float64).copy()

    for step in range(1, n_questions + 1):
        if adaptive:
            remaining = [q for q in g.t.questions if q not in asked]
            info = sel.information(w[None, :], remaining)
            q = max(remaining, key=lambda x: float(info[x].mean()))
        else:
            if step > len(order):
                break
            q = order[step - 1]

        choices = [(g.t.choice[i], ans.get(q, {}).get(g.t.choice[i], g.t.choice[i]))
                   for i in g.t.rows[q]]
        print(f"\n[{step}/{n_questions}] " + wrap(qs.get(q, f"question {q}"),
                                                  indent="").lstrip())
        for j, (letter, text) in enumerate(choices, 1):
            print(f"   {j:>2}. {text}")
        print("    0. skip / none of these")

        pick = None
        while pick is None:
            raw = input("  > ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                return asked, answers, w
            if raw.isdigit() and 0 <= int(raw) <= len(choices):
                pick = int(raw)
        asked.append(q)
        if pick == 0:
            continue

        letter = choices[pick - 1][0]
        answers.append((q, letter))
        i = g.index[(q, letter)]
        w = w * np.exp(g.loglik(i) - g.loglik(i).max())
        w = w / w.sum()

    return asked, answers, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--questions", type=int, default=N_QUESTIONS)
    ap.add_argument("--fixed", action="store_true",
                    help="use the precomputed order instead of adapting")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    print("loading the model ...", flush=True)
    g = Geolocator()
    places = Places(min_pop=20000)
    sel = Selector(g)
    qs, ans = load_questions()

    order = []
    path = DATA / "model" / "question_order.csv"
    if args.fixed and path.exists():
        with open(path, encoding="utf-8") as f:
            order = [r["question"] for r in csv.DictReader(f)]

    print("\nAnswer with the number. Say what you actually say, not what you")
    print("think is correct. Type q to stop early.\n")

    asked, answers, _ = play(g, sel, places, qs, ans, args.questions,
                             adaptive=not args.fixed, order=order)
    if not answers:
        print("\nnothing answered.")
        return

    post = g.posterior(answers)
    ranked, states = describe(g, post, places)
    tau = g.tau_used(answers)

    print("\n" + "=" * 60)
    best = int(np.argmax(post))
    print(f"after {len(answers)} answers (tau {tau:.2f}):\n")
    if ranked:
        print("  most likely places")
        for name, st, p in ranked:
            print(f"    {name}, {st:<3} {p:6.1%}")
    print("\n  most likely states")
    for s, p in states:
        print(f"    {s:<3} {p:6.1%}")
    area = sorted(post)[::-1]
    n80 = int(np.searchsorted(np.cumsum(area), 0.8) + 1)
    km2 = float(np.sort(g.cell_km2[np.argsort(post)[::-1][:n80]]).sum())
    print(f"\n  80% of the probability is inside {km2:,.0f} km2")
    print(f"  best single point: {g.t.cell_lat[best]:.2f}, "
          f"{g.t.cell_lon[best]:.2f} ({g.t.state[best]})")
    print("=" * 60)

    if args.no_log:
        return
    print("\nWhere did you actually grow up? This is what makes the confidence")
    print("above mean anything; leave blank to skip.")
    truth = input("  city, state > ").strip()
    lat = lon = ""
    if truth:
        raw = input("  lat,lon if you know it (blank is fine) > ").strip()
        if "," in raw:
            lat, _, lon = raw.partition(",")
            lat, lon = lat.strip(), lon.strip()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["played_at", "person", "truth", "lat", "lon",
                         "n_asked", "n_answered", "tau", "map_lat", "map_lon",
                         "map_state", "top_place", "answers"])
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        wr.writerow([now, f"q{abs(hash(now)) % 10**8:08d}", truth, lat, lon,
                     len(asked), len(answers), f"{tau:.3f}",
                     f"{g.t.cell_lat[best]:.4f}", f"{g.t.cell_lon[best]:.4f}",
                     g.t.state[best],
                     ranked[0][0] if ranked else "",
                     ";".join(f"{q}:{c}" for q, c in answers)])
    print(f"\nlogged to {LOG}")
    print("when there are 50 or so rows with locations, run "
          "export.py to turn them into a validation set.")


if __name__ == "__main__":
    main()
