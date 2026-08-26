"""Turn played games into a validation set.

This is the small script that matters most, because it is the only route this
project has to an honest calibration number.

The problem it exists to solve is stated plainly in the README: no public
dataset has real people with known hometowns answering many dialect questions.
The Harvard survey has the answers but published only state-level aggregates and
rendered dot maps. YGDP has real located people but overlaps Harvard on exactly
two distinct constructions, which is worth 0.06 nats and moves the median error
25 km. Everything else is either aggregate, or simulated from the model's own
beliefs and therefore circular.

Playing the quiz produces the missing thing one row at a time: a real person,
answering questions chosen adaptively, who then says where they are actually
from. Fifty of those is a better calibration set than anything currently public,
because the answers are the model's own questions and the truth is volunteered
rather than inferred.

    ../.venv/bin/python export.py

reads data/quiz/log.csv, geocodes whatever hometown strings lack coordinates,
and writes data/quiz/{people,answers,crosswalk}.csv in the format model/people.py
defines. Then calibrate.py and validate.py work on it unchanged:

    ../.venv/bin/python calibrate.py --set quiz --auto
    ../.venv/bin/python validate.py  --set quiz

One deliberate omission. The log also stores what the model guessed --
map_lat, map_lon, map_state, top_place -- and none of it is carried into
people.csv. Those are predictions, and a validation table that contains the
predictions it is meant to score is one careless join away from being worthless.
Only the truth crosses over.
"""

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

QUIZ = DATA / "quiz"
LOG = QUIZ / "log.csv"
CACHE = QUIZ / "geocode.json"

NOMINATIM = "https://nominatim.openstreetmap.org/search"
AGENT = "american-dialects/1.0 (research; contact via repository)"
PAUSE = 1.1


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")


def geocode(place, cache, session_state):
    """Resolve a hometown string, one request per second, cached forever.

    A null result is cached as well as a hit. Somebody typing "the sticks" will
    never resolve, and re-asking on every run is both slow and rude to a free
    service.
    """
    key = place.strip().lower()
    if key in cache:
        return cache[key]

    if session_state["elapsed"] is not None:
        wait = PAUSE - (time.monotonic() - session_state["elapsed"])
        if wait > 0:
            time.sleep(wait)

    query = urllib.parse.urlencode({
        "q": place, "format": "json", "limit": 1, "countrycodes": "us",
    })
    req = urllib.request.Request(f"{NOMINATIM}?{query}",
                                 headers={"User-Agent": AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            hits = json.load(r)
    except Exception as e:
        print(f"    geocode failed for {place!r}: {type(e).__name__}: {e}")
        session_state["elapsed"] = time.monotonic()
        return None
    session_state["elapsed"] = time.monotonic()

    result = None
    if hits:
        result = {"lat": float(hits[0]["lat"]), "lon": float(hits[0]["lon"]),
                  "display": hits[0].get("display_name", "")}
    cache[key] = result
    return result


def parse_answers(blob):
    out = []
    for pair in (blob or "").split(";"):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        q, _, c = pair.partition(":")
        q, c = q.strip(), c.strip()
        if q and c:
            out.append((q, c))
    return out


def export(dry_run=False):
    if not LOG.exists():
        print(f"no games logged yet. {LOG} does not exist.")
        print("run quiz.py, play a few rounds, and tell it where you are from.")
        return 1

    with open(LOG, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} games in the log")

    cache = load_cache()
    session_state = {"elapsed": None}
    people, answers = [], []
    no_truth = no_answers = ungeocodable = 0

    for r in rows:
        truth = (r.get("truth") or "").strip()
        if not truth:
            no_truth += 1
            continue

        pairs = parse_answers(r.get("answers"))
        if not pairs:
            no_answers += 1
            continue

        lat = lon = None
        raw_lat, raw_lon = (r.get("lat") or "").strip(), (r.get("lon") or "").strip()
        if raw_lat and raw_lon:
            try:
                lat, lon = float(raw_lat), float(raw_lon)
            except ValueError:
                lat = lon = None

        if lat is None:
            if dry_run:
                print(f"    would geocode {truth!r}")
                ungeocodable += 1
                continue
            hit = geocode(truth, cache, session_state)
            if not hit:
                print(f"    could not place {truth!r}, skipping")
                ungeocodable += 1
                continue
            lat, lon = hit["lat"], hit["lon"]

        pid = r.get("person") or f"g{len(people):05d}"
        people.append({"person": pid, "lat": f"{lat:.5f}", "lon": f"{lon:.5f}",
                       "place": truth, "played_at": r.get("played_at", "")})
        for q, c in pairs:
            answers.append({"person": pid, "question": q, "choice": c})

    if not dry_run:
        save_cache(cache)

    print(f"  usable      {len(people)}")
    print(f"  no hometown {no_truth}")
    print(f"  no answers  {no_answers}")
    print(f"  unplaceable {ungeocodable}")

    if dry_run:
        print("\ndry run, nothing written")
        return 0

    if not people:
        print("\nnothing usable to write")
        return 1

    QUIZ.mkdir(parents=True, exist_ok=True)
    with open(QUIZ / "people.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["person", "lat", "lon", "place",
                                           "played_at"])
        wr.writeheader()
        wr.writerows(people)
    with open(QUIZ / "answers.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["person", "question", "choice"])
        wr.writeheader()
        wr.writerows(answers)

    (QUIZ / "crosswalk.csv").write_text(
        "source_item,hds_question,hds_choice,confidence,note\n"
        "*,*,*,exact,"
        "\"The quiz asks Harvard items directly and records the choice letter "
        "the player picked, so unlike YGDP or Cambridge there is no mapping "
        "step and nothing to reject. This file exists because people.py asks "
        "every source for one.\"\n",
        encoding="utf-8")

    print(f"\nwrote {QUIZ}/people.csv, answers.csv, crosswalk.csv")
    per = len(answers) / len(people)
    print(f"{len(people)} people, {len(answers)} answers, {per:.1f} each")
    if len(people) < 30:
        print(f"\n{len(people)} is too few to calibrate on. The YGDP fit moved "
              "by less than 0.01 nats\nacross base in [0.3, 0.85], and a set "
              "this small will be flatter still.\nKeep playing.")
    else:
        print("\nnow:  ../.venv/bin/python calibrate.py --set quiz --auto")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, geocode nothing")
    args = ap.parse_args()
    raise SystemExit(export(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
