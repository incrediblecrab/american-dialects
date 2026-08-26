"""A common format for people whose real locations are known.

Everything that can honestly test this model is a set of people who answered
some dialect questions and grew up somewhere known. Those sets arrive in
different shapes: YGDP ships Likert ratings of syntactic constructions, the
Cambridge survey ships multiple choice, and a list of friends is whatever they
typed. Rather than teach the calibrator about each one, each source is converted
once into the same two tables.

    people.csv   person, lat, lon, [place, age, race, education, moved]
    answers.csv  person, question, choice

`question` and `choice` are Harvard question numbers and choice letters, because
that is what the tensor is indexed by. Converting a source into them is the part
that requires judgement and is where the errors will be, so each source writes a
crosswalk next to its data saying which of its items map to which Harvard items
and what was rejected.

`moved` matters more than it looks. The estimand is where a person was raised,
and a set of people who all still live where they grew up will flatter the model
in a way that will not survive contact with a party. Where a source knows both,
keep both, and score against each.
"""

import csv
from pathlib import Path

import numpy as np


class People:
    ID_FIELDS = ("person", "person_id", "id", "respondent")

    def __init__(self, root):
        self.root = Path(root)
        self.rows = []
        with open(self.root / "people.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            key = next((k for k in self.ID_FIELDS if k in reader.fieldnames), None)
            if key is None:
                raise ValueError(f"{root}/people.csv has no id column "
                                 f"(looked for {self.ID_FIELDS})")
            for r in reader:
                try:
                    lat, lon = float(r["lat"]), float(r["lon"])
                except (TypeError, ValueError):
                    continue
                if not (-180 <= lon <= -50 and 18 <= lat <= 72):
                    continue
                r["person"] = r[key]
                r["lat"], r["lon"] = lat, lon
                self.rows.append(r)
        self.n_geocoded = len(self.rows)
        self.by_id = {r["person"]: r for r in self.rows}

        self.answers = {}
        path = self.root / "answers.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. people.csv alone is not a validation set; "
                f"see the module docstring for the two-table format.")
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            key = next((k for k in self.ID_FIELDS if k in reader.fieldnames), None)
            for r in reader:
                pid = r[key]
                if pid in self.by_id and r["choice"]:
                    self.answers.setdefault(pid, []).append(
                        (str(r["question"]), str(r["choice"])))

        self.rows = [r for r in self.rows if self.answers.get(r["person"])]
        self.by_id = {r["person"]: r for r in self.rows}

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for r in self.rows:
            yield r, self.answers[r["person"]]

    def summary(self):
        n_ans = [len(self.answers[r["person"]]) for r in self.rows]
        qs = {q for a in self.answers.values() for q, _ in a}
        return {
            "people": len(self.rows),
            "questions": len(qs),
            "answers_per_person_median": float(np.median(n_ans)) if n_ans else 0.0,
            "answers_per_person_min": int(min(n_ans)) if n_ans else 0,
            "answers_per_person_max": int(max(n_ans)) if n_ans else 0,
            "question_list": sorted(qs, key=lambda s: int(s) if s.isdigit() else 0),
        }

    def strata(self, field, bins=None):
        """Group people by a demographic field, for stratified calibration.

        Pooled calibration can look perfect while every subgroup is wrong, if
        the subgroups are miscalibrated in opposite directions. Age and mobility
        are the two most likely to do that here.
        """
        out = {}
        for r in self.rows:
            v = (r.get(field) or "").strip()
            if not v:
                continue
            if bins is not None:
                try:
                    x = float(v)
                except ValueError:
                    continue
                i = int(np.searchsorted(bins, x))
                lo = bins[i - 1] if i else "<"
                hi = bins[i] if i < len(bins) else "+"
                v = f"{lo}-{hi}"
            out.setdefault(v, []).append(r["person"])
        return out
