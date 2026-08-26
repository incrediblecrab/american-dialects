"""Download the Yale Grammatical Diversity Project survey maps.

Unlike the Harvard survey, YGDP publishes respondent-level records: each feature
is one person, geocoded to their home city, with demographics and 1-5
acceptability ratings for the sentences in that phenomenon.

Outputs:
  data/ygdp/phenomena.csv    - the 28 phenomena and their source files
  data/ygdp/respondents.csv  - one row per respondent per phenomenon
  data/raw/ygdp/*.geojson    - untouched source files
"""

import csv
import html
import json

from common import fetch, out_dir

INDEX = "https://ygdp.yale.edu/maps/json"

CORE = ["Age", "Age_Bin", "Gender", "Education", "Race", "Raised.CityState",
        "Mother.CityState", "Father.CityState", "Current.CityState", "SurveyType"]


def main():
    d = out_dir("ygdp")
    phenomena = json.loads(fetch(INDEX, "ygdp", "maps.json"))
    print(f"{len(phenomena)} phenomena")

    rows, meta, rating_keys = [], [], set()
    for p in phenomena:
        url = p.get("geojson")
        if not url:
            continue
        name = url.rsplit("/", 1)[-1]
        title = html.unescape(p.get("title") or "")
        try:
            fc = json.loads(fetch(url, "ygdp", name))
        except Exception as e:
            print(f"  FAILED {title}: {str(e)[:70]}")
            continue

        feats = fc.get("features", [])
        meta.append({"title": title, "id": p.get("id"),
                     "file": name, "respondents": len(feats)})
        print(f"  {title[:44]:44s} {len(feats):5d} respondents")

        for feat in feats:
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None]
            ratings = {k: v for k, v in props.items()
                       if k not in CORE and isinstance(v, (int, float))}
            rating_keys.update(ratings)
            row = {"phenomenon": title, "file": name,
                   "lon": coords[0], "lat": coords[1]}
            row.update({k: props.get(k) for k in CORE})
            row["ratings"] = json.dumps(ratings, separators=(",", ":"))
            rows.append(row)

    with open(d / "phenomena.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["title", "id", "file", "respondents"])
        w.writeheader()
        w.writerows(meta)

    with open(d / "respondents.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["phenomenon", "file", "lat", "lon"] + CORE + ["ratings"])
        w.writeheader()
        w.writerows(rows)

    geo = sum(1 for r in rows if r["lat"] is not None)
    print(f"\nphenomena={len(meta)} respondent rows={len(rows)} geocoded={geo}")
    print(f"distinct rating fields={len(rating_keys)}")


if __name__ == "__main__":
    main()
