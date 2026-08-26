"""County population, used as the prior over where a person could be from.

Without a prior the posterior drifts into empty country: the likelihood is a
smooth surface, so a sparsely sampled desert cell can score as well as
Chicago. Weighting by how many people actually live somewhere fixes that.

Two vintages are collected because they answer different questions:

  2003  the survey year, so it matches the population the dialect surfaces
        were drawn from
  2024  where people live now

For "where did you grow up", 2003 is the better prior. Today's adults were
children before the Sun Belt reached its current size, and using present-day
counts would quietly assume Phoenix and Orlando raised as many people as they
now house. The Census API began requiring a key, so these come from the static
file server, which does not.
"""

import csv
import io

from common import DATA, fetch, out_dir

BASE = "https://www2.census.gov/programs-surveys/popest/datasets"
VINTAGES = {
    "pop2003": (f"{BASE}/2000-2010/intercensal/county/co-est00int-tot.csv",
                "POPESTIMATE2003"),
    "pop2024": (f"{BASE}/2020-2024/counties/totals/co-est2024-alldata.csv",
                "POPESTIMATE2024"),
}


def read(url, column):
    text = fetch(url, "census", binary=True).decode("latin-1")
    rows = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r.get("SUMLEV", "").lstrip("0") != "50":  # 50 is county; 40 is state total
            continue
        fips = r["STATE"].zfill(2) + r["COUNTY"].zfill(3)
        rows[fips] = {
            "state": r["STNAME"], "county": r["CTYNAME"],
            "value": int(float(r[column])),
        }
    return rows


def main():
    out = out_dir("census")
    merged = {}
    for key, (url, column) in VINTAGES.items():
        rows = read(url, column)
        print(f"{key}: {len(rows)} counties, {sum(r['value'] for r in rows.values()):,} people")
        for fips, r in rows.items():
            m = merged.setdefault(fips, {"fips": fips, "state": r["state"],
                                         "county": r["county"]})
            m[key] = r["value"]

    path = out / "counties.csv"
    fields = ["fips", "state", "county", "pop2003", "pop2024"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fips in sorted(merged):
            row = merged[fips]
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"wrote {path} ({len(merged)} counties)")


if __name__ == "__main__":
    main()
