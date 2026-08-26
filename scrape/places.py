"""Named towns with coordinates and population, so results can be spoken aloud.

A posterior over grid cells is the honest output, but "somewhere around 40.4 N,
80.0 W" is not a party trick. This joins the Census gazetteer, which has a
point for every place, to the population estimates, which have a size for every
place, and keeps the ones big enough to be recognised.

Population is not decoration here. Posterior mass is per unit area, so the
single most probable cell is often a rural one that happens to sit under a
sharp likelihood ridge. Weighting candidate towns by how many people live in
them is what turns a probable location into a probable answer.
"""

import csv
import io
import zipfile

from common import DATA, fetch, out_dir

GAZETTEER = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
             "2023_Gazetteer/2023_Gaz_place_national.zip")
POPULATION = ("https://www2.census.gov/programs-surveys/popest/datasets/"
              "2020-2024/cities/totals/sub-est2024.csv")
MIN_POP = 5000
SKIP = ("AK", "HI", "PR", "VI", "GU", "AS", "MP")


def gazetteer():
    raw = fetch(GAZETTEER, "census", filename="gazetteer_place.zip", binary=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith(".txt"))
        text = z.read(name).decode("latin-1")
    out = {}
    for r in csv.DictReader(io.StringIO(text), delimiter="\t"):
        r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        if r["USPS"] in SKIP:
            continue
        try:
            out[r["GEOID"]] = {
                "name": r["NAME"], "state": r["USPS"],
                "lat": float(r["INTPTLAT"]), "lon": float(r["INTPTLONG"]),
            }
        except (KeyError, ValueError):
            continue
    return out


def populations():
    text = fetch(POPULATION, "census", filename="sub_est_2024.csv",
                 binary=True).decode("latin-1")
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r["PLACE"] == "00000" or r["SUMLEV"] not in ("162", "157"):
            continue
        geoid = r["STATE"].zfill(2) + r["PLACE"].zfill(5)
        try:
            pop = int(float(r["POPESTIMATE2024"]))
        except ValueError:
            continue
        out[geoid] = max(out.get(geoid, 0), pop)
    return out


def trim(name):
    """'Pittsburgh city' -> 'Pittsburgh'."""
    for suffix in (" city", " town", " village", " borough", " CDP",
                   " municipality", " (balance)", " unified government",
                   " consolidated government", " metro government",
                   " metropolitan government", " urban county government"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def main():
    geo = gazetteer()
    pop = populations()
    rows = []
    for geoid, g in geo.items():
        n = pop.get(geoid)
        if n and n >= MIN_POP:
            rows.append({"geoid": geoid, "name": trim(g["name"]),
                         "state": g["state"], "lat": round(g["lat"], 5),
                         "lon": round(g["lon"], 5), "pop": n})
    rows.sort(key=lambda r: -r["pop"])

    path = out_dir("census") / "places.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["geoid", "name", "state", "lat", "lon", "pop"])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(geo)} gazetteer places, {len(pop)} with population, "
          f"{len(rows)} at >= {MIN_POP:,}")
    print(f"largest: {', '.join(r['name'] + ' ' + r['state'] for r in rows[:8])}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
