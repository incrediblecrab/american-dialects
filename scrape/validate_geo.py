"""Check the map-derived geography against the survey's own published numbers.

For every question the survey published a percentage breakdown per state. Those
numbers were computed from the raw respondent records, so they are independent
ground truth for the coverage surfaces recovered from the dot maps in hds_geo.py.

Agreement is expected to be strong but not exact: a pixel records that at least
one respondent chose an answer nearby, so overlapping dots in dense metros
saturate and coverage measures spatial extent rather than respondent count.
"""

import csv
import json
from collections import defaultdict

import numpy as np

from common import DATA, fetch, out_dir

STATES_GEOJSON = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
                  "master/data/geojson/us-states.json")

ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "Puerto Rico": "PR",
}


def rings(feature):
    g = feature["geometry"]
    if g["type"] == "Polygon":
        return [g["coordinates"][0]]
    return [poly[0] for poly in g["coordinates"]]


def rasterise_states(lats, lons):
    """Label each grid cell with the state whose polygon contains its centre."""
    raw = fetch(STATES_GEOJSON, "geo", "us-states.json")
    fc = json.loads(raw)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    px, py = lon_grid.ravel(), lat_grid.ravel()
    labels = np.full(px.shape, "", dtype=object)

    for feat in fc["features"]:
        abbr = ABBR.get(feat["properties"]["name"])
        if not abbr:
            continue
        for ring in rings(feat):
            poly = np.asarray(ring, dtype=float)
            x, y = poly[:, 0], poly[:, 1]
            if px.min() > x.max() or px.max() < x.min():
                continue
            inside = np.zeros(px.shape, dtype=bool)
            j = len(x) - 1
            for i in range(len(x)):
                cond = ((y[i] > py) != (y[j] > py))
                with np.errstate(divide="ignore", invalid="ignore"):
                    xint = (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i]) + x[i]
                inside ^= cond & (px < xint)
                j = i
            labels[inside & (labels == "")] = abbr

    return labels.reshape(lat_grid.shape)


def main():
    geo = DATA / "hds" / "geo"
    z = np.load(geo / "grid.npz", allow_pickle=True)
    grid = z["density"].astype(np.float32)
    question, choice = z["question"], z["choice"]
    lats, lons = z["lats"], z["lons"]

    state_grid = rasterise_states(lats, lons)
    print(f"cells assigned to a state: {(state_grid != '').sum()} of {state_grid.size}")

    # coverage-weighted share per state for each question/answer
    derived = {}
    by_q = defaultdict(list)
    for i, q in enumerate(question):
        by_q[str(q)].append(i)

    states = sorted({s for s in state_grid.ravel() if s})
    masks = {s: (state_grid == s) for s in states}

    for q, idxs in by_q.items():
        for s in states:
            m = masks[s]
            tot = sum(float(grid[i][m].sum()) for i in idxs)
            if tot <= 0:
                continue
            for i in idxs:
                derived[(s, q, str(choice[i]))] = 100.0 * float(grid[i][m].sum()) / tot

    with open(DATA / "hds" / "state_pct.csv", encoding="utf-8") as f:
        truth = {(r["state"], r["question"], r["choice"]): float(r["pct"])
                 for r in csv.DictReader(f)}

    pairs = [(v, derived[k]) for k, v in truth.items() if k in derived]
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    r = float(np.corrcoef(a, b)[0, 1])
    mae = float(np.abs(a - b).mean())

    # does the map pick the same winning answer as the survey?
    tw, dw = {}, {}
    for (s, q, c), v in truth.items():
        if tw.get((s, q), (None, -1))[1] < v:
            tw[(s, q)] = (c, v)
    for (s, q, c), v in derived.items():
        if dw.get((s, q), (None, -1))[1] < v:
            dw[(s, q)] = (c, v)
    common = [k for k in tw if k in dw]
    agree = sum(1 for k in common if tw[k][0] == dw[k][0])

    print(f"\ncompared {len(pairs)} state x question x answer percentages")
    print(f"  pearson r = {r:.4f}")
    print(f"  mean absolute error = {mae:.2f} percentage points")
    print(f"  modal answer agrees on {agree}/{len(common)} state-questions "
          f"({100.0 * agree / len(common):.1f}%)")

    out = out_dir("hds/geo")
    with open(out / "validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state", "question", "choice", "pct_published", "pct_from_map"])
        for (s, q, c), v in sorted(truth.items()):
            if (s, q, c) in derived:
                w.writerow([s, q, c, f"{v:.2f}", f"{derived[(s, q, c)]:.2f}"])


if __name__ == "__main__":
    main()
