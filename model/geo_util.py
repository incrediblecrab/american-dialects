"""Rasterise state and county polygons onto the survey's 200x456 grid.

The dot maps are plate carree with the transform solved in scrape/hds_geo.py,
so every grid cell has a known centre in degrees and can be labelled with the
polygon that contains it. Results are cached as .npy.
"""

import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA, fetch  # noqa: E402

STATES_URL = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
              "master/data/geojson/us-states.json")
COUNTIES_URL = ("https://raw.githubusercontent.com/plotly/datasets/"
                "master/geojson-counties-fips.json")

CACHE = DATA / "model"

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


def _rings(geom):
    if geom["type"] == "Polygon":
        return geom["coordinates"]
    out = []
    for poly in geom["coordinates"]:
        out.extend(poly)
    return out


def _rasterise(features, key_fn, lats, lons):
    """Label each cell with the first polygon containing its centre."""
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    labels = np.full(pts.shape[0], "", dtype=object)
    todo = np.ones(pts.shape[0], dtype=bool)

    for feat in features:
        key = key_fn(feat)
        if not key:
            continue
        for ring in _rings(feat["geometry"]):
            ring = np.asarray(ring, dtype=float)
            if ring.ndim != 2 or len(ring) < 4:
                continue
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            cand = todo & (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) \
                        & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
            if not cand.any():
                continue
            idx = np.nonzero(cand)[0]
            inside = MplPath(ring).contains_points(pts[idx])
            hit = idx[inside]
            if len(hit):
                labels[hit] = key
                todo[hit] = False

    return labels.reshape(lat_grid.shape)


def _load_grid_axes():
    z = np.load(DATA / "hds" / "geo" / "grid.npz", allow_pickle=True)
    return z["lats"], z["lons"]


def state_raster(force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "state_raster.npy"
    if path.exists() and not force:
        return np.load(path, allow_pickle=True)
    lats, lons = _load_grid_axes()
    fc = json.loads(fetch(STATES_URL, "geo", "us-states.json"))
    r = _rasterise(fc["features"],
                   lambda f: ABBR.get(f["properties"]["name"], ""), lats, lons)
    np.save(path, r)
    return r


def county_raster(force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "county_raster.npy"
    if path.exists() and not force:
        return np.load(path, allow_pickle=True)
    lats, lons = _load_grid_axes()
    fc = json.loads(fetch(COUNTIES_URL, "geo", "counties.json"))
    r = _rasterise(fc["features"], lambda f: str(f.get("id") or ""), lats, lons)
    np.save(path, r)
    return r


if __name__ == "__main__":
    s = state_raster(force=True)
    print(f"state cells labelled: {(s != '').sum()} / {s.size}, "
          f"{len({x for x in s.ravel() if x})} states")
    c = county_raster(force=True)
    print(f"county cells labelled: {(c != '').sum()} / {c.size}, "
          f"{len({x for x in c.ravel() if x})} counties")
