"""Recover the geography of the Cambridge survey from its dot tiles.

Each question's respondents are plotted as coloured dots on a transparent raster
overlay served at maps/<ID>/{x}/{y}/{z}.png (note the x/y/z path order). The
tiles are standard 256px Web Mercator, zoom 4-9. We download CONUS at zoom 6
(~1.9 km/pixel, ~7x finer than the Harvard GIFs).

Compositing model (verified empirically, see the module notes): the PNGs are
straight (non-premultiplied) RGBA. A dot is drawn in one of seven pure palette
colours; anti-aliasing and sub-pixel coverage live entirely in the alpha
channel, and the background is fully transparent. Overlapping dots of different
answers composite "over", producing blended RGB in the (~13% minority) overlap
pixels. So per pixel: total coverage = alpha/255, and the answer is the palette
colour nearest the (straight) RGB. This differs from the Harvard maps, which had
no alpha and were un-blended against a white background.

Dense metros still saturate (alpha pins at 255 where many dots stack), so as in
hds_geo.py we model respondents as Poisson in a local window and take
-ln(1-f) of the local coverage fraction as relative density.

Outputs (mirroring data/hds/geo/):
  data/cambridge/geo/grid.npz   - coverage + density on the Harvard 200x456
                                  plate carree grid, per answer (cell-comparable)
  data/cambridge/geo/cells.csv.gz - native zoom-6 inked pixels: lat, lon, coverage
  data/cambridge/geo/index.csv  - per-answer colour and totals
"""

import csv
import gzip
import io
import math
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

from common import DATA, RAW, out_dir

BASE = "https://tekstlab.uio.no/cambridge_survey/maps"
UA = {"User-Agent": "american-dialects-research/1.0 (personal research project)"}
RATE = 6.0                # aggregate tile requests per second (polite ceiling)
WORKERS = 8               # concurrent downloads, throttled by the shared limiter
Z = 6

# Harvard plate carree grid (identical constants to hds_geo.py) so the recovered
# surfaces line up cell for cell with the Harvard survey.
LON_M, LON_B = 0.12664511, -125.42846
LAT_M, LAT_B = -0.12718663, 49.21165
GW, GH = 456, 200

MIN_ALPHA = 0.06          # drop anti-aliasing noise below this coverage
PURE_THRESH = 40.0        # RGB distance within which a pixel is a single colour
BOX = 9                   # Harvard-grid density window (matches hds_geo.py)
BOX_NATIVE = 57           # ~110 km at CONUS mid-latitude: ~ the Harvard window
MAX_FRACTION = 0.995
CELL_MIN_COV = 0.05       # only emit native cells at least this covered

# Palette colours that carry a single RGB channel. The seven-colour palette is
# {R,G,B} primaries plus their pairwise sums {Y=R+G, M=R+B, C=G+B} and black, so
# an overlap of two primaries is indistinguishable pixel-for-pixel from the
# secondary colour that is their sum. Answers are ordered by popularity and the
# top three always take the primaries, so the dominant (dialect-relevant)
# answers are exactly the ones recovered without that ambiguity.
CHANNEL = {"#ff0000": 0, "#00ff00": 1, "#0000ff": 2}

_rl_lock = threading.Lock()
_next_slot = [0.0]


def _throttle():
    """Thread-safe token bucket: hand out request slots spaced 1/RATE apart."""
    with _rl_lock:
        t = max(time.time(), _next_slot[0])
        _next_slot[0] = t + 1.0 / RATE
    wait = t - time.time()
    if wait > 0:
        time.sleep(wait)


def _tile_path(qid, x, y, z):
    d = RAW / "cambridge" / "tiles"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"q{qid}_{x}_{y}_{z}.png"


def deg2tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def conus_tile_range(z):
    """Tile x/y bounds that cover the Harvard grid's lon/lat extent."""
    lon_w, lon_e = LON_B, LON_B + GW * LON_M
    lat_n, lat_s = LAT_B, LAT_B + (GH - 1) * LAT_M
    x0, y0 = deg2tile(lon_w, lat_n, z)
    x1, y1 = deg2tile(lon_e, lat_s, z)
    return x0, x1, y0, y1


def tile_fetch(qid, x, y, z):
    """Polite, cached, single-attempt tile fetch. Returns bytes or None.

    Empty regions legitimately return a tiny transparent PNG (HTTP 200); the
    survey's tile pyramid also 500s on some out-of-CONUS edge tiles. Neither
    warrants the aggressive retry/backoff in common.fetch, so this is bespoke.
    A miss is not cached (it may be transient), a valid PNG is. Thread-safe via
    the shared rate limiter so a pool of workers can prefetch a question.
    """
    path = _tile_path(qid, x, y, z)
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    _throttle()
    try:
        r = requests.get(f"{BASE}/{qid}/{x}/{y}/{z}.png",
                         headers={**UA, "Referer": f"{BASE}/{qid}"}, timeout=25)
        if r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n":
            path.write_bytes(r.content)
            return r.content
    except Exception:
        pass
    return None


def prefetch(qid, coords):
    """Concurrently download all not-yet-cached tiles for a question."""
    todo = [(x, y) for x, y in coords
            if not (_tile_path(qid, x, y, Z).exists()
                    and _tile_path(qid, x, y, Z).stat().st_size > 0)]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda xy: tile_fetch(qid, xy[0], xy[1], Z), todo))


def box_mean(a, k):
    """Mean of a over a k x k window, zero-padded, via a summed-area table."""
    p = np.pad(a, k // 2, mode="constant")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    h, w = a.shape
    return (c[k:k + h, k:k + w] - c[0:h, k:k + w]
            - c[k:k + h, 0:w] + c[0:h, 0:w]) / float(k * k)


def density(cov, box):
    return -np.log1p(-np.clip(box_mean(cov, box), 0.0, MAX_FRACTION))


def build_reprojection(x0, x1, y0, y1, z):
    """Separable mercator-mosaic -> Harvard-grid mapping matrices R, C.

    Both projections are separable (mercator column <-> lon, mercator row <->
    lat), so a mosaic pixel maps to a Harvard cell by an independent row and
    column lookup. R (GH x Hm) and C (GW x Wm) are 0/1 assignment matrices, so
    R @ mosaic @ C.T sums each answer's coverage into Harvard cells.
    """
    Wm, Hm = (x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256
    n = 256 * 2 ** z

    cols = np.arange(Wm)
    lon = (x0 * 256 + cols + 0.5) / n * 360.0 - 180.0
    hcol = np.rint((lon - LON_B) / LON_M).astype(int)
    cvalid = (hcol >= 0) & (hcol < GW)

    rows = np.arange(Hm)
    yg = y0 * 256 + rows + 0.5
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * yg / n))))
    hrow = np.rint((lat - LAT_B) / LAT_M).astype(int)
    rvalid = (hrow >= 0) & (hrow < GH)

    C = np.zeros((GW, Wm), dtype=np.float32)
    C[hcol[cvalid], cols[cvalid]] = 1.0
    R = np.zeros((GH, Hm), dtype=np.float32)
    R[hrow[rvalid], rows[rvalid]] = 1.0

    count = R.sum(1)[:, None] * C.sum(1)[None, :]
    # native-pixel lon/lat lookups for writing cells
    return R, C, count, lon, lat


def tile_coverage(data, active_rgb, channels):
    """Per-answer fractional coverage for one tile, shape (k, 256, 256).

    Straight-alpha model: coverage = alpha/255, colour identified by RGB.
      * "Pure" pixels (RGB within PURE_THRESH of one active palette colour) are
        assigned their full coverage to that answer -- this captures solid dot
        cores and anti-aliased edges (which keep a pure hue), including genuine
        secondary-colour (yellow/magenta/cyan/black) answers.
      * "Blend" pixels (an overlap of different-coloured dots) are decomposed
        into the active PRIMARY answers by their premultiplied R/G/B channels,
        so an overlap of two primaries is split back to those two primaries
        instead of being mis-read as the secondary colour that is their sum.
    """
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    arr = np.asarray(img, dtype=np.float32)
    a = arr[..., 3] / 255.0
    rgb = arr[..., :3]
    k = len(active_rgb)
    cov = np.zeros((k,) + a.shape, dtype=np.float32)

    inked = a > MIN_ALPHA
    if not inked.any():
        return cov

    d2 = ((rgb[..., None, :] - active_rgb[None, None, :, :]) ** 2).sum(-1)
    nearest = d2.argmin(-1)
    dist = np.sqrt(d2.min(-1))
    pure = inked & (dist < PURE_THRESH)
    blend = inked & ~pure

    for j in range(k):
        cov[j] = np.where(pure & (nearest == j), a, 0.0)
    if blend.any():
        premult = a[..., None] * (rgb / 255.0)  # coverage per RGB channel
        for j, ch in enumerate(channels):
            if ch is not None:
                cov[j] += np.where(blend, premult[..., ch], 0.0)
    return cov


def process_question(qid, colors, x0, x1, y0, y1, R, C, count, lonv, latv):
    """Return per-colour (hds_cov, hds_density, cells) for one question.

    colors: list of (choice_index, color_hex). cells: list of (lat, lon, cov).
    """
    active_rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
                           for _, h in colors], dtype=np.float32)
    channels = [CHANNEL.get(h) for _, h in colors]
    Wm, Hm = (x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256
    k = len(colors)
    mosaics = [np.zeros((Hm, Wm), dtype=np.float32) for _ in range(k)]

    empty = 0
    prefetch(qid, [(tx, ty) for tx in range(x0, x1 + 1)
                   for ty in range(y0, y1 + 1)])
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            data = tile_fetch(qid, tx, ty, Z)
            if not data:
                empty += 1
                continue
            cov = tile_coverage(data, active_rgb, channels)
            if not cov.any():
                continue
            r0, c0 = (ty - y0) * 256, (tx - x0) * 256
            for j in range(k):
                mosaics[j][r0:r0 + 256, c0:c0 + 256] = cov[j]

    results = []
    for j, (ci, hexc) in enumerate(colors):
        M = mosaics[j]
        hds_sum = R @ M @ C.T
        with np.errstate(invalid="ignore", divide="ignore"):
            hds_cov = np.where(count > 0, hds_sum / count, 0.0).astype(np.float32)
        hds_den = density(hds_cov, BOX).astype(np.float32)

        # native inked cells (sparse), with a saturation-corrected density
        nat_den = density(M, BOX_NATIVE)
        ys, xs = np.nonzero(M > CELL_MIN_COV)
        cells = list(zip(latv[ys], lonv[xs], M[ys, xs], nat_den[ys, xs]))
        results.append((ci, hexc, hds_cov, hds_den, cells, float(M.sum())))
    return results, empty


def main():
    only = None
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        only = {int(a) for a in sys.argv[1:]}

    geo = out_dir("cambridge/geo")
    with open(DATA / "cambridge" / "answers.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_q = defaultdict(list)
    for r in rows:
        by_q[int(r["id"])].append((int(r["choice_index"]), r["color_hex"]))
    qids = sorted(by_q)
    if only:
        qids = [q for q in qids if q in only]

    x0, x1, y0, y1 = conus_tile_range(Z)
    ntiles = (x1 - x0 + 1) * (y1 - y0 + 1)
    print(f"zoom {Z}: CONUS tiles x[{x0}..{x1}] y[{y0}..{y1}] = {ntiles}/question")
    print(f"processing {len(qids)} questions")
    R, C, count, lonv, latv = build_reprojection(x0, x1, y0, y1, Z)

    grids, dens, q_out, c_out = [], [], [], []
    index = []
    cell_buf = io.StringIO()
    cw = csv.writer(cell_buf)
    cw.writerow(["question", "choice", "lat", "lon", "coverage", "density"])
    total_cells = 0

    for n, qid in enumerate(qids, 1):
        colors = sorted(by_q[qid])
        results, empty = process_question(qid, colors, x0, x1, y0, y1,
                                          R, C, count, lonv, latv)
        letters = "abcdefghijklmnop"
        for ci, hexc, hds_cov, hds_den, cells, tot in results:
            choice = letters[ci - 1]
            grids.append(hds_cov)
            dens.append(hds_den)
            q_out.append(str(qid))
            c_out.append(choice)
            index.append((qid, choice, ci, hexc, round(tot, 2), len(cells)))
            for lat, lon, cov, den in cells:
                cw.writerow([qid, choice, f"{lat:.4f}", f"{lon:.4f}",
                             f"{cov:.3f}", f"{den:.5f}"])
                total_cells += 1
        if n % 20 == 0 or n == len(qids):
            print(f"  {n}/{len(qids)} (q{qid}: {len(colors)} answers, "
                  f"{empty}/{ntiles} empty tiles)")

    grids = np.array(grids, dtype=np.float16)
    dens = np.array(dens, dtype=np.float32)
    lats = np.arange(GH, dtype=np.float32) * LAT_M + LAT_B
    lons = np.arange(GW, dtype=np.float32) * LON_M + LON_B

    np.savez_compressed(
        geo / "grid.npz",
        grid=grids, density=dens,
        question=np.array(q_out), choice=np.array(c_out),
        lats=lats, lons=lons,
    )
    with gzip.open(geo / "cells.csv.gz", "wt", encoding="utf-8") as f:
        f.write(cell_buf.getvalue())
    with open(geo / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "choice", "choice_index", "color_hex",
                    "total_coverage", "n_cells"])
        w.writerows(index)

    print(f"\nanswers={len(grids)} grids={grids.shape} native_cells={total_cells}")
    print(f"wrote {geo/'grid.npz'} ({(geo/'grid.npz').stat().st_size/1e6:.1f} MB)")
    print(f"wrote {geo/'cells.csv.gz'} "
          f"({(geo/'cells.csv.gz').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
