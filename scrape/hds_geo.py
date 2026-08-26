"""Recover the geography of the Harvard Dialect Survey from its dot maps.

Each q_N_M.gif plots every respondent who chose answer M at their ZIP centroid,
on a plate carree projection of the lower 48. The ZIP-level microdata was never
published, so these pixels are the only surviving sub-state record.

The projection was solved by least squares against straight state borders whose
longitude/latitude is known exactly (the 49th parallel, 37N, 41N, 42N, 45N,
102.05W, 103W, 104.05W, 109.05W, 114.05W, 88.47W). Residuals are under 7 km,
which is smaller than one pixel (~12.7 km).

Outputs:
  data/hds/geo/cells.csv.gz  - every inked cell: question, choice, lat, lon, coverage
  data/hds/geo/grid.npz      - dense coverage grids, shape (n_answers, 200, 456)
"""

import csv
import gzip
import io

import numpy as np
from PIL import Image

from common import RAW, out_dir

# pixel -> degrees (plate carree); see module docstring
LON_M, LON_B = 0.12664511, -125.42846
LAT_M, LAT_B = -0.12718663, 49.21165
W, H = 456, 200

# The survey used X11 colour names in <font color="...">.
X11 = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0),
    "purple": (160, 32, 240), "orange": (255, 165, 0), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "brown": (165, 42, 42), "yellow": (255, 255, 0),
    "pink": (255, 192, 203), "lightblue": (173, 216, 230),
    "lightgreen": (144, 238, 144),
}

MIN_ALPHA = 0.15
MAX_RESIDUAL = 40.0

# Dots merge where respondents are dense, so coverage saturates towards 1 and
# undercounts cities. Treating respondents as Poisson within a local window,
# expected coverage is 1-exp(-lambda), so -ln(1-f) recovers relative density.
BOX = 9
MAX_FRACTION = 0.995


def box_mean(a, k):
    """Mean of a over a k x k window, zero-padded, via a summed-area table."""
    p = np.pad(a, k // 2, mode="constant")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    h, w = a.shape
    return (c[k:k + h, k:k + w] - c[0:h, k:k + w]
            - c[k:k + h, 0:w] + c[0:h, 0:w]) / float(k * k)


def density(cov):
    """Saturation-corrected relative respondent density from fractional coverage."""
    return -np.log1p(-np.clip(box_mean(cov, BOX), 0.0, MAX_FRACTION))


def coverage(img, colour):
    """Un-blend each pixel against white to recover fractional dot coverage.

    A plotted dot is drawn in `colour` over a white background and antialiased,
    so an edge pixel is alpha*colour + (1-alpha)*white. Solving for alpha in the
    least-squares sense recovers sub-pixel coverage and rejects pixels that are
    not on that line (basemap greys, other answers' colours).
    """
    a = np.asarray(img, dtype=np.float32)
    white = np.float32(255.0)
    d = white - np.asarray(colour, dtype=np.float32)
    denom = float(d @ d)
    alpha = ((white - a) @ d) / denom
    recon = white - alpha[..., None] * d
    residual = np.linalg.norm(recon - a, axis=2)
    alpha = np.where((alpha > MIN_ALPHA) & (residual < MAX_RESIDUAL), alpha, 0.0)
    return np.clip(alpha, 0.0, 1.0)


def main():
    hds = out_dir("hds")
    geo = out_dir("hds/geo")

    with open(hds / "answers.csv", encoding="utf-8") as f:
        answers = [a for a in csv.DictReader(f) if a["has_map"] == "1"]

    cols = np.arange(W, dtype=np.float32)
    rows = np.arange(H, dtype=np.float32)
    lons = cols * LON_M + LON_B
    lats = rows * LAT_M + LAT_B

    grids = np.zeros((len(answers), H, W), dtype=np.float32)
    dens = np.zeros((len(answers), H, W), dtype=np.float32)
    index, mismatches, empty = [], [], []
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["question", "choice", "lat", "lon", "coverage", "density"])

    for i, ans in enumerate(answers):
        q, ci, name = ans["question"], ans["choice_index"], ans["color"]
        path = RAW / "hds_maps" / f"q_{q}_{ci}.gif"
        img = Image.open(path).convert("RGB")
        if img.size != (W, H):
            raise ValueError(f"{path.name}: unexpected size {img.size}")

        colour = X11[name]
        cov = coverage(img, colour)

        # sanity check: the declared colour should actually be the dominant ink
        arr = np.asarray(img, dtype=np.int16)
        sat = (arr.max(axis=2) - arr.min(axis=2)) > 60
        if sat.sum():
            vals, counts = np.unique(arr[sat].reshape(-1, 3), axis=0, return_counts=True)
            dom = tuple(int(v) for v in vals[counts.argmax()])
            if sum(abs(x - y) for x, y in zip(dom, colour)) > 60:
                mismatches.append((q, ci, name, colour, dom))

        grids[i] = cov
        dens[i] = density(cov)
        if cov.sum() == 0:
            empty.append((q, ci, name))

        ys, xs = np.nonzero(cov)
        for y, x in zip(ys, xs):
            w.writerow([q, ans["choice"], f"{lats[y]:.4f}", f"{lons[x]:.4f}",
                        f"{cov[y, x]:.3f}", f"{dens[i, y, x]:.5f}"])

        index.append((q, ans["choice"], ci, name, float(cov.sum())))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(answers)}")

    with gzip.open(geo / "cells.csv.gz", "wt", encoding="utf-8") as f:
        f.write(buf.getvalue())

    np.savez_compressed(
        geo / "grid.npz",
        grid=grids.astype(np.float16),
        density=dens.astype(np.float32),
        question=np.array([r[0] for r in index]),
        choice=np.array([r[1] for r in index]),
        lats=lats, lons=lons,
    )

    with open(geo / "index.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["question", "choice", "choice_index", "color", "total_coverage"])
        cw.writerows(index)

    n_cells = buf.getvalue().count("\n") - 1
    print(f"\nanswers={len(answers)} inked cells={n_cells}")
    print(f"colour mismatches: {len(mismatches)}  empty maps: {len(empty)}")
    for m in mismatches[:10]:
        print(f"  q{m[0]} choice {m[1]} declared {m[2]}{m[3]} but dominant ink {m[4]}")
    for e in empty[:10]:
        print(f"  empty: q{e[0]} choice {e[1]} ({e[2]})")


if __name__ == "__main__":
    main()
