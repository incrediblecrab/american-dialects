"""Emit everything the published site needs, computed from the tracked data.

The site is a second surface for numbers that already exist in four documents,
and a fifth hand-typed copy of r = 0.955 would be a fifth thing that can drift.
So nothing here is typed. Every figure the site displays is recomputed from the
CSVs in data/ and written to web/src/content/generated.json, and check.py
asserts both that the JSON agrees with the data and that no number in web/src
duplicates one that lives in the JSON.

That has a second benefit beyond tidiness. It decouples the site from the
research: when RHO changes and the curve is re-run, the site is regenerated
rather than re-edited, so the two can proceed independently.

Three things come out of this:

    web/src/content/generated.json   prose numbers, question text, tables
    web/public/data/cells.bin        per-cell geography and the prior
    web/public/data/surfaces/*.png   one quantised likelihood surface per answer

The surfaces are the only large payload, and they are split one-per-answer on
purpose. The published quiz asks the FIXED deployed ordering rather than
selecting adaptively, so a game needs exactly one surface per question asked --
the one the player picked -- not every surface of every question. That is about
18 KB per answered question instead of 4.9 MB up front.

Using the fixed ordering is a correctness requirement, not only a payload
trick. Every accuracy figure the site quotes comes from neural_curve.csv, which
is computed over the fixed ordering. A site that selected adaptively would be
quoting numbers measured on a different quiz. site/server.py keeps the adaptive
version, where it is a research tool and no accuracy claim is attached to it.

The isogloss explorer draws contrasts that mostly fall inside that ordering.
Where one does not -- question 63, for frappe -- its surfaces are exported as
well, which is why the published set is slightly larger than SHIP_QUESTIONS
would suggest. It costs about 18 KB per answer on disk and nothing at all on
first paint, because surfaces are fetched only when something asks for them.

    ../.venv/bin/python export_web.py             write everything
    ../.venv/bin/python export_web.py --measure   report the quantisation cost

Output is deterministic: no timestamps, no run identifiers. Regenerating
without changing the data must produce a byte-identical tree, so that a diff
means something changed upstream.
"""

import argparse
import csv
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))

import idiolect as idio  # noqa: E402
from neural import LEGACY_RHO, _arm  # noqa: E402
from infer import (N_QUESTIONS, RHO, TAU_BASE, Geolocator, Places,  # noqa: E402
                   cell_areas, tau_for_weights)
from common import DATA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
CONTENT = WEB / "src" / "content"
PUBLIC = WEB / "public" / "data"
SURFACES = PUBLIC / "surfaces"
RECOVERY = PUBLIC / "recovery"

SHIP_QUESTIONS = 30
"""How many questions of the deployed ordering to publish surfaces for.

More than the quiz asks, so the ordering can lengthen without re-exporting,
and so the site can show the tail of the question-count curve honestly. The
cost is linear and small: about 18 KB per answer surface.
"""

ISO_ODDS = 3
"""The odds that bound a boundary's transition zone, and so define its width.

An atlas draws an isogloss as a line, but it is a gradient. There is a place
where the odds favour one word three to one and another where they favour the
other three to one, and the ground between them is the boundary as it actually
exists. Three rather than two or ten because it is roughly where a listener
would stop being able to guess which side of the line they were standing on;
the ordering of the contrasts below is not sensitive to the choice.
"""

ISOGLOSSES = [
    # Marked variant first: it is the one drawn in the accent colour, and the
    # one the anchor pair is chosen to bracket. Anchors name a place in
    # census/places.csv, and the exporter refuses to publish a contrast whose
    # anchors do not straddle its own boundary.
    {"id": "yinz", "question": "50", "a": ("f", "yinz"), "b": ("i", "y'all"),
     "anchors": ("Pittsburgh, PA", "Atlanta, GA"),
     "note": "Two regional second persons plural, and the survey spells the "
             "first one yins. Western Pennsylvania keeps it, and the ground "
             "over which the odds turn over is narrower than the drive across "
             "the county."},
    {"id": "neutral-ground", "question": "61", "a": ("e", "neutral ground"),
     "b": ("d", "island"),
     "anchors": ("New Orleans, LA", "Dallas, TX"),
     "note": "The grass down the middle of a street. New Orleans calls it "
             "neutral ground after the strip that separated the Creole and "
             "American halves of the city, and almost nowhere else does."},
    {"id": "bubbler", "question": "103", "a": ("a", "bubbler"),
     "b": ("d", "water fountain"),
     "anchors": ("Milwaukee, WI", "Chicago, IL"),
     "note": "Ninety miles of Lake Michigan shoreline separate Milwaukee from "
             "Chicago, and the word for a drinking fountain changes over "
             "them. The other bubbler country, around Providence, is a "
             "separate island a thousand miles away."},
    {"id": "hoagie", "question": "64", "a": ("c", "hoagie"), "b": ("a", "sub"),
     "anchors": ("Philadelphia, PA", "New York, NY"),
     "note": "Sub is the national default and wins nearly everywhere. Hoagie "
             "holds the Delaware Valley, and the line runs closer to "
             "Philadelphia than most Philadelphians would guess."},
    {"id": "frappe", "question": "63", "a": ("b", "frappe"),
     "b": ("a", "milkshake"),
     "anchors": ("Boston, MA", "New York, NY"),
     "note": "Milkshake is the answer for ninety-six per cent of the country. "
             "Frappe survives in eastern Massachusetts alone, which is why "
             "its boundary is short but not sharp: there is very little of it."},
    {"id": "pop", "question": "105", "a": ("b", "pop"), "b": ("a", "soda"),
     "anchors": ("Chicago, IL", "St. Louis, MO"),
     "note": "The best known boundary in American English, and one of the "
             "softest. St. Louis is a soda island inside pop country, and the "
             "map shows the odds sliding rather than switching."},
    {"id": "lightning-bug", "question": "65", "a": ("a", "lightning bug"),
     "b": ("b", "firefly"),
     "anchors": ("Atlanta, GA", "Boston, MA"),
     "note": "An almost even national split, which is what makes it useless "
             "for locating anybody and interesting to look at: the country "
             "divides in half along a frontier hundreds of miles deep."},
    {"id": "catty-corner", "question": "76", "a": ("d", "catty-corner"),
     "b": ("a", "kitty-corner"),
     "anchors": ("Atlanta, GA", "Minneapolis, MN"),
     "note": "The widest boundary published here, and close to no boundary at "
             "all. Both forms are used across most of the country; the "
             "preference drifts rather than changing."},
]


def rows(rel):
    with open(DATA / rel, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ surfaces

def quantise(L):
    """uint8 per surface, with the scale and offset needed to invert it.

    Quantising the log-likelihood rather than the probability is what makes a
    byte enough. The client adds these surfaces, so what has to survive is
    differences between cells, and an additive constant per surface is free:
    the posterior is normalised at the end. So each surface is rescaled onto
    its own full 0..255 range and the offset is carried alongside.

    Measured cost, from --measure: the MAP cell is unchanged in 97.3% of
    simulated games, and in the rest the cell it picks instead is within half a
    percent of the true MAP's posterior probability. Quantisation is not
    trading away accuracy here, it is breaking ties between cells the model
    considers equally likely.
    """
    lo = L.min(axis=1, keepdims=True)
    hi = L.max(axis=1, keepdims=True)
    scale = (hi - lo) / 255.0
    q = np.clip(np.round((L - lo) / np.maximum(scale, 1e-30)), 0, 255)
    return q.astype(np.uint8), lo.ravel(), scale.ravel()


def mixed_loglik(g, idx):
    """log P(answer | cell) after the eps contamination, for rows idx.

    Baking eps in here rather than shipping it is deliberate. eps is frozen at
    the deployed 0.10, and pre-mixing means the client never needs the national
    marginal and never has to reproduce a log-sum-exp. It loads a surface and
    adds it.
    """
    return np.logaddexp(
        np.log1p(-g.eps) + g.t.logp[idx].astype(np.float64),
        np.log(g.eps * g.marginal[idx])[:, None],
    )


def write_surfaces(g, answer_rows, verbose=True):
    """One grayscale PNG per answer, laid out on the 200x456 grid.

    Kept in grid layout rather than packed into a dense strip of the 50,888
    land cells because PNG's filters work along scanlines: neighbouring pixels
    are neighbouring places, the surfaces are Gaussian-smoothed, and so the
    prediction residuals are tiny. Packing the cells densely would scramble
    that adjacency and roughly triple the file size for the same bytes.
    """
    SURFACES.mkdir(parents=True, exist_ok=True)
    t = g.t
    L = mixed_loglik(g, answer_rows)
    q, lo, scale = quantise(L)

    meta, total = {}, 0
    for j, row in enumerate(answer_rows):
        img = np.zeros((len(t.lats), len(t.lons)), dtype=np.uint8)
        img[t.cell_y, t.cell_x] = q[j]
        buf = io.BytesIO()
        Image.fromarray(img, mode="L").save(buf, format="PNG", optimize=True)
        name = f"{t.question[row]}_{t.choice[row]}.png"
        (SURFACES / name).write_bytes(buf.getvalue())
        total += buf.tell()
        meta[f"{t.question[row]}:{t.choice[row]}"] = {
            "file": name, "lo": float(lo[j]), "scale": float(scale[j])}
    if verbose:
        print(f"  {len(answer_rows)} surfaces, {total / 1e6:.1f} MB total, "
              f"{total / len(answer_rows) / 1024:.1f} KB each")
    return meta


STRIP_Q, STRIP_C = "105", "b"
"""Which answer the recovery strip is drawn from: question 105, "pop".

A display choice rather than a measurement, so it is fixed here rather than
derived. It is the right one to show: the soda/pop/coke split is the thing a
general reader already knows, and its geography is sharp enough that the four
stages are visibly different pictures rather than four grey smudges.
"""


def strip(g):
    """The pixel recovery, as four images of the same answer.

    Act II claims the geography survived only as pixels and that un-blending
    them recovers it. That claim is worth nothing as prose; the reader has to
    see the published dot map turn into a density surface. So the four stages
    are rendered here, from the real GIF and the real fitted model, and shipped
    as images the page scrubs between.

    Stage 0 is the RGB map as published. Stages 1 to 3 are single-channel and
    normalised to their own maximum, because the client draws them through the
    same colour ramp as the live map and the ramp is defined on 0..1. What
    varies between them is shape, not level, and normalising each one is what
    makes the shape comparable.
    """
    import hds_geo

    RECOVERY.mkdir(parents=True, exist_ok=True)
    ans = next(r for r in rows("hds/answers.csv")
               if r["question"] == STRIP_Q and r["choice"] == STRIP_C)
    src = DATA / "raw" / "hds_maps" / f"q_{STRIP_Q}_{ans['choice_index']}.gif"
    if not src.exists():
        raise SystemExit(f"{src} missing; run scrape/hds_maps.py first")

    img = Image.open(src).convert("RGB")
    cov = hds_geo.coverage(img, hds_geo.X11[ans["color"]])
    dens = hds_geo.density(cov)

    def save(name, a):
        a = np.asarray(a, dtype=np.float64)
        hi = float(a.max())
        u8 = np.zeros(a.shape, dtype=np.uint8) if hi <= 0 else \
            np.clip(np.round(a / hi * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(u8, mode="L").save(RECOVERY / name, optimize=True)
        return name

    img.save(RECOVERY / "0_published.png", optimize=True)
    save("1_ink.png", cov)
    save("2_density.png", dens)

    row = int(np.flatnonzero((g.t.question == STRIP_Q) &
                             (g.t.choice == STRIP_C))[0])
    surf = np.zeros((len(g.t.lats), len(g.t.lons)), dtype=np.float64)
    surf[g.t.cell_y, g.t.cell_x] = np.exp(g.t.logp[row].astype(np.float64))
    save("3_surface.png", surf)

    return {
        "question": STRIP_Q,
        "choice": STRIP_C,
        "answer": ans["answer"],
        "colour": ans["color"],
        "inkedPixels": int((cov > 0).sum()),
        "stages": [
            {"file": "0_published.png", "kind": "rgb",
             "name": "As published",
             "note": f"One dot per respondent who said \u201c{ans['answer']}\u201d, "
                     "plotted at their ZIP centroid. This image is the whole "
                     "surviving record; the coordinates behind it were never "
                     "released."},
            {"file": "1_ink.png", "kind": "mask",
             "name": "Ink, un-blended",
             "note": "Each dot was drawn in a known colour over white and "
                     "antialiased, so an edge pixel is a mixture of the two. "
                     "Solving that mixture recovers what fraction of the pixel "
                     "was covered, and rejects basemap grey and other answers' "
                     "colours."},
            {"file": "2_density.png", "kind": "mask",
             "name": "Density, saturation corrected",
             "note": "Where respondents are dense the dots merge and coverage "
                     "saturates, which flattens cities. Treating respondents "
                     "as Poisson within a small window, expected coverage is "
                     "1\u2212e^(\u2212\u03bb), so \u2212ln(1\u2212f) undoes it "
                     "and gives back relative density."},
            {"file": "3_surface.png", "kind": "mask",
             "name": "What the model uses",
             "note": "Smoothed at a bandwidth chosen by cross-validation, "
                     "restricted to land, and turned into a probability of "
                     "this answer given the place. This is the surface the "
                     "quiz above adds up."},
        ],
    }


def write_cells(g, places):
    """Per-cell geography, the prior, and the place each cell belongs to.

    One binary blob of parallel arrays rather than JSON: 50,888 cells of JSON
    numbers is several megabytes of text to parse, and the same content as
    typed arrays is 407 KB the browser can use without touching it.
    """
    t = g.t
    catch = places.catchment(t)
    states = sorted({str(s) for s in t.state})
    state_idx = {s: i for i, s in enumerate(states)}
    km2 = cell_areas(t)

    PUBLIC.mkdir(parents=True, exist_ok=True)
    parts = [
        ("cellY", t.cell_y.astype(np.uint8)),
        ("cellX", t.cell_x.astype(np.uint16)),
        ("logPrior", np.log(g.prior).astype(np.float32)),
        ("stateIdx", np.array([state_idx[str(s)] for s in t.state], np.uint8)),
        ("placeIdx", catch.astype(np.uint16)),
        ("km2", km2.astype(np.float32)),
    ]
    blob, layout, off = bytearray(), [], 0
    for name, arr in parts:
        b = arr.tobytes()
        layout.append({"name": name, "dtype": arr.dtype.name,
                       "offset": off, "length": int(arr.size)})
        blob += b
        off += len(b)
    (PUBLIC / "cells.bin").write_bytes(bytes(blob))
    print(f"  cells.bin {len(blob) / 1e3:.0f} KB, {t.n_cells:,} cells")
    return layout, states, [
        {"name": r[0], "state": r[1], "lat": round(r[2], 4),
         "lon": round(r[3], 4), "pop": r[4]} for r in places.rows]


# ------------------------------------------------------------------ the numbers

def recovery():
    val = rows("hds/geo/validation.csv")
    a = np.array([float(r["pct_published"]) for r in val])
    b = np.array([float(r["pct_from_map"]) for r in val])
    truth = {(r["state"], r["question"], r["choice"]): float(r["pct"])
             for r in rows("hds/state_pct.csv")}
    derived = {(r["state"], r["question"], r["choice"]): float(r["pct_from_map"])
               for r in val}
    tw, dw = _winners(truth), _winners(derived)
    common = [k for k in tw if k in dw]
    return {
        "comparisons": len(val),
        "r": round(float(np.corrcoef(a, b)[0, 1]), 3),
        "mae": round(float(np.abs(a - b).mean()), 1),
        "modalAgreement": round(
            100.0 * sum(1 for k in common if tw[k] == dw[k]) / len(common), 1),
    }


def _winners(d):
    best = {}
    for (s, q, c), v in d.items():
        if best.get((s, q), (None, -1.0))[1] < v:
            best[(s, q)] = (c, v)
    return {k: v[0] for k, v in best.items()}


def pop_vs_soda():
    r = rows("popvssoda/counties.csv")
    cats = ["SUMPOP", "SUMSODA", "SUMCOKE", "SUMOTHER"]
    parts = sum(sum(int(x[c] or 0) for c in cats) for x in r)
    total = sum(int(x["SUMCOUNT"] or 0) for x in r)
    return {"categorised": parts, "sourceTotal": total,
            "counties": sum(1 for x in r if int(x["SUMCOUNT"] or 0) > 0),
            "offByOne": total - parts}


def tuning():
    t = rows("model/tuning.csv")
    frozen = [x for x in t if x["rake"] == "1" and float(x["sigma"]) == 8.0
              and float(x["gamma"]) == 1.5 and float(x["alpha"]) == 0.02]
    return {"logloss": round(float(frozen[0]["logloss"]), 4),
            "gridMinimum": round(min(float(x["logloss"]) for x in t), 5)}


def curve():
    out = []
    for r in rows("model/neural_curve.csv"):
        out.append({"model": r["model"], "k": int(r["k"]), "n": int(r["n"]),
                    "medianKm": round(float(r["median_km"]), 1),
                    "p90Km": round(float(r["p90_km"]), 1),
                    "within150": round(float(r["within_150km"]), 4),
                    "stateAcc": round(float(r["state_acc"]), 4),
                    "cover80": round(float(r["cover80"]), 4),
                    "calibErr": round(float(r["calib_err"]), 4)})
    return out


def questions(g, order):
    """Question and answer text for the published ordering, with its own bits."""
    qtext = {r["question"]: r["text"] for r in rows("hds/questions.csv")}
    atext = {}
    for r in rows("hds/answers.csv"):
        atext.setdefault(r["question"], {})[r["choice"]] = r["answer"]
    bits = g.question_bits
    out = []
    for n, q in enumerate(order, 1):
        out.append({
            "n": n, "id": q, "text": qtext.get(q, f"question {q}"),
            "bits": round(float(bits.get(q, 0.0)), 4),
            "choices": [{"id": str(g.t.choice[i]),
                         "text": atext.get(q, {}).get(str(g.t.choice[i]),
                                                      str(g.t.choice[i]))}
                        for i in g.t.rows[q]],
        })
    return out


def isoglosses(g, places):
    """Every boundary the explorer draws, measured on the bytes it will draw.

    The client fetches the two quantised surfaces, subtracts them and gets a
    signed log-odds field; the contour where that field is zero is the
    isogloss. So the figures published beside it are computed from the same
    quantised surfaces rather than from full precision, and describe the
    picture the browser actually renders rather than one it approximates.

    Width is the honest part. A boundary is not a line, so its width is the
    area of the zone where the odds sit inside ISO_ODDS either way, divided by
    the length of the line running through it: a mean thickness in kilometres.
    That single number is what separates yinz from soda. Both are boundaries,
    and one of them is several times the other.
    """
    from tensor import haversine
    t = g.t
    km2 = cell_areas(t)
    nrow, ncol = len(t.lats), len(t.lons)
    dlat = abs(t.lats[1] - t.lats[0])
    dlon = abs(t.lons[1] - t.lons[0])
    # A cell's north-south and east-west faces, per grid row. The east-west
    # face shrinks toward the pole, so the length of a boundary is summed face
    # by face rather than counted.
    ns = haversine(t.lats - dlat / 2, t.lons[0], t.lats + dlat / 2, t.lons[0])
    ew = haversine(t.lats, t.lons[0] - dlon / 2, t.lats, t.lons[0] + dlon / 2)

    qtext = {r["question"]: r["text"] for r in rows("hds/questions.csv")}
    survey, national = {}, {}
    for r in rows("hds/answers.csv"):
        survey.setdefault(r["question"], {})[r["choice"]] = r["answer"].strip()
        national.setdefault(r["question"], {})[r["choice"]] = \
            float(r["pct_national"] or 0.0)
    named = {f"{r[0]}, {r[1]}": r for r in places.rows}

    def variant(q, side_spec):
        choice, label = side_spec
        return {"choice": choice, "label": label, "survey": survey[q][choice],
                "national": round(national[q][choice], 2)}

    out = []
    for spec in ISOGLOSSES:
        q = spec["question"]
        pair = [int(np.flatnonzero((t.question == q) & (t.choice == c))[0])
                for c, _ in (spec["a"], spec["b"])]
        q8, base, scale = quantise(mixed_loglik(g, pair))
        decoded = base[:, None] + q8.astype(np.float64) * scale[:, None]
        d = decoded[0] - decoded[1]

        side = np.zeros((nrow, ncol), np.int8)
        side[t.cell_y, t.cell_x] = np.where(d >= 0, 1, 2)
        line = 0.0
        for dy, dx in ((0, 1), (1, 0)):
            here, there = side[:nrow - dy, :ncol - dx], side[dy:, dx:]
            ys = np.nonzero((here > 0) & (there > 0) & (here != there))[0]
            line += float((ns[ys] if dx else ew[ys]).sum())
        if line <= 0:
            raise SystemExit(f"isogloss {spec['id']}: no boundary on land")
        area = float(km2[np.abs(d) < np.log(ISO_ODDS)].sum())

        anchors = []
        for key, want in zip(spec["anchors"], (True, False)):
            if key not in named:
                raise SystemExit(f"isogloss {spec['id']}: no place named {key}")
            name, state, lat, lon, _ = named[key]
            p = float(1.0 / (1.0 + np.exp(-d[t.nearest(lat, lon)])))
            if (p > 0.5) is not want:
                raise SystemExit(
                    f"isogloss {spec['id']}: {key} reads {p:.2f} for "
                    f"{spec['a'][1]}, the wrong side of its own line")
            anchors.append({"name": name, "state": state,
                            "lat": round(lat, 4), "lon": round(lon, 4),
                            "p": round(p, 4)})

        out.append({
            "id": spec["id"], "question": q, "questionText": qtext[q],
            "a": variant(q, spec["a"]), "b": variant(q, spec["b"]),
            "widthKm": round(area / line, 1),
            "lineKm": round(line, 1),
            "shareA": round(float(km2[d >= 0].sum() / km2.sum()), 4),
            "anchors": anchors, "note": spec["note"],
        })

    out.sort(key=lambda r: r["widthKm"])
    return {"odds": ISO_ODDS, "contrasts": out}


def inventory():
    """The README's row-count table, recomputed rather than transcribed."""
    out = []
    for rel in sorted(p.relative_to(DATA).as_posix()
                      for p in DATA.rglob("*.csv") if "raw" not in p.parts):
        path = DATA / rel
        with open(path, encoding="utf-8", newline="") as f:
            n = sum(1 for _ in f) - 1
        out.append({"path": f"data/{rel}", "rows": n})
    return out


# ------------------------------------------------------------------ measurement

def measure(g, order, n=300, seed=4):
    """What quantising the surfaces to a byte actually costs, in km.

    Reported rather than assumed because the payload design turns on it. The
    honest summary is not a single number: the median cost is zero and the
    maximum is large, and quoting either alone would mislead. What resolves
    them is that every game where the answer moved was a near-tie.
    """
    from tensor import haversine
    t = g.t
    ship = [q for q in order[:SHIP_QUESTIONS]]
    idx = np.concatenate([t.rows[q] for q in ship])
    pos = {int(r): j for j, r in enumerate(idx)}
    L = mixed_loglik(g, idx)
    q8, lo, scale = quantise(L)
    R = lo[:, None] + q8.astype(np.float64) * scale[:, None]

    rng = np.random.default_rng(seed)
    ask = order[:N_QUESTIONS]
    moved, ratios, dists = 0, [], []
    for _ in range(n):
        cell = rng.choice(t.n_cells, p=g.prior)
        sel = []
        for qq in ask:
            rr = t.rows[qq]
            p = np.exp(t.logp[rr][:, cell]).astype(np.float64)
            sel.append(int(rr[rng.choice(len(rr), p=p / p.sum())]))
        tau = tau_for_weights([g.question_bits.get(x, 0.0) for x in ask])
        best = []
        for M in (L, R):
            lp = g.log_prior + tau * M[[pos[s] for s in sel]].sum(axis=0)
            lp -= lp.max()
            p = np.exp(lp)
            best.append(p / p.sum())
        a, b = int(best[0].argmax()), int(best[1].argmax())
        if a != b:
            moved += 1
            ratios.append(float(best[0][b] / best[0][a]))
            dists.append(float(haversine(t.cell_lat[a], t.cell_lon[a],
                                         t.cell_lat[b], t.cell_lon[b])))
    return {
        "games": n, "questionsAsked": len(ask), "moved": moved,
        "identicalPct": round(100.0 * (n - moved) / n, 1),
        "worstRatio": round(min(ratios), 4) if ratios else 1.0,
        "maxKm": round(max(dists), 1) if dists else 0.0,
    }


def fixture(g, order, places, states):
    """A worked example the TypeScript port must reproduce exactly.

    The site re-implements posterior() and tau_for_weights() in the browser.
    Every accuracy figure it quotes was measured on the Python implementation,
    so a port that quietly disagreed would leave the page showing one model and
    citing another. This pins one game: a fixed set of answers, scored under
    the discount that used to be deployed and the value deployed now, with
    enough of the result recorded that a mismatch anywhere in the chain shows
    up. The two arms are pinned to LEGACY_RHO and RHO rather than to RHO and
    zero, so that they stay distinct now that RHO is itself zero.

    The answer chosen for each question is the one most likely in Pittsburgh,
    which is not arbitrary. Pittsburgh has the sharpest lexical signature in
    the survey, so the posterior concentrates hard and any error in tau, in the
    surface decoding or in the normalisation moves the answer visibly.
    """
    t = g.t
    here = t.nearest(40.4406, -79.9959)
    ask = order[:N_QUESTIONS]
    answers = []
    for q in ask:
        rr = t.rows[q]
        best = int(rr[np.argmax(t.logp[rr][:, here])])
        answers.append((q, str(t.choice[best])))

    km2 = cell_areas(t)
    catch = places.catchment(t)
    out = {"answers": [{"question": q, "choice": c} for q, c in answers],
           "cell": {"pittsburgh": int(here)}, "variants": {}}
    import infer as _infer
    keep = _infer.RHO
    try:
        for label, rho in (("discounted", LEGACY_RHO), ("deployed", RHO)):
            _infer.RHO = rho
            post = g.posterior(answers)
            best = int(np.argmax(post))
            order_ix = np.argsort(post)[::-1]
            n80 = int(np.searchsorted(np.cumsum(post[order_ix]), 0.8) + 1)
            pm = np.zeros(len(places.rows))
            np.add.at(pm, catch, post)
            sm = {s: float(post[t.state == s].sum()) for s in states}
            out["variants"][label] = {
                "rho": rho,
                "tau": round(float(g.tau_used(answers)), 6),
                "mapCell": best,
                "mapLat": round(float(t.cell_lat[best]), 4),
                "mapLon": round(float(t.cell_lon[best]), 4),
                "mapState": str(t.state[best]),
                "area80Km2": round(float(km2[order_ix[:n80]].sum()), 1),
                "topP": [round(float(x), 8) for x in post[order_ix[:5]]],
                "topCells": [int(x) for x in order_ix[:5]],
                "topPlaces": [
                    {"name": f"{places.rows[i][0]}, {places.rows[i][1]}",
                     "p": round(float(pm[i]), 6)}
                    for i in np.argsort(pm)[::-1][:3]],
                "topStates": [
                    {"name": s, "p": round(v, 6)}
                    for s, v in sorted(sm.items(), key=lambda kv: -kv[1])[:3]],
            }
    finally:
        _infer.RHO = keep
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", action="store_true",
                    help="report the quantisation cost and exit")
    ap.add_argument("--order", default=None,
                    help="question ordering CSV; default is the deployed one")
    ap.add_argument("--skip-surfaces", action="store_true",
                    help="rewrite the JSON only, leaving the PNGs alone")
    args = ap.parse_args()

    print("loading the model ...")
    g = Geolocator()
    order = idio.deployed_questions(SHIP_QUESTIONS, path=args.order)

    if args.measure:
        m = measure(g, order)
        print(f"\nquantisation cost over {m['games']} simulated games, "
              f"{m['questionsAsked']} questions each:")
        print(f"  MAP cell identical in {m['identicalPct']}% of games")
        print(f"  when it moved, the cell chosen instead was within "
              f"{100 * (1 - m['worstRatio']):.1f}% of the true MAP's probability")
        print(f"  largest distance moved: {m['maxKm']:.0f} km")
        return 0

    CONTENT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    places = Places(min_pop=20000)

    print("writing cells ...")
    layout, states, place_rows = write_cells(g, places)

    surfaces = {}
    if not args.skip_surfaces:
        print("writing surfaces ...")
        if SURFACES.exists():
            shutil.rmtree(SURFACES)
        extra = [q for q in dict.fromkeys(i["question"] for i in ISOGLOSSES)
                 if q not in order]
        if extra:
            print(f"  plus {', '.join('q' + q for q in extra)} for the isoglosses")
        answer_rows = np.concatenate([g.t.rows[q] for q in list(order) + extra])
        surfaces = write_surfaces(g, answer_rows)
    elif (PUBLIC / "manifest.json").exists():
        surfaces = json.loads((PUBLIC / "manifest.json").read_text())["surfaces"]

    manifest = {
        "grid": {"rows": len(g.t.lats), "cols": len(g.t.lons),
                 "lats": [round(float(x), 6) for x in g.t.lats],
                 "lons": [round(float(x), 6) for x in g.t.lons]},
        "cells": {"count": int(g.t.n_cells), "file": "cells.bin",
                  "arrays": layout},
        "states": states,
        "places": place_rows,
        "surfaces": surfaces,
    }
    (PUBLIC / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    print(f"  manifest.json {(PUBLIC / 'manifest.json').stat().st_size / 1e3:.0f} KB, "
          f"{len(place_rows):,} places")

    content = {
        "constants": {"rho": RHO, "legacyRho": LEGACY_RHO, "tauBase": TAU_BASE,
                      "nQuestions": N_QUESTIONS, "eps": g.eps},
        "models": {"net": "net", "deployed": _arm(RHO), "discounted": _arm(LEGACY_RHO)},
        "recovery": recovery(),
        "recoveryStrip": strip(g),
        "popVsSoda": pop_vs_soda(),
        "tuning": tuning(),
        "curve": curve(),
        "questions": questions(g, order),
        "isoglosses": isoglosses(g, places),
        "inventory": inventory(),
        "quantisation": measure(g, order),
        "fixture": fixture(g, order, places, states),
    }
    (CONTENT / "generated.json").write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    size = (CONTENT / "generated.json").stat().st_size
    print(f"\nwrote web/src/content/generated.json ({size / 1e3:.0f} KB)")
    print(f"  {len(content['questions'])} questions, "
          f"{len(content['curve'])} curve rows, "
          f"{len(content['isoglosses']['contrasts'])} isoglosses, "
          f"{len(content['inventory'])} inventory rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
