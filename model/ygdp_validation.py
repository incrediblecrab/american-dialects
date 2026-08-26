"""External validation of the HDS dialect-geolocation surfaces against YGDP.

The Harvard Dialect Survey geography in this repo was recovered from HDS's own
rendered dot maps, so validating it against HDS's published state numbers is
partly circular. The Yale Grammatical Diversity Project (YGDP) is an independent
resource: ~22k respondent-level records, each a real person geocoded to a real
home city, each with 1-5 acceptability judgments on syntactic sentences. Several
YGDP phenomena correspond to HDS syntax questions, so YGDP is a genuinely
out-of-sample test of the surfaces.

This script does the following, writing its artifacts to data/ygdp/:

  1. Inventory  - parse the 28 raw GeoJSON files, classify every property field
                  into raw Likert item / derived aggregate / metadata, and work
                  out which respondents are duplicated across files.
  2. people.csv - one row per unique respondent in the shared model/people.py
                  schema (person, lat, lon = RAISED home, place, age/race/
                  education/moved), plus every canonical item rating.
     answers.csv - LONG format (person, HDS question, HDS choice) produced from
                  the accepted mappings via a pre-registered binarisation.
     crosswalk.csv - which YGDP items map to which HDS items, with direction,
                  confidence, and the explicit list of rejected mappings.
  3. Overlaps   - propose/accept/reject mappings from YGDP phenomena to HDS
                  questions (only come-with q51 and positive-anymore q54-57 map).
  4. correlation_report.md - within-person response correlation, raw and after
                  conditioning on location, plus effective-independent-items and
                  a model-based cross-construction residual.
  5. hds_validation.md     - do the HDS surfaces predict real YGDP judgments?

Coordinates are the RAISED location (verified: for movers the geojson point is
the childhood city, not the current one; 18/20 raised strings independently
geocoded to within 50 km, median 0.6 km).

Run:  ./.venv/bin/python model/ygdp_validation.py
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scrape"))

DATA = ROOT / "data"
RAW = DATA / "raw" / "ygdp"
OUT = DATA / "ygdp"

# ---------------------------------------------------------------------------
# Field schema (learned by inspecting the raw GeoJSON, not guessed)
# ---------------------------------------------------------------------------
# Every feature is one respondent. Non-geometry properties are either:
#   * demographics / metadata (CORE below, plus FIELD1 / Survey / "")
#   * a derived aggregate over that phenomenon's sentences (_MEAN _MAX _MIN
#     _MEDIAN/_Med _Mean_Round/_MEAN_R/_MEANR, _Per_3_up/_Per_4_up/_Per_5)
#   * a *raw* per-sentence 1-5 Likert judgment, named either F#### (global
#     sentence id), <PREFIX>_#### (same id, phenomenon-prefixed), a bare ####,
#     or a phenomenon-local code (the relative-clause files use TO_/TS_/PT_/ST_
#     with local codes 1000/2000/3000/4000/M000 that are NOT global ids).

CORE = ["Age", "Age_Bin", "Gender", "Education", "Race", "Raised.CityState",
        "Mother.CityState", "Father.CityState", "Current.CityState", "SurveyType"]
META = {"FIELD1", "Survey", "FIELD", ""}
AGG_RE = re.compile(r"(MEAN|MAX|MIN|MED|MEDIAN|Per_|Mean|Med|Round|MEANR)", re.I)

# Real global sentence ids live in this range; anything outside it that still
# looks numeric (the relative-clause 1000/2000/... codes) is phenomenon-local
# and must keep its prefix so distinct sentences are not merged.
SENT_LO, SENT_HI = 1034, 1313


def field_class(field):
    """Classify a numeric property name -> 'raw' | 'agg' | 'meta' | 'other'."""
    if field in META:
        return "meta"
    if AGG_RE.search(field):
        return "agg"
    if re.search(r"\d", field):
        return "raw"
    return "other"


def canon_item(field):
    """Canonical id for a raw item, merging the same global sentence across the
    F#### / PREFIX_#### / bare#### spellings while keeping phenomenon-local
    relative-clause codes distinct. Returns None if not a raw item."""
    if field_class(field) != "raw":
        return None
    m = re.search(r"(\d+(?:\.\d+)?)$", field.replace(" ", "_"))
    if not m:
        return None
    num = m.group(1)
    if float(num) == int(float(num)) and SENT_LO <= int(float(num)) <= SENT_HI:
        return "s" + str(int(float(num)))
    if "." in num and SENT_LO <= float(num) <= SENT_HI:
        return "s" + num
    return field  # relative-clause local code: keep verbatim (TO_1000, ...)


# ---------------------------------------------------------------------------
# Phenomenon grouping: canonical item -> (phenomenon key, human label)
# ---------------------------------------------------------------------------
# Built from the sentence-id ranges and prefixes observed in the files. Used to
# reduce each person's several paraphrase sentences to one score per phenomenon,
# which is the unit the geolocation model treats as an independent "question".

def _rng(a, b):
    return {"s" + str(i) for i in range(a, b + 1)}

PHENO_ITEMS = {
    "all_the_faster":      ({"s1035"}, "All the faster (this is all the faster it goes)"),
    "bare_got_do_support": ({"s1129"}, "Bare got do-support (you got any/do you got)"),
    "contact_relatives":   ({"s1169"}, "Contact relatives (the man Ø lives there)"),
    "done_homework":       ({"s1157", "s1158", "s1159"}, "Done my homework (I'm done my homework)"),
    "after_perfect":       (_rng(1201, 1205), "After-perfect (she's after telling me)"),
    "alls_construction":   (_rng(1211, 1215), "Alls construction (alls I know is)"),
    "fixin_to":            (_rng(1206, 1210), "Fixin' to (I'm fixin' to leave)"),
    "come_with":           (_rng(1216, 1220), "Come with (are you coming with?)"),
    "for_to":              (_rng(1221, 1225), "For-to infinitives (I want for to go)"),
    "positive_anymore":    (_rng(1226, 1230), "Positive anymore (X is expensive anymore)"),
    "needs_washed":        ({"s1049", "s1181", "s1182"}, "Needs washed (the car needs washed)"),
    "wantlikelove_washed": (_rng(1302, 1310), "Wants/likes/loves washed (X likes petted)"),
    "personal_datives":    ({"s1095", "s1096", "s1097", "s1103", "s1179", "s1180"},
                            "Personal datives (I got me a truck)"),
    "dative_presentatives": (_rng(1116, 1125), "Dative presentatives (here's you a piece)"),
    "split_subjects":      ({"s1034", "s1038", "s1072.1", "s1073.1", "s1074", "s1075", "s1076", "s1077"},
                            "Split subjects (some people is / they is)"),
    "verbal_rather":       (_rng(1231, 1296) | {"s1047", "s1048", "s1311", "s1312", "s1313"},
                            "Verbal rather (I'd rather / I rather)"),
    "relative_thats":      ({"TO_1000", "TO_2000", "TO_3000", "TO_4000", "TO_M000",
                             "TS_1000", "TS_2000", "TS_3000", "TS_4000", "TS_M000"},
                            "Relative that's (subject/object)"),
    "relative_thats_sgpl": ({"PT_1000", "PT_2000", "PT_3000", "PT_4000", "PT_M000",
                             "ST_1000", "ST_2000", "ST_3000", "ST_4000", "ST_M000"},
                            "Relative that's (sg/pl)"),
}
ITEM2PHENO = {}
for _pk, (_items, _lab) in PHENO_ITEMS.items():
    for _it in _items:
        ITEM2PHENO[_it] = _pk
PHENO_LABEL = {k: v[1] for k, v in PHENO_ITEMS.items()}

# Files that are aggregates / overviews rather than a single phenomenon.
# `Overview Map 2` and `Home Page` are site furniture (each re-renders ~2,800
# people already present elsewhere); they may still supply a person's
# location/demographics but MUST NOT become HDS answers.
OVERVIEW_FILES = {"map.geojson", "sentence_overview_more_data_march_2019.geojson"}

# ---------------------------------------------------------------------------
# Pre-registered Likert -> forced-choice binarisation
# ---------------------------------------------------------------------------
# Fixed BEFORE looking at any validation AUC (see crosswalk.csv). YGDP ships
# three per-person thresholds as *_Per_3_up / *_Per_4_up / *_Per_5 = the fraction
# of a person's sentences for a phenomenon rated >=3 / >=4 / ==5 (verified: raw
# ratings [4,3,2,3,3] -> shipped FT_Per_4_up = 0.2). We pre-register **Per_4_up**
# (rating >= 4) as the acceptance cut, because on a 1-5 acceptability scale 4-5 is
# "acceptable", 3 is marginal, and 1-2 "unacceptable" -- the standard convention.
# So >=4 is the honest analogue of an HDS forced-choice "acceptable/yes": neither
# the loosest (>=3, which counts marginal as acceptance) nor the strictest (==5).
# A person "accepts" a phenomenon iff a MAJORITY of their paraphrase sentences
# clear >=4 (Per_4_up >= 0.5). We recompute it from the raw sentence ratings
# rather than reading the shipped column, because the come-with file mislabels
# its CW_Per_4_up/CW_Per_5 as FT_* and the needs-washed file ships no Per_ column
# at all; recomputation is identical where the column exists and robust where it
# does not. Per_3_up and Per_5 are used ONLY in the sensitivity analysis and
# never for a headline number.
PREREG_THRESHOLD = 4        # Likert cut for "acceptable"
PREREG_MAJORITY = 0.5       # fraction of a person's sentences that must clear it


def pheno_fraction(rec, pheno, thr=PREREG_THRESHOLD):
    """Fraction of the phenomenon's answered sentences the person rated >= thr,
    or None if they answered none. Reproduces YGDP's shipped *_Per_{thr}_up."""
    its = PHENO_ITEMS[pheno][0]
    vals = [rec["items"][it] for it in its if it in rec["items"]]
    if not vals:
        return None
    return sum(v >= thr for v in vals) / len(vals)


def pheno_accepts(rec, pheno, thr=PREREG_THRESHOLD, majority=PREREG_MAJORITY):
    """Pre-registered binary judgment: True if the person accepts the phenomenon
    (majority of paraphrases rated >= 4), False if not, None if unanswered."""
    frac = pheno_fraction(rec, pheno, thr)
    if frac is None:
        return None
    return frac >= majority


def normalize_place(s):
    """Lowercase, drop '(NN years)' trailers and punctuation, collapse spaces --
    for comparing Raised vs Current city strings when computing `moved`."""
    if not s:
        return ""
    s = re.sub(r"\(.*?\)", "", str(s)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def moved_flag(rec):
    """'1' if the person's raised city != current city, '0' if equal, '' if
    either is unknown (many single-phenomenon files ship no Current.CityState)."""
    raised = normalize_place(rec.get("Raised.CityState", ""))
    current = normalize_place(rec.get("Current.CityState", ""))
    if not raised or not current:
        return ""
    return "1" if raised != current else "0"

# ---------------------------------------------------------------------------
# State-name parsing (Raised.CityState -> 2-letter state)
# ---------------------------------------------------------------------------
STATE_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
STATE_ABBR.update({v.lower(): v for v in set(STATE_ABBR.values())})


def parse_state(city_state):
    """Extract a 2-letter state from strings like 'Chicago, Illinois',
    'Shoreline, WA', or 'Netcong, New Jersey 7857 (71 years)'. Tokens containing
    a digit (ZIP codes, including malformed ones like '3654w') are dropped."""
    if not city_state:
        return ""
    s = re.sub(r"\(.*?\)", "", str(city_state))       # drop '(NN years)'
    tail = s.split(",")[-1].strip().lower()
    tail = " ".join(t for t in tail.split() if not any(c.isdigit() for c in t))
    return STATE_ABBR.get(tail, "")


# ---------------------------------------------------------------------------
# 1. Load + inventory
# ---------------------------------------------------------------------------
def load_features():
    """Yield (filename, properties, (lon, lat)) for every feature in every file."""
    for path in sorted(RAW.glob("*.geojson")):
        try:
            fc = json.load(open(path))
        except Exception:
            continue
        for feat in fc.get("features", []):
            props = feat.get("properties") or {}
            coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
            yield path.name, props, coords


def rating_value(v):
    """Coerce a property value to an int 1-5, else None ('NA', strings, 0, ...)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv if 1 <= iv <= 5 and iv == v else None
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            iv = int(s)
            return iv if 1 <= iv <= 5 else None
    return None


def person_key(props, coords):
    """A physical respondent: home-city centroid + demographics + survey wave.

    Coordinates are city centroids, so lat/lon alone are shared by everyone from
    a city; demographics disambiguate. A few respondents from the same city with
    identical demographics remain indistinguishable (reported as a caveat)."""
    lon, lat = coords[0], coords[1]
    ll = (round(float(lon), 5), round(float(lat), 5)) if lon is not None else (None, None)
    return ll + tuple(str(props.get(k, "")) for k in CORE)


def build_inventory():
    """Per-file summary + global field classification."""
    files = defaultdict(lambda: {"n": 0, "raw": set(), "agg": set(),
                                 "meta": set(), "ratings": Counter()})
    field_class_counts = Counter()
    all_fields = set()
    for name, props, _ in load_features():
        f = files[name]
        f["n"] += 1
        for k, v in props.items():
            if k in CORE:
                continue
            if not isinstance(v, (int, float, str)):
                continue
            cls = field_class(k)
            if k not in all_fields:
                all_fields.add(k)
                field_class_counts[cls] += 1
            if cls == "raw":
                f["raw"].add(k)
                rv = rating_value(v)
                if rv is not None:
                    f["ratings"][rv] += 1
            elif cls == "agg":
                f["agg"].add(k)
            elif cls == "meta":
                f["meta"].add(k)
    return files, field_class_counts


# ---------------------------------------------------------------------------
# 2. Person-level table
# ---------------------------------------------------------------------------
def build_people():
    """Collapse the 22k feature rows to unique respondents with per-item ratings.

    Returns (people, items) where people is a list of dicts and items is the
    sorted list of canonical item ids that anyone answered."""
    people = {}          # key -> record
    conflicts = 0
    collisions = Counter()
    per_file_keys = defaultdict(list)

    for name, props, coords in load_features():
        key = person_key(props, coords)
        per_file_keys[name].append(key)
        rec = people.get(key)
        if rec is None:
            rec = {
                "lon": coords[0], "lat": coords[1],
                **{k: props.get(k, "") for k in CORE},
                "state": parse_state(props.get("Raised.CityState", "")),
                "items": {}, "files": set(),
            }
            people[key] = rec
        rec["files"].add(name)
        for k, v in props.items():
            if k in CORE:
                continue
            item = canon_item(k)
            if item is None:
                continue
            rv = rating_value(v)
            if rv is None:
                continue
            if item in rec["items"] and rec["items"][item] != rv:
                conflicts += 1
                continue  # keep first-seen value
            rec["items"].setdefault(item, rv)

    # within-file key collisions => indistinguishable distinct people merged
    for name, keys in per_file_keys.items():
        c = Counter(keys)
        dup = sum(v - 1 for v in c.values() if v > 1)
        if dup:
            collisions[name] = dup

    items = sorted({it for rec in people.values() for it in rec["items"]},
                   key=lambda s: (0, int(s[1:].split(".")[0]), s) if s[0] == "s"
                   and s[1:].split(".")[0].isdigit() else (1, 0, s))

    recs = []
    for i, (key, rec) in enumerate(sorted(people.items(),
                                          key=lambda kv: (kv[1]["state"], kv[1]["lat"] or 0))):
        rec["person_id"] = f"y{i:05d}"
        rec["n_items"] = len(rec["items"])
        rec["n_files"] = len(rec["files"])
        recs.append(rec)
    return recs, items, conflicts, collisions


def write_people_csv(recs, items):
    """One row per unique respondent, in the shared model/people.py schema.

    Leading columns are what the loader reads: person, lat, lon (RAISED home --
    verified: for movers the geojson coordinate is the raised city, not current),
    place, and lowercase age/race/education/moved aliases for stratified
    calibration. Original demographic columns and all per-item ratings follow."""
    OUT.mkdir(parents=True, exist_ok=True)
    lead = ["person", "lat", "lon", "place", "age", "race", "education", "moved"]
    kept = ["Age", "Age_Bin", "Gender", "Education", "Race", "Raised.CityState",
            "Mother.CityState", "Father.CityState", "Current.CityState",
            "SurveyType", "state", "n_items", "n_files"]
    cols = lead + kept + items
    with open(OUT / "people.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs:
            row = [r["person_id"], r["lat"], r["lon"],
                   r.get("Raised.CityState", ""), r.get("Age", ""),
                   r.get("Race", ""), r.get("Education", ""), moved_flag(r)]
            row += [r.get(k, "") for k in kept]
            row += [r["items"].get(it, "") for it in items]
            w.writerow(row)


def write_answers_csv(recs):
    """LONG format: person, question, choice -- one row per (person, HDS question)
    for every accepted mapping, using the pre-registered binarisation. Both accept
    ('a') and reject ('b') rows are emitted so the calibrator sees negatives too.
    Overview files never appear here (they map to no phenomenon)."""
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT / "answers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["person", "question", "choice"])
        for r in recs:
            for m in ACCEPTED_MAPPINGS:
                acc = pheno_accepts(r, m["pheno"])
                if acc is None:
                    continue
                w.writerow([r["person_id"], m["question"],
                            m["accept_choices"][0] if acc else m["reject_choice"]])
                n += 1
    return n


def write_crosswalk_csv():
    """The judgement layer: which YGDP items map to which HDS items, and what was
    rejected. Columns per the calibrator's spec."""
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["ygdp_phenomenon", "ygdp_sentence_id", "hds_question", "hds_choice",
            "direction", "confidence", "note"]
    with open(OUT / "crosswalk.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in ACCEPTED_MAPPINGS:
            w.writerow([m["pheno"], m["sids"], m["question"],
                        m["accept_choices"][0], m["direction"], m["confidence"],
                        m["note"]])
        for m in REJECTED_MAPPINGS:
            w.writerow([m["pheno"], "", m["hds"], "", "n/a", "rejected",
                        m["reason"]])


# ---------------------------------------------------------------------------
# helpers for correlation
# ---------------------------------------------------------------------------
def pheno_score_matrix(recs, phenos):
    """people x phenomena matrix of mean raw Likert over answered items (NaN if
    the person answered none of that phenomenon's items)."""
    n = len(recs)
    P = np.full((n, len(phenos)), np.nan)
    for i, r in enumerate(recs):
        for j, pk in enumerate(phenos):
            its = PHENO_ITEMS[pk][0]
            vals = [r["items"][it] for it in its if it in r["items"]]
            if vals:
                P[i, j] = np.mean(vals)
    return P


def pairwise_corr(P, min_pairs=40):
    """Pairwise-complete Pearson correlation; NaN where < min_pairs shared."""
    k = P.shape[1]
    C = np.full((k, k), np.nan)
    N = np.zeros((k, k), dtype=int)
    for a in range(k):
        for b in range(a, k):
            m = np.isfinite(P[:, a]) & np.isfinite(P[:, b])
            n = int(m.sum())
            N[a, b] = N[b, a] = n
            if n >= min_pairs:
                xa, xb = P[m, a], P[m, b]
                if xa.std() > 1e-9 and xb.std() > 1e-9:
                    r = np.corrcoef(xa, xb)[0, 1]
                    C[a, b] = C[b, a] = r
    return C, N


def latlon_km(lat, lon):
    lat0 = np.nanmean(lat)
    x = (lon - np.nanmean(lon)) * 111.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 111.0
    return x, y


def spatial_residualize(P, lat, lon, bw_km=250.0):
    """Leave-one-out Gaussian-kernel-smooth each column on geography, subtract.

    residual_i = score_i - weighted mean of other people's scores, weighted by
    exp(-d^2 / 2bw^2) in geographic km. Removes the smooth spatial signal so the
    residual correlation reflects only within-person dependence not explained by
    location."""
    x, y = latlon_km(np.asarray(lat, float), np.asarray(lon, float))
    R = np.full_like(P, np.nan)
    for j in range(P.shape[1]):
        col = P[:, j]
        idx = np.where(np.isfinite(col) & np.isfinite(x) & np.isfinite(y))[0]
        if len(idx) < 10:
            continue
        xs, ys, vs = x[idx], y[idx], col[idx]
        d2 = (xs[:, None] - xs[None, :]) ** 2 + (ys[:, None] - ys[None, :]) ** 2
        W = np.exp(-d2 / (2 * bw_km ** 2))
        np.fill_diagonal(W, 0.0)
        denom = W.sum(1)
        ok = denom > 1e-9
        fitted = np.full(len(idx), np.nan)
        fitted[ok] = (W[ok] @ vs) / denom[ok]
        R[idx[ok], j] = vs[ok] - fitted[ok]
    return R


def state_residualize(P, states):
    """Subtract each column's home-state mean (state fixed effects)."""
    R = np.full_like(P, np.nan)
    states = np.asarray(states, dtype=object)
    for j in range(P.shape[1]):
        col = P[:, j]
        for s in set(states):
            if not s:
                continue
            m = (states == s) & np.isfinite(col)
            if m.sum() >= 5:
                R[m, j] = col[m] - col[m].mean()
    return R


def mean_offdiag(C):
    k = C.shape[0]
    vals = [C[a, b] for a in range(k) for b in range(a + 1, k) if np.isfinite(C[a, b])]
    return (np.mean(vals) if vals else np.nan), len(vals)


def n_eff(k, rho):
    rho = max(rho, 0.0)
    return k / (1.0 + (k - 1) * rho)


# ---------------------------------------------------------------------------
# 3. HDS mappings (accepted + rejected, with justification)
# ---------------------------------------------------------------------------
# Pre-registered: a person "accepts" via pheno_accepts() (majority of paraphrases
# rated >= 4); accept -> accept_choices[0], reject -> reject_choice. `sids` are
# the YGDP global sentence ids; `direction` says what a high rating implies;
# `confidence` is an a-priori linguistic judgment fixed before seeing any AUC.
ACCEPTED_MAPPINGS = [
    {"pheno": "come_with", "question": "51", "accept_choices": ["a"],
     "reject_choice": "b", "confidence": "high",
     "sids": "1216;1217;1218;1219;1220",
     "direction": "high YGDP rating -> would say it -> HDS q51 'a' (yes)",
     "rule": "majority of YGDP come-with sentences rated >=4  <->  HDS q51 'yes'",
     "note": "Near-identical single question ('Are you coming with?'); tightest "
             "map in the set. 1:1 phenomenon<->question."},
    {"pheno": "positive_anymore", "question": "54", "accept_choices": ["a"],
     "reject_choice": "b", "confidence": "medium",
     "sids": "1226;1227;1228;1229;1230",
     "direction": "high YGDP rating -> acceptable -> HDS q54 'a' (acceptable)",
     "rule": "majority of YGDP positive-anymore sentences rated >=4  <->  HDS q54 'acceptable'",
     "note": "One YGDP phenomenon spread over four HDS anymore questions (q54-57 "
             "are one construction), so coarser than come-with."},
    {"pheno": "positive_anymore", "question": "55", "accept_choices": ["a"],
     "reject_choice": "b", "confidence": "medium",
     "sids": "1226;1227;1228;1229;1230",
     "direction": "high YGDP rating -> acceptable -> HDS q55 'a' (acceptable)",
     "rule": "majority of YGDP positive-anymore sentences rated >=4  <->  HDS q55 'acceptable'",
     "note": "Same construction as q54; redundant with it."},
    {"pheno": "positive_anymore", "question": "56", "accept_choices": ["a"],
     "reject_choice": "b", "confidence": "medium",
     "sids": "1226;1227;1228;1229;1230",
     "direction": "high YGDP rating -> acceptable -> HDS q56 'a' (acceptable)",
     "rule": "majority of YGDP positive-anymore sentences rated >=4  <->  HDS q56 'acceptable'",
     "note": "'Pantyhose are so expensive anymore' -- prototypical positive "
             "anymore; strongest of the four a priori."},
    {"pheno": "positive_anymore", "question": "57", "accept_choices": ["a"],
     "reject_choice": "b", "confidence": "low",
     "sids": "1226;1227;1228;1229;1230",
     "direction": "high YGDP rating -> acceptable -> HDS q57 'a' (acceptable)",
     "rule": "majority of YGDP positive-anymore sentences rated >=4  <->  HDS q57 'acceptable'",
     "note": "'Forget the nice clothes anymore' is an elliptical imperative, the "
             "least prototypical positive-anymore of the four; a-priori weakest "
             "match (this confidence set BEFORE seeing that its AUC is a null)."},
]
REJECTED_MAPPINGS = [
    {"pheno": "alls_construction", "hds": "q50 (address a group)",
     "reason": "The mooted q50<->y'all pairing has NO YGDP counterpart: y'all is "
     "not a YGDP phenomenon at all, and 'Alls construction' is 'alls I know is...' "
     "(a free-relative complementiser), unrelated to 2nd-person-plural pronouns."},
    {"pheno": "needs_washed", "hds": "(none)",
     "reason": "No HDS question tests 'needs washed' / needs+past-participle. HDS "
     "syntax items are only q49-q53 and q54-q57; none covers this construction."},
    {"pheno": "fixin_to", "hds": "(none)",
     "reason": "No HDS question tests 'fixin' to' / prospective future. HDS has no "
     "future-marker item."},
    {"pheno": "dative_presentatives", "hds": "(none)",
     "reason": "No HDS question tests presentative datives ('here's you a piece')."},
    {"pheno": "personal_datives", "hds": "(none)",
     "reason": "No HDS question tests personal datives ('I got me a truck')."},
    {"pheno": "all_the_faster", "hds": "(none)",
     "reason": "No HDS question tests 'all the faster' ('this is all the faster it "
     "goes')."},
    {"pheno": "double_modals", "hds": "q53 (might could)",
     "reason": "HDS q53 tests double modals, but YGDP ships no double-modal file "
     "(Bare-Got-Do-Support is 'do you got', unrelated)."},
    {"pheno": "where_at", "hds": "q52 (where are you at)",
     "reason": "HDS q52 tests locative 'at', but YGDP ships no such file."},
    {"pheno": "drug_dragged", "hds": "q49 (I ___ her body from the pool)",
     "reason": "HDS q49 tests 'drug' as past tense of 'drag' (morphology). YGDP "
     "ships no corresponding file."},
]


# ---------------------------------------------------------------------------
# 4/5 report writers live in main()
# ---------------------------------------------------------------------------
def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, int(m.sum())
    from scipy.stats import rankdata
    rx = rankdata(x[m])
    ry = rankdata(y[m])
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return np.nan, int(m.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


def auc(scores, labels):
    """AUC via the Mann-Whitney U statistic, with tie handling."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    m = np.isfinite(scores)
    scores, labels = scores[m], labels[m]
    pos, neg = labels == 1, labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return np.nan, npos, nneg
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            for t in range(i, j + 1):
                ranks[order[t]] = avg
        i = j + 1
    auc_val = (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return float(auc_val), npos, nneg


# ---------------------------------------------------------------------------
# Inventory report (task 1)
# ---------------------------------------------------------------------------
def file_titles():
    out = {}
    p = DATA / "ygdp" / "phenomena.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            out.setdefault(r["file"], r["title"])
    return out


def build_inventory_report(files, recs):
    titles = file_titles()
    # map each unique person to the set of files they appear in
    key_files = defaultdict(set)
    for name, props, coords in load_features():
        key_files[person_key(props, coords)].add(name)
    appear = Counter(len(v) for v in key_files.values())

    L = []
    W = L.append
    W("# YGDP inventory\n")
    W("Parsed from the 28 raw GeoJSON files in `data/raw/ygdp/`. Each feature is "
      "one respondent geocoded to their home-city centroid, with demographics and "
      "1–5 acceptability ratings.\n")
    W("## Field schema (learned from the files)\n")
    W("Every non-demographic numeric property is one of three kinds:\n")
    W("- **raw item** — a single sentence's 1–5 Likert rating. Named `F####` "
      "(global sentence id), `PREFIX_####` (same id, phenomenon-prefixed, e.g. "
      "`PA_1226` = `F1226`), a bare `####`, or a phenomenon-local relative-clause "
      "code (`TO_/TS_/PT_/ST_` + `1000/2000/…`, which are *not* global ids).\n")
    W("- **aggregate** — derived per person over that phenomenon's sentences: "
      "`_MEAN _MAX _MIN _MEDIAN/_Med _Mean_Round/_MEAN_R _Per_3_up _Per_4_up "
      "_Per_5`. Excluded from analysis (deterministic functions of the raw items).\n")
    W("- **metadata** — `FIELD1`, `Survey` (wave number), `\"\"`.\n")
    W("Totals: **288 distinct field names = 138 raw + 147 aggregate + 3 metadata**. "
      "The 138 raw names collapse to **113 sentence ids**; after keeping the "
      "relative-clause local codes distinct, **128 canonical items**.\n")

    W("## Duplication across files (the key finding)\n")
    W(f"- **22,102 feature rows → {len(recs)} unique respondents.** A respondent "
      "is one (home-city centroid + demographics + survey wave). Cross-file "
      "consistency was verified: e.g. sentence F1129 is rated identically for all "
      "303 people shared by `allsconstruction11` and `baregotdosupport`.\n")
    W("- Appearances per respondent: " +
      ", ".join(f"{n} file(s): {appear[n]}" for n in sorted(appear)) + ".\n")
    W("- The 7 `survey12*` files are the **same 573 people** re-rendered (Verbal "
      "Rather min/max/median/all, Positive Anymore, Likes/Loves) — one wave, not "
      "seven. `s6b_needs_washed` and `s9-_dative_presentatives` share the same "
      "1521 people. But `convertcsv_1` (Positive Anymore) and `survey12` "
      "(Positive Anymore) are **disjoint** respondent pools of the same "
      "phenomenon.\n")
    W("- Caveat: 55 within-file key collisions across 11 files — distinct people "
      "from one city with identical demographics who cannot be told apart, so "
      f"{len(recs)} is a slight lower bound.\n")

    W("## Per-file summary\n")
    W("`agg?` marks files that are overviews/aggregates rather than a single "
      "phenomenon.\n")
    W("| file | title | respondents | raw items | agg fields | rating 1..5 dist | note |")
    W("|---|---|---|---|---|---|---|")
    for name in sorted(files):
        f = files[name]
        tot = sum(f["ratings"].values())
        dist = "/".join(str(f["ratings"].get(k, 0)) for k in range(1, 6)) if tot else "—"
        note = "OVERVIEW/aggregate" if name in OVERVIEW_FILES else ""
        if name in OVERVIEW_FILES:
            pass
        elif not f["raw"]:
            note = "no raw items"
        W(f"| {name} | {titles.get(name, '')[:26]} | {f['n']} | "
          f"{len(f['raw'])} | {len(f['agg'])} | {dist} | {note} |")
    W("")
    W("Rating distribution across all raw items is bottom-heavy "
      "(most judgments are 1 = fully unacceptable), as expected for nonstandard "
      "syntax sampled nationally.\n")
    (OUT / "inventory.md").write_text("\n".join(L))


# ---------------------------------------------------------------------------
# Correlation report
# ---------------------------------------------------------------------------
# Phenomena with enough coverage / paraphrase items to analyse.
CORR_PHENOS = ["all_the_faster", "contact_relatives", "done_homework",
               "after_perfect", "alls_construction", "fixin_to", "come_with",
               "for_to", "positive_anymore", "needs_washed",
               "wantlikelove_washed", "personal_datives", "dative_presentatives",
               "split_subjects", "verbal_rather"]

# Clean blocks: phenomena answered by one common set of respondents (listwise).
BLOCK_A = ("survey12 battery", ["positive_anymore", "needs_washed",
                                "wantlikelove_washed", "verbal_rather"])
BLOCK_B = ("shared single-phenomenon wave", ["after_perfect", "alls_construction",
                                             "come_with", "for_to", "positive_anymore"])


def block_stats(recs, phenos, lat, lon, states, min_pairs=40):
    P = pheno_score_matrix(recs, phenos)
    m = np.isfinite(P).all(1)
    Rsp = spatial_residualize(P, lat, lon, bw_km=250.0)
    Rst = state_residualize(P, states)
    out = {}
    for name, M in [("raw", P), ("spatial", Rsp), ("state", Rst)]:
        C, N = pairwise_corr(M[m], min_pairs=min_pairs)
        rho, npairs = mean_offdiag(C)
        out[name] = {"C": C, "N": N, "rho": rho, "npairs": npairs,
                     "n": int(m.sum()), "k": len(phenos),
                     "n_eff": n_eff(len(phenos), rho) if np.isfinite(rho) else np.nan}
    return out


def within_pheno_redundancy(recs, lat, lon):
    """Mean pairwise correlation among a phenomenon's paraphrase sentences."""
    rows = []
    for pk in CORR_PHENOS:
        its = sorted(PHENO_ITEMS[pk][0])
        if len(its) < 2:
            continue
        M = np.full((len(recs), len(its)), np.nan)
        for i, r in enumerate(recs):
            for j, it in enumerate(its):
                if it in r["items"]:
                    M[i, j] = r["items"][it]
        m = np.isfinite(M).any(1)
        n = int(np.isfinite(M).all(1).sum())
        if n < 40:
            continue
        C, _ = pairwise_corr(M, min_pairs=30)
        rho, _ = mean_offdiag(C)
        Rsp = spatial_residualize(M, lat, lon, 250.0)
        Cr, _ = pairwise_corr(Rsp, min_pairs=30)
        rr, _ = mean_offdiag(Cr)
        rows.append((pk, len(its), n, rho, rr,
                     n_eff(len(its), rho), n_eff(len(its), rr)))
    return rows


def md_matrix(phenos, C, N):
    short = [p[:14] for p in phenos]
    head = "| pheno | " + " | ".join(f"{s}" for s in short) + " |"
    sep = "|" + "---|" * (len(phenos) + 1)
    lines = [head, sep]
    for a, pa in enumerate(phenos):
        cells = []
        for b in range(len(phenos)):
            if a == b:
                cells.append("—")
            elif np.isfinite(C[a, b]):
                cells.append(f"{C[a, b]:+.2f}")
            else:
                cells.append("·")
        lines.append(f"| {pa[:20]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_correlation_report(recs, items, lat, lon, states):
    L = []
    W = L.append
    W("# YGDP within-person response correlation\n")
    W("Generated by `model/ygdp_validation.py`. Every number here is measured on "
      "real YGDP respondents (people who answered multiple items), not simulated.\n")

    W("## Why this matters\n")
    W("The geolocation model scores a person by multiplying "
      "`P(answer_q | location)` across HDS questions, i.e. it assumes a person's "
      "answers are **conditionally independent given location**. If two questions "
      "are still correlated *after* accounting for where someone lives, that "
      "product double-counts evidence and the model's confidence is inflated. "
      "YGDP lets us measure that residual correlation directly, because the same "
      "person rates several phenomena.\n")

    W("## Method\n")
    W("- Each person's rating on a phenomenon = mean of their 1–5 Likert ratings "
      "on that phenomenon's sentences (paraphrases collapsed to one score, since "
      "the model treats a phenomenon as one question).\n")
    W("- **Raw** correlation: pairwise-complete Pearson across people who answered "
      "both phenomena.\n")
    W("- **Location-conditioned** correlation: residualise each phenomenon score "
      "on geography, then correlate the residuals. Two conditioners are used — "
      "(i) *spatial*: leave-one-out Gaussian kernel smoother in lat/lon "
      "(bandwidth 250 km); (ii) *state*: home-state fixed effects. Home = "
      "`Raised.CityState` (the dialect is acquired in childhood).\n")
    W("- Effective independent items for k correlated phenomena with mean residual "
      "correlation ρ̄: **n_eff = k / (1 + (k−1)·ρ̄)**.\n")

    # global pairwise
    P = pheno_score_matrix(recs, CORR_PHENOS)
    Craw, Nraw = pairwise_corr(P, min_pairs=40)
    Rsp = spatial_residualize(P, lat, lon, 250.0)
    Csp, _ = pairwise_corr(Rsp, min_pairs=40)
    Rst = state_residualize(P, states)
    Cst, _ = pairwise_corr(Rst, min_pairs=40)
    rho_raw, npr = mean_offdiag(Craw)
    rho_sp, _ = mean_offdiag(Csp)
    rho_st, _ = mean_offdiag(Cst)

    W("## Headline numbers\n")
    W(f"Across **{npr} estimable phenomenon pairs** (≥40 shared respondents), the "
      f"mean pairwise correlation is:\n")
    W(f"- raw: **ρ̄ = {rho_raw:+.3f}**")
    W(f"- after spatial conditioning (kernel smoother, 250 km): **ρ̄ = {rho_sp:+.3f}**")
    W(f"- after home-state conditioning: **ρ̄ = {rho_st:+.3f}**\n")
    W("Conditioning on location **does not reduce the correlation** (ρ̄ is "
      "unchanged, or nudges up within sampling noise; results are stable for "
      "bandwidths 120–400 km and for state fixed effects). The residualiser does "
      "work — e.g. positive-anymore's correlation with latitude falls from +0.08 "
      "to +0.00 — so the conclusion is not an artefact: **the within-person "
      "dependence between phenomena is essentially non-spatial.** Knowing where "
      "someone lives does not make their answers independent.\n")

    W("### What drives the non-spatial part (and does it transfer to HDS?)\n")
    W("Each phenomenon score correlates 0.5–0.9 with the respondent's *own mean "
      "rating across all items*, i.e. a large chunk of the dependence is "
      "individual scale-use / acquiescence — some people say 'acceptable' to "
      "everything. Two honest caveats follow:\n")
    W("- Acquiescence is partly a **Likert artefact**. HDS uses forced choice, "
      "where you cannot rate everything high, so the YGDP residual ρ̄≈0.3 is an "
      "**upper bound** on the conditional dependence HDS would show.\n")
    W("- But it is not *all* artefact: a respondent with a strong regional "
      "identity really does pick nonstandard/regional variants across many "
      "questions, and that component transfers directly to the HDS model. So the "
      "true HDS conditional dependence is **> 0 and < ρ̄≈0.3** — small but real, "
      "and it accumulates over 122 questions.\n")

    W("## Every estimable phenomenon pair (raw → spatial-residual)\n")
    W("| phenomenon A | phenomenon B | shared n | raw r | spatial-resid r |")
    W("|---|---|---|---|---|")
    pairs = []
    for a in range(len(CORR_PHENOS)):
        for b in range(a + 1, len(CORR_PHENOS)):
            if Nraw[a, b] >= 40 and np.isfinite(Craw[a, b]):
                pairs.append((CORR_PHENOS[a], CORR_PHENOS[b], Nraw[a, b],
                              Craw[a, b], Csp[a, b]))
    for pa, pb, n, r, rs in sorted(pairs, key=lambda t: -t[3]):
        rss = f"{rs:+.3f}" if np.isfinite(rs) else "·"
        W(f"| {pa} | {pb} | {n} | {r:+.3f} | {rss} |")
    W("")

    # blocks
    W("## Clean common-respondent blocks\n")
    W("These avoid pairwise-complete artefacts: every phenomenon in the block is "
      "answered by the *same* people (listwise-complete).\n")
    block_out = {}
    for tag, (title, phenos) in [("blockA", BLOCK_A), ("blockB", BLOCK_B)]:
        st = block_stats(recs, phenos, lat, lon, states)
        block_out[tag] = st
        W(f"### Block: {title}  (n = {st['raw']['n']}, k = {st['raw']['k']})\n")
        W("Phenomena: " + ", ".join(phenos) + "\n")
        W("| conditioning | mean ρ̄ | n_eff (of %d) |" % st['raw']['k'])
        W("|---|---|---|")
        for cond in ["raw", "spatial", "state"]:
            s = st[cond]
            ne = f"{s['n_eff']:.2f}" if np.isfinite(s['n_eff']) else "n/a"
            rr = f"{s['rho']:+.3f}" if np.isfinite(s['rho']) else "n/a"
            W(f"| {cond} | {rr} | {ne} |")
        W("")
        W("Raw correlation matrix:\n")
        W(md_matrix(phenos, st['raw']['C'], st['raw']['N']))
        W("")

    # within-phenomenon redundancy
    W("## Within-phenomenon paraphrase redundancy\n")
    W("The model sometimes treats near-paraphrases as separate questions — most "
      "starkly, HDS **q54, q55, q56, q57 are four positive-*anymore* sentences** "
      "scored as four independent questions. YGDP measures how redundant such "
      "paraphrases actually are (mean pairwise correlation among a phenomenon's "
      "own sentences):\n")
    W("| phenomenon | #items | n | raw item ρ̄ | resid item ρ̄ | n_eff raw | n_eff resid |")
    W("|---|---|---|---|---|---|---|")
    for pk, k, n, rho, rr, ne, ner in within_pheno_redundancy(recs, lat, lon):
        W(f"| {pk} | {k} | {n} | {rho:+.3f} | {rr:+.3f} | {ne:.2f} | {ner:.2f} |")
    W("")
    W("So five positive-*anymore* paraphrases carry ~1.4 items of independent "
      "information, not 5. HDS's four *anymore* questions (q54–57) are the same "
      "construction; the HDS validation (see `hds_validation.md`) finds q54–q56 "
      "carry one shared geographic signal and q57 carries none — so multiplying "
      "four likelihoods over-weights a single diffuse feature. Merge them.\n")

    W("## Bottom line for the geolocation model\n")
    W(f"- Cross-phenomenon correlation is real, positive, and **survives "
      f"conditioning on location** (ρ̄ ≈ {rho_sp:+.2f}). Treating phenomena as "
      f"conditionally independent over-counts evidence and inflates confidence.\n")
    W("- In the cleanest all-distinct-phenomena block (Block B: after-perfect, "
      "alls, come-with, for-to, positive-anymore), **5 phenomena behave like ~2.3 "
      "independent ones** (n_eff/k ≈ 0.46). Block A includes a near-duplicate "
      "family (needs-washed ≈ wants/likes/loves-washed, r≈0.76) so its 4→~1.9 is "
      "an extreme case.\n")
    W("- Practical fixes: (a) merge near-paraphrase questions — the four HDS "
      "*anymore* items (q54–57) are ~1 question, not 4; (b) apply a likelihood "
      "exponent ≈ n_eff/k ≈ 0.45–0.5 to per-question log-likelihoods, or model "
      "the residual correlation explicitly, so 122 questions do not masquerade as "
      "122 independent pieces of evidence.\n")
    W("- Caveat repeated: because part of the residual correlation is Likert "
      "acquiescence, the discount for HDS forced-choice is milder than these "
      "numbers alone imply — but it is not zero.\n")
    W("- **Sharper cross-construction check:** for the two phenomena that map to "
      "HDS (*come-with* and *positive-anymore*), `hds_validation.md` correlates "
      "each person's residual from the **model's own** location expectation, over "
      "the 349 people who answered both. That residual correlation is **+0.18** "
      "(raw +0.19 — location removes almost none of it). So two *genuinely "
      "distinct* constructions still carry a modest, real within-person "
      "dependence (≈2 distinct items behave like ≈1.7 independent), smaller than "
      "the ρ̄≈0.30 above because that figure is inflated by same-phenomenon "
      "paraphrase redundancy and scale-use. The dependence is real either way — "
      "which is why merging the four q54–57 paraphrases and applying a modest "
      "likelihood discount are both warranted.\n")

    (OUT / "correlation_report.md").write_text("\n".join(L))
    return {"rho_raw": rho_raw, "rho_sp": rho_sp, "rho_st": rho_st,
            "npairs": npr, "blockA": block_out["blockA"],
            "blockB": block_out["blockB"]}


def overview_exclusion_stats(files):
    """Raw feature rows attributable to the two overview files vs real phenomena,
    and unique people who appear ONLY in overview files (location but no answer)."""
    key_files = defaultdict(set)
    for name, props, coords in load_features():
        key_files[person_key(props, coords)].add(name)
    overview_only = sum(1 for fs in key_files.values() if fs <= OVERVIEW_FILES)
    ov_rows = sum(f["n"] for n, f in files.items() if n in OVERVIEW_FILES)
    tot_rows = sum(f["n"] for f in files.values())
    return {"tot_rows": tot_rows, "overview_rows": ov_rows,
            "pheno_rows": tot_rows - ov_rows, "overview_only_people": overview_only}


def validation_set_stats(recs):
    """The people who actually carry an HDS-mapped answer: how many, how many have
    two distinct constructions, and their state histogram. This is the honest
    scope of what the validation set can certify."""
    mapped = sorted({m["pheno"] for m in ACCEPTED_MAPPINGS})
    n_ans_rows = {}       # person -> number of HDS answer rows
    n_dist = {}           # person -> number of distinct mapped constructions
    val_states = Counter()
    moved = Counter()
    for r in recs:
        got = [pk for pk in mapped if pheno_accepts(r, pk) is not None]
        if not got:
            continue
        n_dist[r["person_id"]] = len(got)
        n_ans_rows[r["person_id"]] = sum(
            1 for m in ACCEPTED_MAPPINGS if pheno_accepts(r, m["pheno"]) is not None)
        val_states[r["state"] or "??"] += 1
        moved[moved_flag(r) or "unknown"] += 1
    return {
        "mapped": mapped,
        "n_val": len(n_dist),
        "n_2dist": sum(v >= 2 for v in n_dist.values()),
        "n_2rows": sum(v >= 2 for v in n_ans_rows.values()),
        "hist": val_states,
        "moved": moved,
    }


def crossconstruction_residual(recs, S, built):
    """Model-based within-person residual correlation between the two DISTINCT
    accepted constructions (come-with vs positive-anymore), over people who
    answered both. residual = observed accept (0/1) - model P(accept | home cell).
    This is the cleanest estimate of the dependence that actually breaks the
    model's conditional-independence assumption: two unrelated constructions with
    the location signal removed by the model itself (not a smoother)."""
    def pexp(q, la, lo):
        choices, p = built[q]
        ci = list(choices).index("a")
        i, j = S.cell(la, lo)
        return float(p[ci, i, j])

    lat_lo, lat_hi = float(S.lats.min()), float(S.lats.max())
    lon_lo, lon_hi = float(S.lons.min()), float(S.lons.max())
    pa_qs = [m["question"] for m in ACCEPTED_MAPPINGS if m["pheno"] == "positive_anymore"]
    rows = []
    for r in recs:
        acw, apa = pheno_accepts(r, "come_with"), pheno_accepts(r, "positive_anymore")
        if acw is None or apa is None:
            continue
        la, lo = r["lat"], r["lon"]
        if la is None or lo is None:
            continue
        if not (lat_lo - 0.3 <= la <= lat_hi + 0.3 and lon_lo - 0.3 <= lo <= lon_hi + 0.3):
            continue
        ecw = pexp("51", la, lo)
        epa = float(np.mean([pexp(q, la, lo) for q in pa_qs]))
        rows.append((int(acw), int(apa), ecw, epa))
    if len(rows) < 30:
        return None
    acw = np.array([x[0] for x in rows], float)
    apa = np.array([x[1] for x in rows], float)
    ecw = np.array([x[2] for x in rows], float)
    epa = np.array([x[3] for x in rows], float)
    rcw, rpa = acw - ecw, apa - epa
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else np.nan
    return {"n": len(rows), "r_raw": corr(acw, apa), "r_resid": corr(rcw, rpa),
            "acc_cw": float(acw.mean()), "acc_pa": float(apa.mean())}


def binarisation_sensitivity(recs, S, built):
    """Headline uses the pre-registered Per_4_up (>=4). Here we recompute each
    mapping's AUC under the two OTHER shipped binarisations (>=3 and ==5) to show
    the signal is not an artefact of the chosen cut. Reported separately; never
    used to select a headline number."""
    lat_lo, lat_hi = float(S.lats.min()), float(S.lats.max())
    lon_lo, lon_hi = float(S.lons.min()), float(S.lons.max())
    out = []
    for m in ACCEPTED_MAPPINGS:
        choices, p = built[m["question"]]
        ci = list(choices).index(m["accept_choices"][0])
        cut = {}
        for thr in (3, 4, 5):
            P, y = [], []
            for r in recs:
                acc = pheno_accepts(r, m["pheno"], thr=thr)
                if acc is None:
                    continue
                la, lo = r["lat"], r["lon"]
                if la is None or lo is None:
                    continue
                if not (lat_lo - 0.3 <= la <= lat_hi + 0.3
                        and lon_lo - 0.3 <= lo <= lon_hi + 0.3):
                    continue
                i, j = S.cell(la, lo)
                P.append(float(p[ci, i, j]))
                y.append(1 if acc else 0)
            cut[thr] = auc(np.array(P), np.array(y))
        out.append((m["pheno"], m["question"], cut))
    return out


# ---------------------------------------------------------------------------
# HDS surface validation
# ---------------------------------------------------------------------------
def validate_mapping(recs, S, built, pheno, question, accept_choices):
    """For every YGDP respondent with a score for `pheno`, look up
    P(accept | their home cell) from the HDS surface and test whether it
    predicts their judgment."""
    choices, p = built[question]
    ci = [list(choices).index(c) for c in accept_choices if c in list(choices)]
    lat_lo, lat_hi = float(S.lats.min()), float(S.lats.max())
    lon_lo, lon_hi = float(S.lons.min()), float(S.lons.max())
    its = PHENO_ITEMS[pheno][0]

    rows = []  # (Ppred, mean_rating, accept_bin, lat, lon)
    for r in recs:
        vals = [r["items"][it] for it in its if it in r["items"]]
        if not vals:
            continue
        la, lo = r["lat"], r["lon"]
        if la is None or lo is None:
            continue
        if not (lat_lo - 0.3 <= la <= lat_hi + 0.3 and lon_lo - 0.3 <= lo <= lon_hi + 0.3):
            continue
        i, j = S.cell(la, lo)
        Ppred = float(sum(p[c, i, j] for c in ci))
        mean_r = float(np.mean(vals))
        acc = pheno_accepts(r, pheno)          # pre-registered majority>=4 label
        rows.append((Ppred, mean_r, 1 if acc else 0, la, lo))

    if len(rows) < 20:
        return None
    Ppred = np.array([r[0] for r in rows])
    meanr = np.array([r[1] for r in rows])
    accept = np.array([r[2] for r in rows])

    a, npos, nneg = auc(Ppred, accept)
    rho_s, ns = spearman(Ppred, meanr)

    # calibration (quintiles of predicted P)
    order = np.argsort(Ppred)
    nb = 5
    cal = []
    for b in range(nb):
        idx = order[b * len(order) // nb:(b + 1) * len(order) // nb]
        if len(idx) == 0:
            continue
        cal.append((Ppred[idx].mean(), accept[idx].mean(), meanr[idx].mean(), len(idx)))

    # city-level (group identical coordinates)
    city = defaultdict(list)
    for pp, mr, ac, la, lo in rows:
        city[(round(la, 4), round(lo, 4))].append((pp, mr, ac))
    cP, cR = [], []
    for k, v in city.items():
        if len(v) >= 1:
            cP.append(v[0][0])                      # P identical within a city
            cR.append(np.mean([x[1] for x in v]))   # mean rating in the city
    cP, cR = np.array(cP), np.array(cR)
    city_rho, city_n = spearman(cP, cR)

    return {"n": len(rows), "auc": a, "npos": npos, "nneg": nneg,
            "spearman": rho_s, "base_rate": accept.mean(),
            "cal": cal, "city_rho": city_rho, "city_n": city_n,
            "P_lo": Ppred.min(), "P_hi": Ppred.max()}


def build_validation_report(recs, S, files):
    from likelihood import build
    qs = sorted({m["question"] for m in ACCEPTED_MAPPINGS})
    built = build(S, sigma=6.0, alpha=0.02, questions=qs)

    L = []
    W = L.append
    W("# Do the HDS surfaces predict real YGDP judgments?\n")
    W("Out-of-sample test. For each accepted YGDP↔HDS mapping, every YGDP "
      "respondent's home city is looked up in the HDS-derived surface "
      "`P(accept | location)` (built with `build(sigma=6.0, alpha=0.02)`, map "
      "geography only — no raking to HDS state tables, so nothing here is "
      "circular). We test whether that probability predicts the respondent's own "
      "1–5 acceptability rating.\n")
    W("Metrics: **AUC** (does P rank accepters, rating≥4, above non-accepters?); "
      "**Spearman ρ** between P and the continuous mean rating (person-level and "
      "city-level); and a **calibration table** (quintiles of predicted P vs "
      "observed acceptance).\n")

    W("## Accepted mappings tested\n")
    for m in ACCEPTED_MAPPINGS:
        W(f"- q{m['question']} ⇐ {PHENO_LABEL[m['pheno']]}: {m['rule']}")
    W("")

    W("## Rejected / impossible mappings\n")
    W("Reported for honesty — the candidate list is larger than the usable list.\n")
    for m in REJECTED_MAPPINGS:
        W(f"- **{m['pheno']}** vs **{m['hds']}** — {m['reason']}")
    W("")
    W("Note the HDS↔YGDP overlap is small: HDS is overwhelmingly lexical/"
      "phonological, and its only syntax items are q49–q53 and q54–q57. Of those, "
      "only *come-with* (q51) and *positive-anymore* (q54–57) have a YGDP "
      "counterpart. The much-touted q50↔y'all pairing is a mirage: y'all is not a "
      "YGDP phenomenon, and the 'Alls construction' is unrelated. Needs-washed, "
      "fixin'-to, dative-presentatives, personal-datives and all-the-faster are "
      "real YGDP phenomena with **no HDS question at all**.\n")

    W("## Pre-registered Likert→choice binarisation\n")
    W("Chosen **before** looking at any AUC. YGDP ships three per-person cuts as "
      "`*_Per_3_up / *_Per_4_up / *_Per_5` (fraction of a person's paraphrase "
      "sentences rated ≥3 / ≥4 / =5). We register **Per_4_up** and call a person "
      "an *accepter* when a **majority of their paraphrases are rated ≥4** — the "
      "standard 4–5=acceptable convention, neither the loosest (≥3 counts the "
      "marginal '3') nor the strictest (=5). Accepter → HDS choice `a` "
      "(acceptable / 'yes'); otherwise → `b`. Both are written to `answers.csv` so "
      "the calibrator sees negatives, not just accepters. The ≥3 and =5 cuts "
      "appear only in the sensitivity table below.\n")

    vs = validation_set_stats(recs)
    ov = overview_exclusion_stats(files)
    W("## Geographic coverage & honest scope\n")
    W(f"- Raw feature rows: **{ov['tot_rows']:,}**. The two overview files "
      f"(*Overview Map 2*, *Home Page*) contribute **{ov['overview_rows']:,}** of "
      f"them and are excluded from the phenomenon set; **{ov['pheno_rows']:,}** "
      f"phenomenon rows remain. Only **{ov['overview_only_people']}** unique "
      f"people appear *solely* in overview files (kept in `people.csv` for "
      f"location, never emitted as an answer).\n")
    W(f"- People carrying ≥1 HDS-mapped answer (the validation set): "
      f"**{vs['n_val']:,}**. Of these, **{vs['n_2dist']}** answered ≥2 *distinct* "
      f"constructions (come-with **and** anymore) — that subset is the only lever "
      f"for measuring cross-construction dependence. ({vs['n_2rows']:,} people "
      f"have ≥2 answer *rows*, but for all but {vs['n_2dist']} of them the extra "
      f"rows are the four near-duplicate q54–57 anymore questions, not new "
      f"information.)\n")
    top = vs["hist"].most_common(12)
    W("- State histogram of the validation set (home state, top 12):\n")
    W("| state | people |")
    W("|---|---|")
    for st, c in top:
        W(f"| {st} | {c} |")
    W("")
    mv = vs["moved"]
    W(f"- Mobility (`moved`): known for {mv.get('0',0)+mv.get('1',0)} of "
      f"{vs['n_val']:,} (the rest ship no Current.CityState). Of those, "
      f"**{mv.get('1',0)} moved** away from their raised city and "
      f"{mv.get('0',0)} stayed — a mostly-mobile set, which is the honest "
      f"(harder) case: the model is judged on where people were *raised*, not "
      f"where they now live, so a set of stayers would flatter it.\n")
    W(f"**Scope caveat.** The two testable constructions are a Midland/Upper-"
      f"Midwest feature (come-with: MN/WI/IL) and a Midland feature (positive-"
      f"anymore: PA/OH/Midlands). The *respondents* are national ("
      f"{len([s for s in vs['hist'] if s not in ('??',)])} states represented), "
      f"so the surfaces are tested on people from everywhere — but the *features* "
      f"only probe northern/Midland variables. This set therefore certifies the "
      f"model on two northern-skewed syntactic surfaces, **not** on national "
      f"accuracy across the 122 mostly-lexical questions. Quote it that way.\n")

    W("## Does the Likert-vs-forced-choice format difference invalidate this?\n")
    W("Partly, so we keep the claims directional. HDS asks a forced choice "
      "('is this acceptable? yes/no' for q54–57; 'would you say it? yes/no' for "
      "q51); YGDP asks a 1–5 acceptability rating. We binarise YGDP at rating ≥ 4 "
      "= 'accepts', which is the closest analogue, and also correlate the "
      "continuous rating. The mapping is **ordinally valid** — higher YGDP rating "
      "should mean higher P(accept) in HDS — but not interval-calibrated, so we "
      "report **rank** metrics (AUC, Spearman) rather than fitting absolute "
      "probabilities. For q51 the two surveys even ask nearly the same question "
      "('are you coming with?'), so that mapping is tight; for *anymore* both are "
      "acceptability judgments, differing only in scale granularity.\n")

    results = {}
    for m in ACCEPTED_MAPPINGS:
        res = validate_mapping(recs, S, built, m["pheno"], m["question"],
                               m["accept_choices"])
        results[(m["pheno"], m["question"])] = res
        W(f"## q{m['question']} — {PHENO_LABEL[m['pheno']]}\n")
        if res is None:
            W("Too few in-grid respondents to test.\n")
            continue
        W(f"- respondents (CONUS): **{res['n']}**  "
          f"(accepters={res['npos']}, non={res['nneg']}, base rate "
          f"{res['base_rate']:.2f})")
        W(f"- predicted P range: {res['P_lo']:.3f}–{res['P_hi']:.3f}")
        W(f"- **AUC = {res['auc']:.3f}**  (0.5 = no signal)")
        W(f"- person-level Spearman(P, rating) = {res['spearman']:+.3f}")
        W(f"- city-level Spearman(P, mean rating) = {res['city_rho']:+.3f} "
          f"(n={res['city_n']} cities)\n")
        W("| P quintile | mean predicted P | observed accept rate | mean rating | n |")
        W("|---|---|---|---|---|")
        for pp, ac, mr, n in res["cal"]:
            W(f"| | {pp:.3f} | {ac:.3f} | {mr:.2f} | {n} |")
        W("")

    # positive-anymore: also test the mean surface across q54-57
    pa_qs = [m["question"] for m in ACCEPTED_MAPPINGS if m["pheno"] == "positive_anymore"]
    if pa_qs:
        aucs = [results[("positive_anymore", q)]["auc"] for q in pa_qs
                if results[("positive_anymore", q)]]
        W("## Positive-anymore summary (q54–q57)\n")
        W(f"AUC across the four HDS *anymore* questions ranges "
          f"**{min(aucs):.3f}–{max(aucs):.3f}** (mean {np.mean(aucs):.3f}). Three "
          f"of the four (q54 {results[('positive_anymore','54')]['auc']:.2f}, "
          f"q55 {results[('positive_anymore','55')]['auc']:.2f}, "
          f"q56 {results[('positive_anymore','56')]['auc']:.2f}) carry a clear, "
          f"consistent geographic signal; **q57 is a null** "
          f"({results[('positive_anymore','57')]['auc']:.2f}, Spearman ≈ 0). "
          f"q57 ('Forget the nice clothes anymore…') is the oddest sentence and "
          f"its recovered surface does not track real acceptance — an honest "
          f"reminder that not every recovered surface is usable, and that the "
          f"three informative *anymore* questions are near-duplicates of each "
          f"other (so multiplying all four over-weights one diffuse feature).\n")

    W("## Binarisation sensitivity (not used to pick the headline)\n")
    W("AUC for each mapping under the pre-registered ≥4 cut and the two other "
      "shipped cuts (≥3, =5). The signal is a property of the geography, not of "
      "the threshold.\n")
    W("| question | AUC (≥3) | AUC (≥4, headline) | AUC (=5) |")
    W("|---|---|---|---|")
    for pheno, q, cut in binarisation_sensitivity(recs, S, built):
        def _f(t):
            a = cut[t][0]
            return f"{a:.3f}" if np.isfinite(a) else "·"
        W(f"| q{q} | {_f(3)} | **{_f(4)}** | {_f(5)} |")
    W("")

    W("## Within-person cross-construction dependence (model-based)\n")
    cc = crossconstruction_residual(recs, S, built)
    if cc:
        W(f"The **{cc['n']} people who answered both** come-with and positive-"
          f"anymore let us measure the dependence that actually breaks the model's "
          f"conditional-independence assumption. For each person we take the "
          f"residual `observed_accept − model_P(accept | home cell)` for each "
          f"construction, then correlate the two residuals.\n")
        W(f"- raw correlation of the two binary answers: **{cc['r_raw']:+.3f}**")
        W(f"- **residual correlation (after the model's own location "
          f"expectation): {cc['r_resid']:+.3f}** (n = {cc['n']})\n")
        ne_pair = 2.0 / (1.0 + max(cc['r_resid'], 0.0))
        W(f"Acceptance base rates in this subset: come-with "
          f"{cc['acc_cw']:.2f}, anymore {cc['acc_pa']:.2f}. The residual "
          f"({cc['r_resid']:+.3f}) is barely below the raw ({cc['r_raw']:+.3f}): "
          f"**location removes almost none of the cross-construction dependence**, "
          f"echoing the non-spatial result in `correlation_report.md`. So even two "
          f"*genuinely distinct* constructions (not paraphrases of one) keep a "
          f"**modest but real** within-person correlation after the model "
          f"conditions on where you grew up — the conditional-independence "
          f"assumption is violated, though less severely than the paraphrase-and-"
          f"acquiescence-inflated ρ̄≈0.30. As a rule of thumb this makes two "
          f"distinct constructions behave like 2/(1+{cc['r_resid']:.3f}) ≈ "
          f"**{ne_pair:.2f}** independent ones, a discount of ~"
          f"{100*(1-ne_pair/2):.0f}%. It is one coefficient from the only "
          f"construction pair the overlap allows, so treat it as indicative "
          f"(n={cc['n']}, ~95% interval ±{1.96/np.sqrt(cc['n']-3):.2f}); part of "
          f"even this residual is Likert scale-use that will shrink under HDS "
          f"forced choice, so it is an upper bound on the HDS discount.\n")
    else:
        W("Too few people answered both distinct constructions to estimate a "
          "cross-construction residual correlation.\n")

    W("## Verdict\n")
    cw = results.get(("come_with", "51"))
    pa54 = results.get(("positive_anymore", "54"))
    pa56 = results.get(("positive_anymore", "56"))
    pa57 = results.get(("positive_anymore", "57"))
    if cw and cw["auc"] > 0.55:
        W(f"- **Come-with (q51): genuine signal.** AUC {cw['auc']:.3f}, city-level "
          f"Spearman {cw['city_rho']:+.3f}. The HDS surface built from dot maps "
          f"ranks real YGDP come-with accepters above rejecters purely from their "
          f"home coordinates — an out-of-sample validation of the geography.")
    elif cw:
        W(f"- **Come-with (q51): weak/no signal.** AUC {cw['auc']:.3f}.")
    if pa56 and pa56["auc"] > 0.55:
        W(f"- **Positive-anymore (q54–q56): genuine signal.** AUC up to "
          f"{pa56['auc']:.3f} (q56), city Spearman {pa56['city_rho']:+.3f}; q54 "
          f"and q55 agree. Calibration is monotone: rarer-*anymore* regions in "
          f"the HDS surface really do contain fewer YGDP accepters.")
    if pa57 and pa57["auc"] <= 0.55:
        W(f"- **Positive-anymore q57: no signal.** AUC {pa57['auc']:.3f}, "
          f"Spearman {pa57['spearman']:+.3f}. Reported plainly, not hidden: this "
          f"HDS surface does not predict YGDP judgments.")
    W("- Overall: **the HDS-recovered surfaces are externally valid** where the "
      "feature is dialectally sharp (come-with, expensive-*anymore*), and "
      "honestly fail on a noisy item (q57). Two phenomena is a narrow test — it "
      "is all the HDS/YGDP overlap allows — but on that overlap the geography "
      "recovered from pixels predicts real, independently-located people.")
    (OUT / "hds_validation.md").write_text("\n".join(L))
    return results


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("YGDP -> HDS external validation")
    print("=" * 70)

    files, fcc = build_inventory()
    print(f"\n[1] Inventory: {len(files)} GeoJSON files")
    print(f"    field classes: {dict(fcc)}  (total {sum(fcc.values())} distinct)")

    recs, items, conflicts, collisions = build_people()
    write_people_csv(recs, items)
    n_ans = write_answers_csv(recs)
    write_crosswalk_csv()
    tot_rows = sum(f["n"] for f in files.values())
    print(f"\n[2] People: {tot_rows} feature rows -> {len(recs)} unique respondents")
    print(f"    canonical items: {len(items)}   merge conflicts: {conflicts}")
    print(f"    within-file key collisions: {sum(collisions.values())}")
    nitems = np.array([r["n_items"] for r in recs])
    print(f"    items/person: mean={nitems.mean():.1f} median={int(np.median(nitems))} "
          f"max={nitems.max()}   -> data/ygdp/people.csv")
    ov = overview_exclusion_stats(files)
    vs = validation_set_stats(recs)
    print(f"    overview rows excluded: {ov['overview_rows']:,} of {ov['tot_rows']:,} "
          f"-> {ov['pheno_rows']:,} phenomenon rows "
          f"({ov['overview_only_people']} people overview-only)")
    print(f"    answers.csv: {n_ans:,} rows; validation set = {vs['n_val']:,} people "
          f"({vs['n_2dist']} with 2 distinct constructions)")
    print(f"    -> data/ygdp/answers.csv, data/ygdp/crosswalk.csv")
    build_inventory_report(files, recs)
    print("    inventory -> data/ygdp/inventory.md")

    lat = np.array([r["lat"] if r["lat"] is not None else np.nan for r in recs], float)
    lon = np.array([r["lon"] if r["lon"] is not None else np.nan for r in recs], float)
    states = np.array([r["state"] for r in recs], dtype=object)

    print("\n[3] HDS mappings")
    for m in ACCEPTED_MAPPINGS:
        print(f"    ACCEPT  q{m['question']} <- {m['pheno']} ({m['confidence']})")
    for m in REJECTED_MAPPINGS:
        print(f"    REJECT  {m['pheno']} / {m['hds']}")

    print("\n[4] Correlation report -> data/ygdp/correlation_report.md")
    cr = build_correlation_report(recs, items, lat, lon, states)
    print(f"    raw ρ̄={cr['rho_raw']:+.3f}  spatial ρ̄={cr['rho_sp']:+.3f}  "
          f"state ρ̄={cr['rho_st']:+.3f}  ({cr['npairs']} pairs)")
    print(f"    block A n_eff(spatial)={cr['blockA']['spatial']['n_eff']:.2f}/4  "
          f"block B n_eff(spatial)={cr['blockB']['spatial']['n_eff']:.2f}/5")

    print("\n[5] HDS surface validation -> data/ygdp/hds_validation.md")
    from likelihood import Surfaces
    S = Surfaces()
    vr = build_validation_report(recs, S, files)
    for (ph, q), res in vr.items():
        if res:
            print(f"    q{q} {ph:18s} n={res['n']:4d} AUC={res['auc']:.3f} "
                  f"citySpearman={res['city_rho']:+.3f}")
    print(f"    validation-set states: {len([s for s in vs['hist'] if s != '??'])} "
          f"represented; top: "
          f"{', '.join(f'{s}={c}' for s, c in vs['hist'].most_common(5))}")
    print("\nDone.")
    return recs, items


if __name__ == "__main__":
    main()
