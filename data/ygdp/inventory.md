# YGDP inventory

Parsed from the 28 raw GeoJSON files in `data/raw/ygdp/`. Each feature is one respondent geocoded to their home-city centroid, with demographics and 1–5 acceptability ratings.

## Field schema (learned from the files)

Every non-demographic numeric property is one of three kinds:

- **raw item** — a single sentence's 1–5 Likert rating. Named `F####` (global sentence id), `PREFIX_####` (same id, phenomenon-prefixed, e.g. `PA_1226` = `F1226`), a bare `####`, or a phenomenon-local relative-clause code (`TO_/TS_/PT_/ST_` + `1000/2000/…`, which are *not* global ids).

- **aggregate** — derived per person over that phenomenon's sentences: `_MEAN _MAX _MIN _MEDIAN/_Med _Mean_Round/_MEAN_R _Per_3_up _Per_4_up _Per_5`. Excluded from analysis (deterministic functions of the raw items).

- **metadata** — `FIELD1`, `Survey` (wave number), `""`.

Totals: **288 distinct field names = 138 raw + 147 aggregate + 3 metadata**. The 138 raw names collapse to **113 sentence ids**; after keeping the relative-clause local codes distinct, **128 canonical items**.

## Duplication across files (the key finding)

- **22,102 feature rows → 6629 unique respondents.** A respondent is one (home-city centroid + demographics + survey wave). Cross-file consistency was verified: e.g. sentence F1129 is rated identically for all 303 people shared by `allsconstruction11` and `baregotdosupport`.

- Appearances per respondent: 1 file(s): 1418, 2 file(s): 1468, 3 file(s): 835, 4 file(s): 496, 5 file(s): 1840, 7 file(s): 572.

- The 7 `survey12*` files are the **same 573 people** re-rendered (Verbal Rather min/max/median/all, Positive Anymore, Likes/Loves) — one wave, not seven. `s6b_needs_washed` and `s9-_dative_presentatives` share the same 1521 people. But `convertcsv_1` (Positive Anymore) and `survey12` (Positive Anymore) are **disjoint** respondent pools of the same phenomenon.

- Caveat: 55 within-file key collisions across 11 files — distinct people from one city with identical demographics who cannot be told apart, so 6629 is a slight lower bound.

## Per-file summary

`agg?` marks files that are overviews/aggregates rather than a single phenomenon.

| file | title | respondents | raw items | agg fields | rating 1..5 dist | note |
|---|---|---|---|---|---|---|
| afterperf_1.geojson | After Perfect | 349 | 5 | 8 | 988/501/171/54/31 |  |
| afterperfect.geojson | After-perfect | 349 | 5 | 8 | 988/501/171/54/31 |  |
| alls_construction_1.geojson | Alls Construction | 349 | 5 | 8 | 318/386/412/316/313 |  |
| allsconstruction11.geojson | Alls Construction | 652 | 6 | 16 | 322/386/422/378/540 |  |
| allthefaster.geojson | All the faster | 807 | 1 | 8 | 436/194/93/56/28 |  |
| baregotdosupport.geojson | Bare Got Do Support | 303 | 1 | 8 | 4/0/10/62/227 |  |
| contactrelatives.geojson | Contact Relatives | 807 | 1 | 8 | 204/235/175/123/70 |  |
| convertcsv.geojson | Fixin' to | 360 | 5 | 8 | 140/282/400/406/572 |  |
| convertcsv_0.geojson | Come with | 349 | 5 | 8 | 102/227/286/462/668 |  |
| convertcsv_1.geojson | Positive Anymore | 899 | 5 | 8 | 1431/866/605/622/971 |  |
| donemyhomework8.geojson | Done your homework | 539 | 3 | 8 | 910/403/123/47/134 |  |
| for_to_1.geojson | For to infinitives | 349 | 5 | 8 | 661/516/353/107/108 |  |
| map.geojson | Home Page | 2800 | 16 | 0 | 5032/2846/2107/1623/1828 | OVERVIEW/aggregate |
| relthats_subob.geojson | Relative that's | 135 | 10 | 0 | 508/278/233/163/166 |  |
| relthatssgpl_formap_0.geojson | Relative that's SGPL | 151 | 10 | 0 | 556/408/221/175/134 |  |
| s5b-_split_subjects.geojson | Split Subjects | 510 | 8 | 0 | 1919/1042/567/319/233 |  |
| s5b-_verbal_rather.geojson | Verbal rather | 510 | 2 | 0 | 340/219/157/123/181 |  |
| s6b-_personal_datives.geojson | Personal Datives | 1521 | 6 | 0 | 3004/2072/1707/1288/1055 |  |
| s6b_needs_washed.geojson | Needs washed | 1521 | 3 | 0 | 1602/891/591/450/1029 |  |
| s9-_dative_presentatives.geojson | Dative presentatives | 2031 | 12 | 0 | 3591/2007/1252/900/1403 |  |
| sentence_overview_more_data_march_2019.geojson | Overview Map 2 | 2800 | 16 | 0 | 5032/2846/2107/1623/1828 | OVERVIEW/aggregate |
| survey12_0.geojson | Positive Anymore | 573 | 42 | 58 | 8324/4463/3290/3041/4948 |  |
| survey12_fixed.geojson | Verbal Rather (All) | 573 | 43 | 60 | 8465/4592/3388/3137/5057 |  |
| survey12_fixed_0.geojson | Verbal Rather (Max) | 573 | 43 | 60 | 8465/4592/3388/3137/5057 |  |
| survey12_fixed_3.geojson | Verbal Rather (Min / Max / | 573 | 43 | 60 | 8465/4592/3388/3137/5057 |  |
| survey12_fixed_4.geojson | Likes/loves carried | 573 | 43 | 60 | 8465/4592/3388/3137/5057 |  |
| survey12_refixed.geojson | Verbal Rather (Median) | 573 | 43 | 60 | 8465/4592/3388/3137/5057 |  |
| survey12_refixed_0.geojson | Verbal Rather (Min) | 573 | 43 | 60 | 8465/4592/3388/3137/5057 |  |

Rating distribution across all raw items is bottom-heavy (most judgments are 1 = fully unacceptable), as expected for nonstandard syntax sampled nationally.
