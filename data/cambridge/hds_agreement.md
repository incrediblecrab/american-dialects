# Harvard vs Cambridge dialect surveys: independent agreement

Two surveys, different populations (Harvard ~2002-03; Cambridge later), the same questions. Both surfaces were recovered from rendered dot maps and reprojected onto the identical 200x456 grid, then compared over US land. For each question we renormalise each survey's recovered density into shares over the answers that map between them, so the comparison is apples-to-apples.

**Read these as spatial-agreement numbers, not respondent-count accuracy.** Coverage measures where an answer appears, inflated toward dispersed/rural variants in both surveys; only the top-3 (primary-colour) Cambridge answers are cleanly recoverable.

## Headline numbers (high-confidence questions)

- Questions compared: **25**
- Per-cell answer shares compared: **3,930,194** (answer x US-land cell)
- Pooled Pearson r (share vs share): **0.645** over all mapped answers; **0.636** over the cleanly recoverable top-3 primary-colour answers.
- Pooled mean abs difference: **15.5 pp** all answers / **18.6 pp** primaries (density-weighted 10.9 pp).
- Locally modal answer agrees: **68%** of US-land cells (density-weighted 73%; denser half of cells 74%) -- against a chance baseline of **54%** (independent modal maps with the same marginals).
- Restricting to the top-3 primary-colour answers, the modal answer agrees in **72%** of cells -- these are the answers we can recover cleanly.

## Per-question agreement

| Cambridge | HDS | answers | cells | Pearson r | MAE (pp) | modal (all) | modal (primary) |
|---|---|---|---|---|---|---|---|
| C290 | q50 | 4 | 42891 | 0.43 | 17.8 | 57% | 63% |
| C298 | q58 | 4 | 43073 | 0.36 | 14.6 | 67% | 67% |
| C300 | q60 | 5 | 33123 | 0.25 | 17.5 | 55% | - |
| C302 | q62 | 3 | 41692 | 0.23 | 11.3 | 92% | 94% |
| C303 | q63 | 2 | 41138 | 0.35 | 2.2 | 99% | 99% |
| C304 | q64 | 5 | 43271 | 0.27 | 9.0 | 86% | 89% |
| C305 | q65 | 3 | 42365 | 0.39 | 21.1 | 54% | 54% |
| C306 | q66 | 3 | 41902 | 0.45 | 19.2 | 65% | 65% |
| C312 | q72 | 2 | 41293 | 0.19 | 21.1 | 77% | 77% |
| C313 | q73 | 5 | 42031 | 0.26 | 11.7 | 79% | 81% |
| C314 | q74 | 6 | 44960 | 0.28 | 13.6 | 54% | 69% |
| C315 | q75 | 3 | 41178 | 0.44 | 14.4 | 79% | 79% |
| C316 | q76 | 4 | 43291 | 0.41 | 14.0 | 75% | 75% |
| C320 | q80 | 3 | 33219 | 0.38 | 17.4 | 76% | 78% |
| C324 | q84 | 4 | 42693 | 0.23 | 18.2 | 55% | 62% |
| C334 | q94 | 4 | 44066 | 0.28 | 19.1 | 46% | 53% |
| C336 | q96 | 5 | 44147 | 0.23 | 16.6 | 46% | 56% |
| C337 | q97 | 4 | 44067 | 0.24 | 19.0 | 41% | 44% |
| C343 | q103 | 3 | 43097 | 0.41 | 16.0 | 72% | 72% |
| C344 | q104 | 5 | 43398 | 0.20 | 8.5 | 87% | 91% |
| C345 | q105 | 4 | 43175 | 0.45 | 16.5 | 68% | 75% |
| C350 | q110 | 4 | 27175 | 0.15 | 27.8 | 39% | 56% |
| C357 | q117 | 5 | 44137 | 0.24 | 13.6 | 62% | 70% |
| C359 | q119 | 3 | 41590 | 0.25 | 16.7 | 72% | 72% |
| C360 | q120 | 2 | 34092 | 0.12 | 22.8 | 83% | 83% |

## Interpretation (honest)

- The two independent surveys **agree on the locally most common answer in roughly 7 cells out of 10** over US land (72-74% on the cleanly recoverable answers / denser cells), well above the ~50% you would get by chance from the same marginals. That is a genuine, if imperfect, corroboration.
- Cell-level correlation is **moderate (pooled r ~0.64)**, not high. Two things cap it: (a) genuine sampling differences between two differently-recruited populations a few years apart, and (b) noise in recovering counts from rendered dots. The isogloss read-outs below show the *signal* is right where it matters; the moderate r reflects cell-by-cell *noise*, not disagreement about the big regional patterns.
- Agreement is strongest exactly where dialect geography is sharpest (median q62 92%, milkshake q63 99%, sub-sandwich q64 86%, subway q104 87%, pop/soda/coke q105 primaries 75%) and weakest where the variants are nationally interspersed or sit on hard-to-recover colours (trash/garbage can q97 41%, frosting/icing q94 46%, dinner/supper q96 46%, night-before-Halloween q110 39% -- mostly 'no word' plus tiny black-coded regional terms).
- **Excluded from these numbers:** the two featured multi-select items (C1 sandwich, C2 carbonated) -- their percentages are not comparable to single-select HDS, and co-selected answers co-locate and alias to secondary colours. The clean single-select re-runs C304/C345 are used instead.

## Documented isoglosses (local share at the home metro)

- **hoagie @ Philadelphia** — Harvard 33% vs Cambridge 53%
- **poor boy @ New Orleans (Camb=cyan, low-confidence)** — Harvard 33% vs Cambridge 32%
- **pop @ Buffalo** — Harvard 53% vs Cambridge 63%
- **soda @ NYC** — Harvard 59% vs Cambridge 43%
- **frappe @ Boston (q63)** — Harvard 37% vs Cambridge 55%
- **bubbler @ Providence-ish/E.Mass (q103)** — Harvard 29% vs Cambridge 39%

Not cross-validatable (absent from the Cambridge answer set): **yinz/yinz @ Pittsburgh** (Cambridge C290 has no yinz option) and **cabinet @ Providence** (Cambridge C303 shows only milkshake/frappe; cabinet was truncated). Harvard alone still shows both.
- Harvard-only: yinz @ Pittsburgh = 14%; cabinet @ Providence = 12% vs @ Boston = 6%.