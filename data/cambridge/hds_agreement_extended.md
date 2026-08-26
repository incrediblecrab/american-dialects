# Harvard vs Cambridge: coverage of the deployed questions, and the medium tier

Companion to `hds_agreement.md`, which owns the headline high-confidence numbers and is left untouched. Method is identical: same grid, same renormalisation over mapped answers, same chance baseline computed from the two modal maps' own marginals.

## Coverage of the twenty deployed questions

In the order the geolocator asks them. The recommendation is to stop at 12, so the first 12 rows are the ones that decide what can be claimed.

| # | HDS | tier | answers | cells | mean r | MAE (pp) | modal | chance | margin | modal (primary) |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | q105 | high | 4 | 43175 | 0.45 | 16.5 | 68% | 31% | +37% | 71% |
| **2** | q74 | high | 6 | 44960 | 0.28 | 13.6 | 54% | 44% | +10% | 69% |
| **3** | q95 | medium | 5 | 39382 | 0.26 | 13.0 | 74% | 67% | +7% | 92% |
| **4** | q99 | medium | 6 | 43877 | 0.31 | 12.8 | 57% | 31% | +26% | 66% |
| **5** | q106 | medium | 5 | 43862 | 0.37 | 13.3 | 64% | 43% | +21% | 72% |
| **6** | q110 | high | 4 | 27175 | 0.15 | 27.8 | 39% | 30% | +9% | 59% |
| **7** | q79 | medium | 5 | 42014 | 0.27 | 13.5 | 67% | 52% | +15% | 70% |
| **8** | q64 | high | 5 | 43271 | 0.27 | 9.0 | 86% | 82% | +4% | 88% |
| **9** | q118 | medium | 4 | 42822 | 0.31 | 18.0 | 57% | 40% | +17% | 65% |
| **10** | q103 | high | 3 | 43097 | 0.41 | 16.0 | 72% | 63% | +9% | 72% |
| **11** | q73 | high | 5 | 42031 | 0.26 | 11.7 | 79% | 66% | +14% | 81% |
| **12** | q58 | high | 4 | 43073 | 0.36 | 14.6 | 67% | 44% | +23% | 67% |
| 13 | q76 | high | 4 | 43291 | 0.41 | 14.0 | 75% | 46% | +29% | 75% |
| 14 | q66 | high | 3 | 41902 | 0.45 | 19.2 | 65% | 36% | +29% | 65% |
| 15 | q50 | high | 4 | 42891 | 0.43 | 17.8 | 57% | 32% | +26% | 63% |
| 16 | q65 | high | 3 | 42365 | 0.39 | 21.1 | 54% | 36% | +18% | 54% |
| 17 | q84 | high | 4 | 42693 | 0.23 | 18.2 | 55% | 44% | +11% | 61% |
| 18 | q59 | medium | 3 | 34377 | 0.19 | 27.7 | 48% | 40% | +8% | — |
| 19 | q83 | medium | 4 | 39967 | 0.30 | 16.6 | 69% | 58% | +11% | 85% |
| 20 | q60 | high | 5 | 33123 | 0.25 | 17.5 | 55% | 41% | +14% | — |

Of the twenty deployed questions, **13 are in the high-confidence tier**. Mean margin over chance across the first 12: **+16.0%** (min +4%, max +37%).

## The medium tier, by question class

Never pooled with the high tier. Harvard's question numbering is contiguous by type, so the split is q1-48 pronunciation, q49-57 syntax, q58-122 lexical.

| tier | class | questions | cell comparisons | pooled r | modal | chance | margin |
|---|---|---|---|---|---|---|---|
| high | all | 25 | 3,930,194 | 0.645 | 68% | 54% | +13% |
| medium | lexical | 51 | 7,740,483 | 0.632 | 69% | 61% | +9% |
| medium | phonetic | 43 | 6,044,443 | 0.560 | 61% | 53% | +9% |
| medium | syntactic | 9 | 1,138,359 | 0.647 | 71% | 63% | +8% |
| medium | all | 103 | 14,923,285 | 0.608 | 66% | 57% | +9% |

The margin over chance is **+9% on medium-tier lexical items** and **+9% on phonetic items**. The margin holds across both classes, so the corroboration is not an artefact of easy lexical items.
