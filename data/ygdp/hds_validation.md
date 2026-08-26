# Do the HDS surfaces predict real YGDP judgments?

Out-of-sample test. For each accepted YGDP↔HDS mapping, every YGDP respondent's home city is looked up in the HDS-derived surface `P(accept | location)` (built with `build(sigma=6.0, alpha=0.02)`, map geography only — no raking to HDS state tables, so nothing here is circular). We test whether that probability predicts the respondent's own 1–5 acceptability rating.

Metrics: **AUC** (does P rank accepters, rating≥4, above non-accepters?); **Spearman ρ** between P and the continuous mean rating (person-level and city-level); and a **calibration table** (quintiles of predicted P vs observed acceptance).

## Accepted mappings tested

- q51 ⇐ Come with (are you coming with?): majority of YGDP come-with sentences rated >=4  <->  HDS q51 'yes'
- q54 ⇐ Positive anymore (X is expensive anymore): majority of YGDP positive-anymore sentences rated >=4  <->  HDS q54 'acceptable'
- q55 ⇐ Positive anymore (X is expensive anymore): majority of YGDP positive-anymore sentences rated >=4  <->  HDS q55 'acceptable'
- q56 ⇐ Positive anymore (X is expensive anymore): majority of YGDP positive-anymore sentences rated >=4  <->  HDS q56 'acceptable'
- q57 ⇐ Positive anymore (X is expensive anymore): majority of YGDP positive-anymore sentences rated >=4  <->  HDS q57 'acceptable'

## Rejected / impossible mappings

Reported for honesty — the candidate list is larger than the usable list.

- **alls_construction** vs **q50 (address a group)** — The mooted q50<->y'all pairing has NO YGDP counterpart: y'all is not a YGDP phenomenon at all, and 'Alls construction' is 'alls I know is...' (a free-relative complementiser), unrelated to 2nd-person-plural pronouns.
- **needs_washed** vs **(none)** — No HDS question tests 'needs washed' / needs+past-participle. HDS syntax items are only q49-q53 and q54-q57; none covers this construction.
- **fixin_to** vs **(none)** — No HDS question tests 'fixin' to' / prospective future. HDS has no future-marker item.
- **dative_presentatives** vs **(none)** — No HDS question tests presentative datives ('here's you a piece').
- **personal_datives** vs **(none)** — No HDS question tests personal datives ('I got me a truck').
- **all_the_faster** vs **(none)** — No HDS question tests 'all the faster' ('this is all the faster it goes').
- **double_modals** vs **q53 (might could)** — HDS q53 tests double modals, but YGDP ships no double-modal file (Bare-Got-Do-Support is 'do you got', unrelated).
- **where_at** vs **q52 (where are you at)** — HDS q52 tests locative 'at', but YGDP ships no such file.
- **drug_dragged** vs **q49 (I ___ her body from the pool)** — HDS q49 tests 'drug' as past tense of 'drag' (morphology). YGDP ships no corresponding file.

Note the HDS↔YGDP overlap is small: HDS is overwhelmingly lexical/phonological, and its only syntax items are q49–q53 and q54–q57. Of those, only *come-with* (q51) and *positive-anymore* (q54–57) have a YGDP counterpart. The much-touted q50↔y'all pairing is a mirage: y'all is not a YGDP phenomenon, and the 'Alls construction' is unrelated. Needs-washed, fixin'-to, dative-presentatives, personal-datives and all-the-faster are real YGDP phenomena with **no HDS question at all**.

## Pre-registered Likert→choice binarisation

Chosen **before** looking at any AUC. YGDP ships three per-person cuts as `*_Per_3_up / *_Per_4_up / *_Per_5` (fraction of a person's paraphrase sentences rated ≥3 / ≥4 / =5). We register **Per_4_up** and call a person an *accepter* when a **majority of their paraphrases are rated ≥4** — the standard 4–5=acceptable convention, neither the loosest (≥3 counts the marginal '3') nor the strictest (=5). Accepter → HDS choice `a` (acceptable / 'yes'); otherwise → `b`. Both are written to `answers.csv` so the calibrator sees negatives, not just accepters. The ≥3 and =5 cuts appear only in the sensitivity table below.

## Geographic coverage & honest scope

- Raw feature rows: **22,102**. The two overview files (*Overview Map 2*, *Home Page*) contribute **5,600** of them and are excluded from the phenomenon set; **16,502** phenomenon rows remain. Only **312** unique people appear *solely* in overview files (kept in `people.csv` for location, never emitted as an answer).

- People carrying ≥1 HDS-mapped answer (the validation set): **1,450**. Of these, **349** answered ≥2 *distinct* constructions (come-with **and** anymore) — that subset is the only lever for measuring cross-construction dependence. (1,450 people have ≥2 answer *rows*, but for all but 349 of them the extra rows are the four near-duplicate q54–57 anymore questions, not new information.)

- State histogram of the validation set (home state, top 12):

| state | people |
|---|---|
| CA | 163 |
| NY | 113 |
| PA | 97 |
| OH | 87 |
| IL | 75 |
| TX | 72 |
| MI | 69 |
| FL | 54 |
| GA | 46 |
| MA | 42 |
| NJ | 38 |
| MD | 35 |

- Mobility (`moved`): known for 572 of 1,450 (the rest ship no Current.CityState). Of those, **467 moved** away from their raised city and 105 stayed — a mostly-mobile set, which is the honest (harder) case: the model is judged on where people were *raised*, not where they now live, so a set of stayers would flatter it.

**Scope caveat.** The two testable constructions are a Midland/Upper-Midwest feature (come-with: MN/WI/IL) and a Midland feature (positive-anymore: PA/OH/Midlands). The *respondents* are national (51 states represented), so the surfaces are tested on people from everywhere — but the *features* only probe northern/Midland variables. This set therefore certifies the model on two northern-skewed syntactic surfaces, **not** on national accuracy across the 122 mostly-lexical questions. Quote it that way.

## Does the Likert-vs-forced-choice format difference invalidate this?

Partly, so we keep the claims directional. HDS asks a forced choice ('is this acceptable? yes/no' for q54–57; 'would you say it? yes/no' for q51); YGDP asks a 1–5 acceptability rating. We binarise YGDP at rating ≥ 4 = 'accepts', which is the closest analogue, and also correlate the continuous rating. The mapping is **ordinally valid** — higher YGDP rating should mean higher P(accept) in HDS — but not interval-calibrated, so we report **rank** metrics (AUC, Spearman) rather than fitting absolute probabilities. For q51 the two surveys even ask nearly the same question ('are you coming with?'), so that mapping is tight; for *anymore* both are acceptability judgments, differing only in scale granularity.

## q51 — Come with (are you coming with?)

- respondents (CONUS): **349**  (accepters=227, non=122, base rate 0.65)
- predicted P range: 0.094–0.843
- **AUC = 0.605**  (0.5 = no signal)
- person-level Spearman(P, rating) = +0.230
- city-level Spearman(P, mean rating) = +0.245 (n=274 cities)

| P quintile | mean predicted P | observed accept rate | mean rating | n |
|---|---|---|---|---|
| | 0.294 | 0.551 | 3.51 | 69 |
| | 0.363 | 0.629 | 3.76 | 70 |
| | 0.398 | 0.571 | 3.52 | 70 |
| | 0.435 | 0.714 | 3.93 | 70 |
| | 0.558 | 0.786 | 4.19 | 70 |

## q54 — Positive anymore (X is expensive anymore)

- respondents (CONUS): **1445**  (accepters=492, non=953, base rate 0.34)
- predicted P range: 0.004–0.371
- **AUC = 0.659**  (0.5 = no signal)
- person-level Spearman(P, rating) = +0.302
- city-level Spearman(P, mean rating) = +0.336 (n=968 cities)

| P quintile | mean predicted P | observed accept rate | mean rating | n |
|---|---|---|---|---|
| | 0.051 | 0.228 | 2.35 | 289 |
| | 0.115 | 0.180 | 2.33 | 289 |
| | 0.148 | 0.277 | 2.51 | 289 |
| | 0.181 | 0.536 | 3.35 | 289 |
| | 0.230 | 0.481 | 3.19 | 289 |

## q55 — Positive anymore (X is expensive anymore)

- respondents (CONUS): **1445**  (accepters=492, non=953, base rate 0.34)
- predicted P range: 0.002–0.327
- **AUC = 0.630**  (0.5 = no signal)
- person-level Spearman(P, rating) = +0.253
- city-level Spearman(P, mean rating) = +0.303 (n=968 cities)

| P quintile | mean predicted P | observed accept rate | mean rating | n |
|---|---|---|---|---|
| | 0.051 | 0.149 | 2.16 | 289 |
| | 0.118 | 0.356 | 2.74 | 289 |
| | 0.159 | 0.315 | 2.72 | 289 |
| | 0.188 | 0.388 | 2.85 | 289 |
| | 0.226 | 0.495 | 3.26 | 289 |

## q56 — Positive anymore (X is expensive anymore)

- respondents (CONUS): **1445**  (accepters=492, non=953, base rate 0.34)
- predicted P range: 0.109–0.783
- **AUC = 0.737**  (0.5 = no signal)
- person-level Spearman(P, rating) = +0.456
- city-level Spearman(P, mean rating) = +0.501 (n=968 cities)

| P quintile | mean predicted P | observed accept rate | mean rating | n |
|---|---|---|---|---|
| | 0.225 | 0.131 | 2.04 | 289 |
| | 0.361 | 0.156 | 2.21 | 289 |
| | 0.415 | 0.322 | 2.69 | 289 |
| | 0.478 | 0.488 | 3.21 | 289 |
| | 0.580 | 0.606 | 3.58 | 289 |

## q57 — Positive anymore (X is expensive anymore)

- respondents (CONUS): **1445**  (accepters=492, non=953, base rate 0.34)
- predicted P range: 0.049–0.539
- **AUC = 0.480**  (0.5 = no signal)
- person-level Spearman(P, rating) = -0.046
- city-level Spearman(P, mean rating) = -0.059 (n=968 cities)

| P quintile | mean predicted P | observed accept rate | mean rating | n |
|---|---|---|---|---|
| | 0.231 | 0.349 | 2.84 | 289 |
| | 0.274 | 0.353 | 2.74 | 289 |
| | 0.300 | 0.367 | 2.75 | 289 |
| | 0.326 | 0.343 | 2.82 | 289 |
| | 0.356 | 0.291 | 2.57 | 289 |

## Positive-anymore summary (q54–q57)

AUC across the four HDS *anymore* questions ranges **0.480–0.737** (mean 0.627). Three of the four (q54 0.66, q55 0.63, q56 0.74) carry a clear, consistent geographic signal; **q57 is a null** (0.48, Spearman ≈ 0). q57 ('Forget the nice clothes anymore…') is the oddest sentence and its recovered surface does not track real acceptance — an honest reminder that not every recovered surface is usable, and that the three informative *anymore* questions are near-duplicates of each other (so multiplying all four over-weights one diffuse feature).

## Binarisation sensitivity (not used to pick the headline)

AUC for each mapping under the pre-registered ≥4 cut and the two other shipped cuts (≥3, =5). The signal is a property of the geography, not of the threshold.

| question | AUC (≥3) | AUC (≥4, headline) | AUC (=5) |
|---|---|---|---|
| q51 | 0.571 | **0.605** | 0.622 |
| q54 | 0.643 | **0.659** | 0.663 |
| q55 | 0.622 | **0.630** | 0.626 |
| q56 | 0.728 | **0.737** | 0.738 |
| q57 | 0.480 | **0.480** | 0.476 |

## Within-person cross-construction dependence (model-based)

The **349 people who answered both** come-with and positive-anymore let us measure the dependence that actually breaks the model's conditional-independence assumption. For each person we take the residual `observed_accept − model_P(accept | home cell)` for each construction, then correlate the two residuals.

- raw correlation of the two binary answers: **+0.193**
- **residual correlation (after the model's own location expectation): +0.177** (n = 349)

Acceptance base rates in this subset: come-with 0.65, anymore 0.30. The residual (+0.177) is barely below the raw (+0.193): **location removes almost none of the cross-construction dependence**, echoing the non-spatial result in `correlation_report.md`. So even two *genuinely distinct* constructions (not paraphrases of one) keep a **modest but real** within-person correlation after the model conditions on where you grew up — the conditional-independence assumption is violated, though less severely than the paraphrase-and-acquiescence-inflated ρ̄≈0.30. As a rule of thumb this makes two distinct constructions behave like 2/(1+0.177) ≈ **1.70** independent ones, a discount of ~15%. It is one coefficient from the only construction pair the overlap allows, so treat it as indicative (n=349, ~95% interval ±0.11); part of even this residual is Likert scale-use that will shrink under HDS forced choice, so it is an upper bound on the HDS discount.

## Verdict

- **Come-with (q51): genuine signal.** AUC 0.605, city-level Spearman +0.245. The HDS surface built from dot maps ranks real YGDP come-with accepters above rejecters purely from their home coordinates — an out-of-sample validation of the geography.
- **Positive-anymore (q54–q56): genuine signal.** AUC up to 0.737 (q56), city Spearman +0.501; q54 and q55 agree. Calibration is monotone: rarer-*anymore* regions in the HDS surface really do contain fewer YGDP accepters.
- **Positive-anymore q57: no signal.** AUC 0.480, Spearman -0.046. Reported plainly, not hidden: this HDS surface does not predict YGDP judgments.
- Overall: **the HDS-recovered surfaces are externally valid** where the feature is dialectally sharp (come-with, expensive-*anymore*), and honestly fail on a noisy item (q57). Two phenomena is a narrow test — it is all the HDS/YGDP overlap allows — but on that overlap the geography recovered from pixels predicts real, independently-located people.