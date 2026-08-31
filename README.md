# american dialects

Scraped source data for American English regional variation, and a model built on it that guesses where you grew up from how you talk.

The interesting part is that it works from data nobody published. The 2013 New York Times dialect quiz was built on respondent-level records with ZIP codes that were never released, against an API that is long dead. What survives publicly is state-level aggregates — and rendered dot maps. The sub-state geography is recovered from the pixels of those maps.

**[findings.md](findings.md)** is the report: how few questions it takes to locate someone, how well, the correction that was carefully derived, correctly measured, and turned out to be the single largest source of error in the system — and why twelve questions, the number this quiz format has used since 2013, is the right answer only for a model that has stopped learning.

## what's here

| file | rows | what it is |
|---|---|---|
| `data/hds/questions.csv` | 122 | Harvard Dialect Survey questions |
| `data/hds/answers.csv` | 697 | answer choices, national %, plot colour |
| `data/hds/state_pct.csv` | 30,197 | % choosing each answer, by state |
| `data/hds/geo/cells.csv.gz` | 5,244,045 | recovered sub-state geography: lat/lon per answer |
| `data/hds/geo/grid.npz` | 680 grids | same, as dense 200×456 arrays |
| `data/hds/geo/validation.csv` | 28,513 | recovered vs published state percentages |
| `data/ygdp/respondents.csv` | 22,102 | Yale Grammatical Diversity Project, respondent-level with lat/lon |
| `data/popvssoda/counties.csv` | 3,141 | pop/soda/coke counts per county |
| `data/popvssoda/states.csv` | 69 | same, by state/province (401,414 responses) |
| `data/cambridge/questions.csv` | 180 | Cambridge Survey of World Englishes questions |
| `data/cambridge/answers.csv` | 807 | answer choices, %, palette colour, multi-select flag |
| `data/cambridge/geo/grid.npz` | 807 grids | recovered dot geography, same 200×456 grid as Harvard |
| `data/cambridge/geo/cells.csv.gz` | 26,496,891 | same, native zoom-6 sub-cells: lat/lon per answer |
| `data/cambridge/hds_crosswalk.csv` | 512 | Cambridge↔Harvard question/answer mapping, with rejections |
| `data/cambridge/hds_agreement.md` | — | independent Harvard-vs-Cambridge agreement, cell by cell |
| `data/hds/state_n.csv` | 51 | per-state sample sizes, recovered from rounding granularity |
| `data/census/counties.csv` | 3,156 | county populations, 2003 and 2024 vintages |
| `data/census/places.csv` | 4,859 | named places ≥5,000 people, for naming an answer |
| `data/ygdp/people.csv` | 6,629 | de-duplicated respondents, raised location verified |
| `data/ygdp/crosswalk.csv` | 14 | YGDP↔Harvard question mapping: 5 accepted, 9 rejected with reasons |
| `data/model/likelihood.npz` | 680×50,888 | fitted `log P(answer \| cell)` over US land |
| `data/model/question_order.csv` | 30 | greedy question ordering and its learning curve |
| `data/model/headline_surface.csv` | 26 | accuracy vs. *k* on stress-tested simulated respondents |
| `data/model/neural_net.pt` | 8,177,664 | trained weights of the discriminative model |
| `data/model/neural_pool.npz` | 400,000 | simulated training people, plus the 1024 output clusters |
| `data/model/neural_curve.csv` | 90 | accuracy vs. *k* for all three models on identical people |
| `data/model/mover_split.csv` | 222 | every model scored separately on people who moved and people who did not |
| `data/model/mover_ratio.csv` | 111 | whether any model's posterior widens for a mover — none does |
| `data/model/mover_probe.csv` | 9 | mover status is not recoverable from the answers, with a positive control |
| `data/model/mover_mix.csv` | 12 | the admixture weight that would fix it, swept and scored |
| `data/model/mover_split_rep.csv` | 96 | the split repeated at an independent seed |
| `data/model/mover_ratio_rep.csv` | 48 | the ratios repeated at an independent seed |
| `data/model/question_signal.csv` | 122 | per question, the bits about place against the bits about the person |
| `data/model/question_order_ratio.csv` | 90 | three alternative question orderings, none of which beat the deployed one |
| `data/model/order_curve.csv` | 180 | those orderings raced against the deployed one on identical people |
| `data/uwm/questions.csv` | 154 | UWM Dialect Survey maps, scraped and then abandoned — see below |
| `data/uwm/hds_crosswalk.csv` | 121 | UWM↔Harvard candidate matches, hand-checked: 17 verbatim, 28 valid |

### what a dialect boundary is worth measuring

The published site draws eight real isoglosses, and drawing them turns out to make something measurable that a printed atlas hides. Subtracting one recovered surface from another gives the odds of one word against another at every point; the contour where the odds are even is the boundary, and the ground over which those odds swing from three-to-one to three-to-one is its width. Those widths run from 73 km for *yinz* in western Pennsylvania to 443 km for *catty-corner*, which is to say the last is barely a boundary at all — yet an atlas prints both as the same confident line. Because the surfaces are smoothed, each width is an upper bound on how sharp the real boundary is; see [findings.md](findings.md).

### a survey that could not be recovered

The project went looking for a third survey to use as a control arm, because Cambridge-vs-Harvard confounds elapsed time with population and method, and a survey run one to three years after Harvard would separate them. The **UWM Dialect Survey** (Vaux & Samuels, 2004–2006, republished as map images from 2018) looked like that survey. It is not usable, and the reasons are worth stating because they are a clean illustration of what makes pixel recovery possible in the first place.

The Harvard maps are recoverable because a dot is a countable object with a known meaning: one dot, one respondent, at a ZIP centroid. The UWM maps are smooth diverging colour surfaces carrying **no legend of any kind** — so a colour cannot be turned into a proportion, because the range it spans is unknown — and **86 of the 154 published questions do not say which answer choice each image belongs to**. Either problem alone is fatal, and neither is in the images to be extracted. `scrape/uwm.py`, `scrape/uwm_crosswalk.py` and `data/uwm/inventory.md` are kept so that the finding is reproducible and the work is not repeated; `todo.md` records what would reopen it.

One number from the planning stage deserves correcting in public. The Harvard overlap was scoped at "at least 79 questions" from matching truncated URL slugs, and described as a floor. Matching full question text and then hand-reading every candidate pair gives **17 verbatim matches and 28 valid ones**, with 8 false positives struck out. The floor was an overestimate by nearly a factor of three.

## the Harvard Dialect Survey geography

The Harvard Dialect Survey (Bert Vaux & Scott Golder, 2002–2003, 30,788 respondents) is the question set the 2013 New York Times dialect quiz was built on. Josh Katz had the respondent-level records with ZIP codes, supplied privately by Vaux. **Those records were never published**, and the NYT quiz computed its heat maps server-side against an API that is long dead. Only state-level aggregates survive publicly.

But the survey's preserved mirror at `dialect.redlog.net` renders, for every answer, a map that plots each respondent at their ZIP centroid. The geography is still there — as pixels.

`scrape/hds_geo.py` recovers it:

1. **Solve the projection.** The maps are plate carrée, 456×200. Fitting pixel coordinates against state borders whose position is known exactly (the 49th parallel; 37°N, 41°N, 42°N, 45°N; 102.05°W, 103°W, 104.05°W, 109.05°W, 114.05°W, 88.47°W) gives `lon = 0.126645x − 125.4285`, `lat = −0.127187y + 49.2117`. Residuals are under 7 km, against a pixel of ~12.7 km.
2. **Un-blend the ink.** A plotted dot is its answer's colour composited over white and antialiased, so an edge pixel is `α·colour + (1−α)·white`. Solving for α in the least-squares sense recovers fractional coverage and rejects anything off that line — basemap greys, other answers.
3. **Correct for saturation.** Where respondents are dense, dots merge and coverage pins at 1, so cities undercount. Treating respondents as Poisson within a 9×9 window, expected coverage is `1 − e^{−λ}`, so `−ln(1−f)` recovers relative density.

**Validation.** The survey published its own per-state breakdowns, computed from the raw records, so they are independent ground truth. Across 28,513 state × question × answer percentages the recovered surfaces give **r = 0.955**, mean absolute error **5.6 points**, and pick the same modal answer in **87.6%** of state-questions. Excluding states too small to raster cleanly (DC is one grid cell) r rises to 0.96. `scrape/validate_geo.py` reproduces this.

Two caveats worth carrying forward. A pixel records that *at least one* respondent nearby chose an answer, so this is a spatial-extent measure corrected toward density, not a respondent count. And only the first 12 choices per question got a plot colour, so 17 of the 697 choices have no map — `answers.csv` marks these with `has_map = 0`, and their state percentages are still complete.

## the model

`model/` turns those surfaces into a geolocator: answer some dialect questions, get a probability map of where you grew up. The point of the exercise is the **smallest number of questions** that will place someone, with a confidence that means what it says.

**What it estimates.** The place whose 2003 survey respondents your idiolect most resembles. Because dialect is acquired young that is usually where you were raised, and that is what it is scored against — but it is not birthplace, not current address, and for someone who has moved a lot it is a blend of everywhere they have lived.

**Likelihood** (`model/likelihood.py`). Three steps beyond the raw coverage, each fitted rather than chosen:

- *Order of operations.* Un-saturate at the scale dots actually merge (`uniform_filter`, then `−log1p(−f)`), and only then pool with a Gaussian. Doing it the other way round dilutes `f` so far that `−ln(1−f) ≈ f` and the saturation correction silently stops working.
- *Contrast* (`gamma`). A dot is about a cell wide, so every dot smears its owner's answer into its neighbours, and clipping saturated coverage truncates the rest. Both compress local log-odds by roughly a constant factor; raising density to a power restores them.
- *Raking.* The dot maps give within-state **shape**; the survey's published state table, computed from the raw records rather than from any map, gives the correct **level**. Iterative proportional fitting per state fuses the two. Raking alone is not enough — it lifts a whole state uniformly, so it fixed New York City and broke Buffalo. Gamma fixes contrast, raking fixes level, and both are needed.

**Fitted externally, not by eye.** `model/tune.py` scores against Pop vs. Soda — 294,079 county-level responses the surfaces never saw. Log-loss falls from 1.1608 (national split, no geography) to **0.7234**. Settings are frozen in `model/tensor.py` at `sigma=8, gamma=1.5, alpha=0.02, box=9, rake=True`.

**Inference** (`model/infer.py`). Bayes over 50,888 land cells with a 2003-population prior, plus two corrections that fix different failures:

- `tau` tempers the whole log-likelihood, because answers are not independent given location and because the surfaces themselves carry recovery error. A Southerner's *y'all*, *fixin' to*, *crawfish* and *coke* are one fact restated, and multiplying them as independent evidence collapses the posterior onto a single county. Two tempering terms were deployed and they turn out to have very different standing. The **base** temper, `TAU_BASE = 0.55`, is confirmed: untempered the model says 50% and is right 40% of the time, at 0.55 it says 50% and is right 57% and says 80% and is right 81%, and its optimum tracks surface error in the direction theory demands. The **design-effect discount** — treating k answers with within-person correlation ρ ≈ 0.18 as worth `k / (1 + (k−1)ρ)` independent ones — is not. Simulating people who genuinely violate conditional independence shows that deploying ρ = 0 beats every positive discount on median error in **253 of 253 matched comparisons**, and that the discount was the largest single source of error in the system. See [findings.md](findings.md); the constant in `model/infer.py` is unchanged pending that decision.
- A subtlety worth keeping even so, since the same trap recurs anywhere a design effect is used. Counting raw questions is wrong, because the first question is worth 0.581 bits and the twentieth 0.094 — a weak question would add a full unit of discount for almost no signal, and effective information would *peak at seven questions and decline*. Kish's effective sample size, `(Σw)² / Σw²`, is used instead.
- `eps` mixes each answer toward its national marginal, bounding what one answer can do. Measured on the fitted tensor, the median answer swings the log-posterior by 5.5 nats and the worst by 9.1 — about 13 bits, comparable to the entire entropy of the location itself. So one misclick or one bidialectal speaker really can outweigh everything else. With `eps=0`, a Pittsburgher who also says *bubbler* gets placed in Indiana, which matches nothing; at `eps=0.05` the model holds Pittsburgh and widens from 25k to 73k km², which is the honest answer.

**Question selection** (`model/choose.py`) is greedy mutual information against the running posterior, which is what lets it see redundancy: once *y'all* has moved the mass south, *fixin' to* is near-constant over everything still plausible and its information collapses.

**Two objections tested and closed.** A 114 km saturation window is far wider than a dot, so it should blunt sharp features — swept 1 to 17 cells against the external target, it is flat from 5 to 17 and *worse* at 1, costing 0.0006 nats. And one global bandwidth should not suit both a point feature and a national gradient — `model/bandwidth.py` confirms the sharpness is real (*yinz*, *yous*, *bubbler*, *grinder*, *hoagie* sharp; *you guys*, *soda*, *y'all*, *sub*, *coke* diffuse, all predicted before fitting) but fitting sigma per question buys at most **0.077%** of the log-loss. Both objections are correct in principle and negligible in practice.

**Answering with a place** (`model/infer.py`, `Places`). Every grid cell is assigned to its nearest named town, so the town ranking is a partition of the whole posterior, and towns are then merged into metro areas within 60 km. Both matter. Scoring only the cells that *contain* a listed town uses 16% of the posterior and discards the rest unevenly — it vanishes wherever people live in places too small to list — so the model reported New York while its own modal cell sat 3 km from Pittsburgh. And without merging, one right answer splits across its own suburbs into Pittsburgh, Bethel Park and Monroeville. *Yinz* alone now returns Pittsburgh 8.6%, PA 20.6%.

### a second model, without the independence assumption

Everything above factorises the likelihood across questions, which is the assumption the measured ρ = 0.177 says is false, and the assumption the design-effect discount was a failed attempt to repair. `model/neural.py` is the alternative: a discriminative network that maps an answer vector straight to a distribution over locations, so there is no factorisation to violate and no discount to undo.

**Why it has to be trained on a simulator.** No public dataset pairs an answer vector with a hometown. Harvard survives only as aggregates, Cambridge is aggregate-only, and YGDP's 1,450 located respondents answered five questions of which four are one construction — about two independent bits per person. This was checked before anything was written. So the training set comes from `model/idiolect.py`, whose people carry a calibrated idiolect and 15% mobility. That is only worth doing because the Bayes model is known to be misspecified against exactly that process; the network's hard ceiling is that **it cannot learn anything the simulator does not contain.**

**Shape.** 802 input bits — 680 for the chosen option of each answered question, plus a 122-bit mask of which were asked, so one network serves any quiz length and any selection policy. A 1024-wide trunk with three residual blocks, **8,177,664 parameters**, and a softmax over 1024 population-weighted k-means centroids rather than all 50,888 cells; that quantisation costs a median 39 km, well inside the error scale being measured. Targets are distance-decayed, `softmax(−d²/2σ²)` at σ = 100 km, so the loss rewards being near the right place rather than treating a neighbouring cluster as no better than the far side of the country. The training pool is redrawn every epoch — freezing it caused real overfitting, with validation loss turning over at epoch 36 while training loss kept falling.

**What it buys.** On 2,000 held-out simulated people with a real idiolect and 15% movers but no surface error, scored against both Bayes configurations on identical people and the identical ordering:

| questions | net | Bayes ρ = 0 (deployed) | Bayes ρ = 0.177 (the old discount) |
|---|---|---|---|
| 5 | 654 km | 685 | 897 |
| 12 | **347 km** | 431 | 891 |
| 14 | **291 km** | 361 | 807 |
| 30 | **148 km** | 176 | 903 |

The network with **five** questions beats the discounted model with any number of questions — 654 km against a best-ever 794. It also settles what the project set out to ask: **twelve was never derived**, it came from the New York Times quiz format, and it is not optimal for anything. The discounted configuration bottoms out at thirteen questions and then climbs 109 km by thirty as the discount outruns the evidence; both correctly specified models fall monotonically and are **still improving at thirty**, with no turnaround anywhere in the measured range.

That last fact is why fourteen is a judgement rather than a derivation. There is no interior minimum to find, so the stopping point comes from a stated rule about diminishing returns. Priced at 30 km of error per question asked, the deployed Bayesian model's optimum is fourteen; requiring 90% of the reduction available by thirty, it is nineteen. The network answers the same two rules with eleven and sixteen. **Fourteen** is the priced optimum of the model actually deployed and sits inside the network's bracket. An earlier draft justified fourteen by noting that the two rules agreed on it — they no longer do, and that agreement turned out to be an artefact of the twenty-question measurement cap.

**Not ready to ship**, and the reasons are specific. It has never been trained against wrong surfaces, which is the failure `TAU_BASE` exists to absorb. Its question ordering is inherited from the model it beats. And it trades a little modal-state accuracy for its distance advantage — 46.5% against 48.0% at twenty questions — because 1024 clusters blur state boundaries that Bayes resolves at full cell resolution.

### what is and is not validated

This is the part worth reading carefully.

| claim | evidence | status |
|---|---|---|
| pixel recovery reproduces the survey | r = 0.955 against 28,513 published state percentages | **measured** |
| surfaces beat a no-geography baseline | log-loss 1.1608 → 0.7234 on 294,079 external responses | **measured** |
| sub-state structure is real, not raking | within-state contrasts raking cannot fake: Pittsburgh 69.7% *pop* vs Philadelphia 1.2%, Buffalo 65.7% vs NYC 2.7% | **measured** |
| surfaces reproduce in a second survey | a differently recruited population answering the same 122 questions verbatim (`cambridge_id = hds_q + 240`, verified with zero divergence) agrees on the locally modal answer in 68% of land cells (72% on cleanly recoverable answers) against a 54% chance baseline, over 3.9M cell comparisons | **measured** |
| surfaces predict real individuals | AUC 0.61–0.74 on 1,450 located YGDP respondents, over two constructions, with one clean null | **measured, narrow** |
| answers are not independent | within-person residual correlation +0.177 after conditioning on location, n=349 | **measured** |
| ...but discounting the likelihood for it makes things worse | ρ = 0 beats every positive discount on median error in 253/253 matched comparisons, over people simulated with real idiolects, 15% mobility and 0–30pp of surface error | **measured in simulation** |
| the base temper `TAU_BASE = 0.55` is right | optimal on log score and coverage at the surface-error level the Cambridge comparison independently measured; agrees with the unrelated YGDP differential-coverage fit | **measured in simulation** |
| dropping the independence assumption beats patching it | a discriminative net cuts the discounted model's error by 61% at equal quiz length (347 km vs 891 at *k*=12) and beats its best-ever accuracy using five questions | **measured in simulation** |
| twelve questions is the right quiz length | it is optimal for nothing; the discounted model bottoms at *k*=13 and degrades thereafter, and both correctly specified models are still improving at *k*=30 | **disconfirmed in simulation** |
| the second survey's disagreement is change, not method | a survey close enough in time that real change should be near zero would separate the two; the UWM Dialect Survey was the candidate, was scraped in full, and its maps carry no legend and often no answer label, so it cannot be recovered | **attempted, not obtained** |
| *k* questions place you within *x* km | — | **not measured** |
| "80% confident" is right 80% of the time | — | **not measured on real people** |

The last two are the ones the project set out to establish, and they cannot currently be established from public data. **No public dataset has real people, with known hometowns, answering many dialect questions.** This is load-bearing — everything simulated rather than measured is simulated because of it — so it was re-checked on August 29, 2026 against DARE, YGDP, IDEA, the Speech Accent Archive, CORE, TalkBank and the crowdsourced surveys, and it still holds; see *the field as of August 2026* under [sources](#sources). YGDP is the closest and it fails twice over: its overlap with the Harvard question set is only two distinct constructions, which beat the population prior by 0.06 nats and 25 km — two syntactic questions cannot locate anyone. And its respondents are not a population sample, so the prior *alone* covers them 63/88/97% of the time at nominal 50/80/95. That sampling skew is larger than the effect being measured.

Simulation cannot close that gap but it can be made much less circular. `model/nullcheck.py` builds people who obey the model exactly, and a curve drawn on them is the model's belief about its own competence — that world does not merely fail to inform, it structurally hides the assumption most likely to be wrong. `model/idiolect.py` builds people who break the model on purpose, in three ways whose magnitudes are pinned to measurements: a within-person idiolect bisected until it exhibits the correlation YGDP measured, a mobile fraction raised somewhere other than where they are recorded, and surfaces wrong by a smooth spatially correlated field calibrated against the Harvard-versus-Cambridge disagreement. On those people, with the correlation discount off, twenty questions give a median error of 329 km, 42% of states, and credible regions covering within four points of nominal at every length — though that last figure is a population average that later turned out to conceal two opposite failures, and [findings.md](findings.md) now says so. The shapes of those three violations are still assumptions, so this is a stress test rather than a validation.

`model/nullcheck.py` also serves as the control that keeps the calibrator honest: on people who obey the model exactly it recovers tau = 1, eps = 0, coverage .510/.800/.955 and a flat PIT histogram. So a tau below 1 on real people is signal, not instrument bias.

**Closing the gap.** `model/quiz.py` plays the trick in the terminal and `site/` plays it in a browser. Both log every game with the player's real hometown, to the same file. Fifty logged games and `calibrate.py --set quiz` turns the confidence claim from an extrapolation into a measurement.

## the site

`site/server.py` serves a local, adaptive version of the quiz. No framework and no new dependencies — the standard library serves the files, Pillow draws the map, and the model behind it is the same one `quiz.py` drives from the terminal.

```
cd site
../.venv/bin/python server.py        # then http://localhost:8000
```

It loads the model once (about three seconds) and holds it in memory, so every question after that is instant: choosing the next question takes 40 ms and recomputing the posterior 340 ms. Questions are picked adaptively — each one is whichever remaining question would tell the model most about *this* player given what they have already said — so no two games ask the same fourteen. Number keys pick answers, and there is a "Guess now" affordance on every screen, because a party trick should never feel like a form.

**Both of the changes the analysis called for have now been made.** It runs the Bayes model with `RHO = 0` and it asks a hardcoded fourteen questions. The discount was the largest single source of error in the system, and twelve was the right stopping point only *because* of the discount — the discounted configuration stops improving at thirteen questions, while a correctly specified model is still improving at thirty. The two constants had to move together, because raising the question count while the discount was still active would have made the model worse rather than better.

The result screen draws the posterior over all 50,888 land cells as a single image: grey country, blue where the belief is. The raw posterior is far too peaked to look at directly, so it is normalised by its maximum and raised to a fractional power, which lifts the shoulders back into view without changing the ordering.

Every finished game is appended to `data/quiz/log.csv` in exactly the format `calibrate.py --set quiz` expects, so playing the web version feeds the one measurement this project still lacks.

**The map is also the clearest illustration of the finding above.** One worked example, drawn twice. The speaker gives the most common Pittsburgh answer to each of the fourteen questions the quiz asks, and the same fourteen answers are scored under the discount that shipped and under the value deployed now:

| | 80% region | modal state | top three metros |
|---|---|---|---|
| `RHO = 0.177` (the old discount) | 1,752,029 km² | NY | New Castle 0.6%, Bethel Park 0.5%, Baltimore 0.5% |
| `RHO = 0` (deployed) | **786,764 km²** | OH | New Castle 2.2%, Bethel Park 2.0%, Pittsburgh 1.7% |

Discounted, the posterior is tempered so hard it slides back onto the population prior: the region is 965,266 km² larger, the modal state moves from the Ohio–Pennsylvania border to New York, and the strongest place in the country is claimed at 0.6% — barely a preference at all. Undiscounted, the same evidence concentrates on the western Pennsylvania corridor and names Pittsburgh's own suburbs.

The sharper version of the same fact is what happens as the questions accumulate. Under the discount the 80% region stops moving after the fourth question and sits between 1.68 and 1.83 million km² for every quiz length from four to twenty; under `RHO = 0` it keeps contracting. The discounted model is not being cautious. It has stopped listening.

This is a worked example, not a headline result — one speaker profile, chosen because Pittsburgh has the sharpest lexical signature in the survey and so makes any error in the chain visible. It is exported by `model/export_web.py`, reproduced live in the browser on the published site, and `check.py` asserts that the two areas above are the two areas the site computes.

### the published site

`web/` is the public version, built with Astro and [Starlight](https://starlight.astro.build), and deployed to GitHub Pages by `.github/workflows/pages.yml`. It is a static build with no server behind it: the model runs in the browser, and every figure in the prose is baked in at build time from `web/src/content/generated.json`.

```
cd web
npm install
npm run dev                          # then http://localhost:4321/american-dialects/
npm run build                        # type-checks, then writes web/dist
npm run verify                       # asserts the browser model matches model/infer.py
```

Only the parts that do something ship JavaScript. The quiz, the isogloss plate, the discount slider and the two charts are hydrated islands; the prose sections are rendered to HTML at build time and ship none. The charts use `d3-scale` and `d3-shape` for their scales, ticks and path geometry, and the maps are drawn straight to canvas, because a projection that is already gridded into 50,888 cells has nothing to gain from a selection layer.

The prose acts are written in MDX, so the essay is edited as Markdown rather than as JSX. Each act is a page in Starlight's `docs` collection under `web/src/content/docs/`, which is what gives the site its sidebar, its per-page contents rail and its search index; the act order is the `sidebar` array in `astro.config.mjs`. `web/src/components/prose/` holds the wrappers Starlight does not provide — the full-bleed figure, the claim ladder in Act V, and `Stat`, which is how a number reaches the page. Interactive components are imported into the MDX and given a client directive there, which is why `recovery.mdx` can put a scrubber in the middle of a paragraph without becoming a component itself.

Three rules that fall out of this and are worth knowing before editing an `.mdx` file. Numbers still come only from `generated.json`; `check.py` scans `.mdx` alongside `.ts`, `.tsx` and `.astro` and fails on any generated figure typed into the prose. Markdown on its own line is block content, so a wrapper meant to hold prose has to be a `div` — a `p` wrapping Markdown's own `p` is invalid, and a browser closes the outer one early, which strips the styling from the text it was marking. And links between acts are real page links, so they carry the `/american-dialects/` base rather than being in-page anchors.

The generated content sits at `web/src/content/index.ts` and `generated.json`, one directory above `content/docs/`, where Starlight's loader does not glob it. That is deliberate: it is the source of truth for every number the site prints, and it is not a page.

The page ships one colour scheme. It reproduces a printed artifact from 2003 — a white sheet with blue dots on it — and on a dark ground that sheet reads as a hole burned in the page rather than as a document. Starlight defaults to dark and offers a switcher, so `web/src/styles/starlight.css` maps its palette onto the atlas tokens under both themes and hides the control, on the grounds that a toggle promising a scheme that was never designed is worse than no toggle.

## running it

Every number quoted in this README, in `findings.md` and in `eli5.md` is recomputed from the tracked CSVs by `check.py`, which also verifies that the deployed constants are each defined in exactly one place and that the prose still describes them correctly. It loads no model and takes about a second.

For the figures it validates it checks *every* printed copy, not merely that a correct one exists somewhere. The distinction is not pedantic. These documents repeat their headline numbers, several of them half a dozen times each, and a check satisfied by finding one good copy is blind to a second copy that has drifted away from it — which is very close to the failure that actually happened here once before, when the worked example in the prose turned out to be a different worked example from the one the code produced. So each validated figure also carries the number of times it is quoted, and matching is done on number boundaries rather than on substrings, since a figure that gains a digit is a different figure and should not pass by being a prefix of itself. The cost is that editing prose around a validated number means updating a count; that is the intended bargain, and this paragraph broke the rule while being written, which is the sort of thing that argues for keeping it.

```
./.venv/bin/python check.py       # 79 checks; -v to see the passing ones too
```

This exists because the report is long and its numbers were typed in by hand. It has already caught several: the README claimed 12 rows for a crosswalk that has 14, and it claimed this suite ran 48 checks when it had grown past sixty — a stale sentence sitting inside the paragraph explaining that stale sentences are what the script prevents. More importantly, `RHO`, `TAU_BASE` and `N_QUESTIONS` are each described in prose in several places, so changing one of them silently falsifies several documents at once — `check.py` turns that into a failing check instead of a stale sentence.

```
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cd scrape
../.venv/bin/python hds.py           # questions, answers, state percentages
../.venv/bin/python hds_maps.py      # 802 dot maps -> data/raw/hds_maps/
../.venv/bin/python hds_geo.py       # recover geography
../.venv/bin/python validate_geo.py  # check against published state numbers
../.venv/bin/python ygdp.py
../.venv/bin/python popvssoda.py
```

All fetches are cached under `data/raw/` (gitignored), so re-runs are free. Requests are rate-limited to ~2.5/second.

Then the model:

```
cd model
../.venv/bin/python tensor.py        # freeze the fitted likelihood (~120 MB)
../.venv/bin/python prior.py         # population prior
../.venv/bin/python isogloss.py      # 16 documented dialect boundaries
../.venv/bin/python tune.py          # refit against Pop vs. Soda
../.venv/bin/python choose.py        # greedy question ordering
../.venv/bin/python nullcheck.py     # build the control population
../.venv/bin/python calibrate.py --set nullcheck   # must return tau=1, eps=0
../.venv/bin/python calibrate.py --set ygdp --auto --strata "moved;race"
../.venv/bin/python idiolect.py --n 1200 --true-rho 0.18 --mover 0.15 \
    --surface-mae 20 --deployed-rho 0 --ks 1,5,12,20   # the stress test
../.venv/bin/python quiz.py -n 12    # play it in the terminal
```

The discriminative model is separate, because it is the only part of the project that needs `torch`:

```
cd model
../.venv/bin/python neural.py prep                      # clusters + 400k training pool
../.venv/bin/python neural.py train --epochs 80         # ~12 s/epoch on an M3 Pro with MPS
../.venv/bin/python neural.py curve --n 2000 --kmax 20  # the three-model question curve
```

`prep` takes about five minutes, almost all of it k-means. `curve` takes about twenty-five minutes, dominated by the 80,000 Bayes posteriors computed for the two comparison arms. It writes `data/model/neural_curve.csv`.

Or in a browser:

```
cd site && ../.venv/bin/python server.py
```

Derived arrays (`grid.npz`, `cells.csv.gz`, `likelihood.npz`, ~210 MB) are gitignored: each is a pure function of the CSVs that are tracked, and every one is rebuilt by the commands above.

## sources

- **Harvard Dialect Survey** — `dialect.redlog.net`, a static mirror of the 2002–2003 survey by Bert Vaux and Scott Golder. The original site at UW-Milwaukee is gone; the Wayback copy is at `web.archive.org/web/20200515154030/http://www4.uwm.edu:80/FLL/linguistics/dialect/maps.html`.
- **Yale Grammatical Diversity Project** — `ygdp.yale.edu/maps/json` indexes 28 GeoJSON files of respondent-level syntactic acceptability judgments (~250,000 ratings over 180+ sentences, collected 2015–2019). Reuse terms are not stated on the feed; confirm with YGDP before publishing.
- **Pop vs. Soda** — `popvssoda.com`, Alan McConchie. The county file needs a same-origin `Referer` header or it 401s. County rows cover 294,080 of the 401,414 total responses; the rest are Canadian or ungeocoded. Two totals appear in this report and both are correct: the source's own `SUMCOUNT` sums to 294,080, while its four category columns sum to 294,079, because Lawrence County, Ohio reports 71 responses against 70 categorised ones. `tune.py` scores the categories, so 294,079 is the number the model was actually fitted against. `check.py` asserts the difference is exactly one, so if the upstream file is ever corrected the discrepancy will surface rather than rot.
- **Cambridge Online Survey of World Englishes** — Bert Vaux & Marius Jøhndal, live at `tekstlab.uio.no/cambridge_survey` (Text Laboratory, University of Oslo; CC BY-NC-SA 3.0). The `survey.johndal.com` front end is dead but the Oslo mirror serves everything. Its `/maps` index lists **180 questions** (non-contiguous IDs 1–362); each question page has a hex-coded legend and plots respondents as Web Mercator raster tiles at `/maps/<id>/{x}/{y}/{z}.png` (note the unusual x/y/z path order), zoom 4–9. `scrape/cambridge.py` scrapes the questions/answers; `scrape/cambridge_geo.py` recovers the dot geography at zoom 6 (~1.9 km/pixel, ~7× finer than the Harvard GIFs). It is worldwide in scope but **~86% of the ink is CONUS** (UK/Ireland ~5%, Canada ~1.6%, Australia ~1.4%), so it is usable as US validation. Respondent count is not published anywhere. Crucially, question IDs **241–362 are a verbatim re-run of the entire Harvard survey (HDS q1–122)** answered by a later, different population — an independent replication. The mapping is exactly `cambridge_id = hds_q + 240` and has been checked question by question: all 122 Harvard question texts match their Cambridge twin, none diverge. `scrape/cambridge_crosswalk.py` builds `data/cambridge/hds_crosswalk.csv`; `scrape/cambridge_validate.py` compares the two recovered surfaces cell-for-cell (`data/cambridge/hds_agreement.md`): 25 high-confidence lexical questions, pooled Pearson r ≈ 0.64, locally-modal answer agreeing in ~68–74 % of US-land cells against a 54 % chance baseline, and every documented isogloss (hoagie/Philadelphia, poor boy/New Orleans, pop/Buffalo, soda/NYC, frappe/Boston, bubbler/RI) independently reproduced.

One source was checked and is not usable. Jack Grieve's Word Mapper county matrices (97,246 words × 3,075 counties, CC BY 4.0) are advertised at `sites.google.com/view/grievejw/word-mapper` but every Google Drive link 404s. All four were re-fetched individually on August 29, 2026 and all four still return 404. Search results suggest a possible rehost under Grieve's OSF profile at `osf.io/56umh`, which could not be confirmed: the page is a JavaScript application that renders nothing to a plain fetch, so it needs checking in a real browser before the source is written off for good.

### the field as of August 2026

The constraint this project is built around was re-checked rather than assumed, because it is load-bearing: everything simulated instead of measured is simulated *because* of it.

**It still holds. No public dataset released 2024–2026 pairs an individual's dialect answer vector with a sub-state location.** DARE (`daredictionary.com`) is now fully digitised but is subscription-only, has collected no new fieldwork since the 1990s, and offers no raw export. YGDP published an addendum in 2025 — Wood & Pereira, *Addendum to the Mapbook of Syntactic Variation in American English: Survey Results, 2020–2021*, `elischolar.library.yale.edu/ygdp/26/`, covering *hella*, causal *where*, the alternative *one* construction, intensificational *fully* and lonely transitives — but it is syntactic-only and respondent-level data still requires IRB approval, so it does not widen the two-construction overlap that limits `data/ygdp/crosswalk.csv`. IDEA and the Speech Accent Archive are both growing but are audio archives built on a single elicitation paragraph, which is not a questionnaire.

Two things did change and both matter here.

**Someone is now collecting exactly the missing resource.** DialectGuessr (`dialectguessr.com`) is live and worldwide, asks 29 questions drawn from a bank of 60, scores against 65 regional profiles, and after each game invites the player to pin their hometown — so it is accumulating answer vectors paired with declared locations, which is the thing whose absence forces this project to simulate. None of it is publicly released and there is no API. It bills itself explicitly as the successor to the Harvard survey and the 2013 quiz. The assets are complementary rather than competing: theirs is people, this project's is validated sub-state surfaces and a calibrated model, and neither substitutes for the other.

**A verbatim Harvard re-run is happening again.** Bert Vaux collaborated with the Houston Chronicle on a Texas dialect survey in 2026, built in the HDS tradition with Texas-specific items (*feeder* / *access road* / *frontage road*) and a companion Spanish-language survey. Results are forthcoming and Texas-only, but it is the same pattern that produced the Cambridge replication this project depends on.

Also worth recording, because it changes how the surfaces should be read rather than how they were built: the large regional **phonological** shifts are measurably retreating, while the **lexical** variation this project actually measures is not. The Southern Vowel Shift shows a sharp generational cliff — Renwick, Stanley, Forrest & Glass, "Boomer Peak or Gen X Cliff? From SVS to LBMS in Georgia English," *Language Variation and Change* (2023) — the Northern Cities Shift is in retreat or counter-shift across Michigan and upstate New York, and both are being absorbed into the Low-Back-Merger Shift, which PADS vol. 104 (2019) reframes as a continental realignment rather than a California one. Against that, *pop*/*soda*/*coke* is documented as essentially stable with only marginal urban *soda* creep, and *y'all* is spreading nationally while being re-indexed from "Southern" to gender-neutral and inclusive (McCurdy, *Y'all Means All*, Penn MA thesis, 2023, n = 1,064) — which is a change in what the word signals, not in where it is said. See `todo.md`, which turns the contrast into a question this repository's data can already answer.

**On comparing this project to published geolocation numbers.** The state of the art for locating text is a median error under 30 km globally and under 15 km on US data (Lutsai & Lampert, "Predicting the geolocation of tweets using transformer models on customized data," *JOSIS*, 2024). That number is not this project's benchmark and should not be quoted as though it were: it locates a corpus of tweets from full text plus posting metadata, whereas this locates a person from roughly thirty bits of forced-choice answers, with no text and no metadata. The honest comparison is against the population prior, which is what `findings.md` uses throughout. Relatedly, the 2023 Twitter API shutdown ended the geotagged-corpus line of work that produced Grieve's county matrices, which makes survey-recovered geography a more attractive path now than it was in 2019, not less.

Finally, one adjacent project worth knowing about: `mydialect.us` is a clean modern re-implementation of the Harvard survey with a free CSV download and seven interactive views, including Gaussian-smoothed maps. It works from the same published **state-level** aggregates, so its smooth maps are interpolated between 51 values rather than recovered below the state, which is the distinction this project exists to make.

## background reading

- Josh Katz, *Speaking American: How Y'all, Youse, and You Guys Talk* (ISBN 978-0544703391)
- Katz's method write-up, archived: `web.archive.org/web/20140530075447/http://www4.ncsu.edu:80/~jakatz2/project-dialect.html`
- Rick Aschmann's pronunciation-based dialect map, `aschmann.net/AmEng/` — reference only, no structured data behind it
