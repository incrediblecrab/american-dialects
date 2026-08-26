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
| `data/ygdp/crosswalk.csv` | 12 | YGDP↔Harvard question mapping, with the rejections |
| `data/model/likelihood.npz` | 680×50,888 | fitted `log P(answer \| cell)` over US land |
| `data/model/question_order.csv` | 20 | greedy question ordering and its learning curve |
| `data/model/headline_surface.csv` | 26 | accuracy vs. *k* on stress-tested simulated respondents |
| `data/model/neural_net.pt` | 8,177,664 | trained weights of the discriminative model |
| `data/model/neural_pool.npz` | 400,000 | simulated training people, plus the 1024 output clusters |
| `data/model/neural_curve.csv` | 60 | accuracy vs. *k* for all three models on identical people |

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

| questions | net | Bayes ρ = 0 | Bayes ρ = 0.177 (deployed) |
|---|---|---|---|
| 5 | 654 km | 685 | 897 |
| 12 | **343 km** | 444 | 760 |
| 20 | **199 km** | 264 | 847 |

The network with **five** questions beats the deployed model with any number of questions. It also settles what the project set out to ask: **twelve was never derived** — it came from the New York Times quiz format — and it is the optimum only for the deployed configuration, which bottoms at eleven-to-twelve and then loses 88 km by twenty as the discount outruns the evidence. Both correctly specified models are still improving at the twenty-question measurement cap. For the network, 90% of the achievable reduction arrives by fourteen questions, which is also the optimum if a question is priced at 30 km. **Fourteen, provisionally**, pending a question ordering re-derived without the discount.

**Not ready to ship**, and the reasons are specific. It has never been trained against wrong surfaces, which is the failure `TAU_BASE` exists to absorb. Its question ordering is inherited from the model it beats. The sweep caps at twenty because `question_order.csv` has twenty rows. And it trades a little modal-state accuracy for its distance advantage — 46.5% against 48.0% at twenty questions — because 1024 clusters blur state boundaries that Bayes resolves at full cell resolution.

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
| dropping the independence assumption beats patching it | a discriminative net halves deployed error at equal quiz length (343 km vs 760 at *k*=12) and beats the deployed model's best-ever accuracy using five questions | **measured in simulation** |
| twelve questions is the right quiz length | it is optimal only for the deployed configuration, which stops improving at *k*=12 and degrades to *k*=20; correctly specified models are still improving at the measurement cap | **disconfirmed in simulation** |
| *k* questions place you within *x* km | — | **not measured** |
| "80% confident" is right 80% of the time | — | **not measured on real people** |

The last two are the ones the project set out to establish, and they cannot currently be established from public data. **No public dataset has real people, with known hometowns, answering many dialect questions.** YGDP is the closest and it fails twice over: its overlap with the Harvard question set is only two distinct constructions, which beat the population prior by 0.06 nats and 25 km — two syntactic questions cannot locate anyone. And its respondents are not a population sample, so the prior *alone* covers them 63/88/97% of the time at nominal 50/80/95. That sampling skew is larger than the effect being measured.

Simulation cannot close that gap but it can be made much less circular. `model/nullcheck.py` builds people who obey the model exactly, and a curve drawn on them is the model's belief about its own competence — that world does not merely fail to inform, it structurally hides the assumption most likely to be wrong. `model/idiolect.py` builds people who break the model on purpose, in three ways whose magnitudes are pinned to measurements: a within-person idiolect bisected until it exhibits the correlation YGDP measured, a mobile fraction raised somewhere other than where they are recorded, and surfaces wrong by a smooth spatially correlated field calibrated against the Harvard-versus-Cambridge disagreement. On those people, with the correlation discount off, twenty questions give a median error of 329 km, 42% of states, and credible regions covering within four points of nominal at every length. The shapes of those three violations are still assumptions, so this is a stress test rather than a validation.

`model/nullcheck.py` also serves as the control that keeps the calibrator honest: on people who obey the model exactly it recovers tau = 1, eps = 0, coverage .510/.800/.955 and a flat PIT histogram. So a tau below 1 on real people is signal, not instrument bias.

**Closing the gap.** `model/quiz.py` plays the trick in the terminal and `site/` plays it in a browser. Both log every game with the player's real hometown, to the same file. Fifty logged games and `calibrate.py --set quiz` turns the confidence claim from an extrapolation into a measurement.

## the site

`site/server.py` serves a local, adaptive version of the quiz. No framework and no new dependencies — the standard library serves the files, Pillow draws the map, and the model behind it is the same one `quiz.py` drives from the terminal.

```
cd site
../.venv/bin/python server.py        # then http://localhost:8000
```

It loads the model once (about three seconds) and holds it in memory, so every question after that is instant: choosing the next question takes 40 ms and recomputing the posterior 340 ms. Questions are picked adaptively — each one is whichever remaining question would tell the model most about *this* player given what they have already said — so no two games ask the same twelve. Number keys pick answers, and there is a "Guess now" affordance on every screen, because a party trick should never feel like a form.

**Two things the site does not yet reflect.** It runs the Bayes model with `RHO = 0.177` and it asks a hardcoded twelve questions, and the analysis has since undercut both. The discount is the largest single source of error in the system, and twelve is the right stopping point only *because* of the discount — the deployed configuration stops improving at twelve, while a correctly specified model is still improving at twenty. Fourteen is the current recommendation. Neither change has been made, because both alter deployed behaviour.

The result screen draws the posterior over all 50,888 land cells as a single image: grey country, blue where the belief is. The raw posterior is far too peaked to look at directly, so it is normalised by its maximum and raised to a fractional power, which lifts the shoulders back into view without changing the ordering.

Every finished game is appended to `data/quiz/log.csv` in exactly the format `calibrate.py --set quiz` expects, so playing the web version feeds the one measurement this project still lacks.

**The map is also the clearest illustration of the finding above.** The same twelve answers, drawn twice — on the left the deployed `RHO = 0.177`, on the right `RHO = 0`:

| | 80% region | top three metros |
|---|---|---|
| `RHO = 0.177` (deployed) | 1,315,932 km² | Chicago 4.5%, New York 4.4%, Worcester 3.7% |
| `RHO = 0` (recommended) | **333,927 km²** | Worcester 10.3%, Springfield 9.7%, Milwaukee 7.8% |

Discounted, the posterior is tempered so hard it slides back onto the population prior, and the map is a picture of where Americans live — Los Angeles, Phoenix, Houston, Miami, all lit up by answers that said nothing about them. Undiscounted, the same evidence resolves to a contiguous Great Lakes and New England corridor four times smaller. That is what "the discount slides the region toward cities rather than widening it around the evidence" looks like when you draw it.

## running it

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
- **Pop vs. Soda** — `popvssoda.com`, Alan McConchie. The county file needs a same-origin `Referer` header or it 401s. County rows cover 294,080 of the 401,414 total responses; the rest are Canadian or ungeocoded.
- **Cambridge Online Survey of World Englishes** — Bert Vaux & Marius Jøhndal, live at `tekstlab.uio.no/cambridge_survey` (Text Laboratory, University of Oslo; CC BY-NC-SA 3.0). The `survey.johndal.com` front end is dead but the Oslo mirror serves everything. Its `/maps` index lists **180 questions** (non-contiguous IDs 1–362); each question page has a hex-coded legend and plots respondents as Web Mercator raster tiles at `/maps/<id>/{x}/{y}/{z}.png` (note the unusual x/y/z path order), zoom 4–9. `scrape/cambridge.py` scrapes the questions/answers; `scrape/cambridge_geo.py` recovers the dot geography at zoom 6 (~1.9 km/pixel, ~7× finer than the Harvard GIFs). It is worldwide in scope but **~86% of the ink is CONUS** (UK/Ireland ~5%, Canada ~1.6%, Australia ~1.4%), so it is usable as US validation. Respondent count is not published anywhere. Crucially, question IDs **241–362 are a verbatim re-run of the entire Harvard survey (HDS q1–122)** answered by a later, different population — an independent replication. The mapping is exactly `cambridge_id = hds_q + 240` and has been checked question by question: all 122 Harvard question texts match their Cambridge twin, none diverge. `scrape/cambridge_crosswalk.py` builds `data/cambridge/hds_crosswalk.csv`; `scrape/cambridge_validate.py` compares the two recovered surfaces cell-for-cell (`data/cambridge/hds_agreement.md`): 25 high-confidence lexical questions, pooled Pearson r ≈ 0.64, locally-modal answer agreeing in ~68–74 % of US-land cells against a 54 % chance baseline, and every documented isogloss (hoagie/Philadelphia, poor boy/New Orleans, pop/Buffalo, soda/NYC, frappe/Boston, bubbler/RI) independently reproduced.

One source was checked and is not usable. Jack Grieve's Word Mapper county matrices (97,246 words × 3,075 counties, CC BY 4.0) are advertised at `sites.google.com/view/grievejw/word-mapper` but every Google Drive link 404s.

## background reading

- Josh Katz, *Speaking American: How Y'all, Youse, and You Guys Talk* (ISBN 978-0544703391)
- Katz's method write-up, archived: `web.archive.org/web/20140530075447/http://www4.ncsu.edu:80/~jakatz2/project-dialect.html`
- Rick Aschmann's pronunciation-based dialect map, `aschmann.net/AmEng/` — reference only, no structured data behind it
