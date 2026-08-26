# How many questions does it take to find where someone is from?

This is the phase 2 report. Phase 1 recovered the Harvard Dialect Survey's respondent locations from the pixels of its rendered dot maps, because the sub-state geography was never published in any other form. This phase asks what those recovered surfaces are actually good for: how few questions can locate a person, how well, and with how much confidence.

The short answer is twenty questions, 329 km, 42% of states, and a stated confidence you can roughly believe. Getting there meant removing a correction that had been carefully derived, correctly measured, and separately debugged, and which was nonetheless the largest single source of error in the system. Most of this report is about how that was established, because the reasoning matters more than the number.

A later section adds a second model and, with it, the answer to the question the project actually asked. Twelve questions was inherited from the New York Times quiz format and never derived. It turns out to be the right stopping point *only for the configuration that was deployed*, which stops improving at twelve and then gets worse — and to be about two questions short for any correctly specified model.

## The headline

1,200 simulated respondents who **violate the model in three calibrated ways at once**: a within-person idiolect at the measured ρ = 0.18, 15% of them raised somewhere other than where they are recorded, and surfaces wrong by 20 points of choice probability — the middle of the range the Cambridge cross-survey comparison measured. Scored with `TAU_BASE` at its fitted 0.55 and the correlation discount switched off, which the next section shows is the right configuration. Median great-circle error, share of correct modal states, and the coverage of the model's own stated credible regions.

| questions | median km | state | 50% region | 80% region | 95% region |
|---|---|---|---|---|---|
| 1 | 1125 | 10.4% | 52.2% | 82.1% | 95.3% |
| 3 | 955 | 16.3% | 54.3% | 82.6% | 96.0% |
| 5 | 833 | 20.7% | 56.3% | 83.3% | 95.8% |
| 8 | 660 | 26.8% | 56.9% | 83.7% | 95.3% |
| 12 | 497 | 34.7% | 56.6% | 82.3% | 93.6% |
| 16 | 401 | 38.3% | 55.6% | 81.3% | 91.8% |
| **20** | **329** | **42.2%** | **56.8%** | **80.7%** | **91.2%** |

Two things to read off it. Error falls at essentially every question — from 1,125 km at one to 329 at twenty, with a single 7 km uptick between three and four that is well inside noise, and no peak or turnaround anywhere — so the answer to "how many questions" is "as many as the player will sit through". And the stated confidence is close to honest at every length, running a few points cautious at the 50% level and a few points optimistic at 95%, with a mean absolute calibration error between 0.015 and 0.039 throughout. This model can say "eighty percent" and roughly mean it.

That is a different report from the one this section carried a day ago, and the reason is a single parameter. Run the identical people through the configuration that was deployed — the same `TAU_BASE`, but with the correlation discount at its measured ρ = 0.177:

| questions | 5 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| ρ = 0 | **833 km / 20.7%** | **660 / 26.8%** | **497 / 34.7%** | **401 / 38.3%** | **329 / 42.2%** |
| ρ = 0.177 | 979 / 16.0% | 940 / 19.7% | 928 / 20.4% | 954 / 19.1% | 1028 / 17.9% |

Three times the error at twenty questions, less than half the state accuracy, and the curve turns over instead of descending. The discount was not a conservative choice. It was the single largest source of error in the system, and it was introduced on purpose, from a correctly measured number, by an argument that turns out not to apply.

## The parameter, and the argument about it that was wrong

ρ is the residual correlation between two answers from the same person after subtracting what the model expects given where they were raised. If it is zero, answers are conditionally independent given location and the likelihood factorises. If it is positive, a person has an idiolect — they are broadly standard or broadly regional, they moved, they read a lot — and their answers repeat information about that person rather than about that place.

ρ = 0.177 is measured, not assumed, and as far as anyone can tell it is measured correctly. It comes from 349 YGDP respondents who answered two different constructions, with a 95% interval of roughly ±0.11. The raw correlation is 0.193 and conditioning on location moves it only to 0.177, which is the part that matters: the dependence between a person's answers is almost entirely **not** explained by geography. It is the person. The measurement is sound and the phenomenon is real.

What was wrong was the next step. The survey statistician's design effect says that k observations with pairwise correlation ρ are worth k / (1 + (k−1)ρ) independent ones, a quantity that ceilings at 1/ρ — 5.65 answers at ρ = 0.177. That formula was imported to temper the likelihood, raising it to a shrinking power as the quiz lengthened, and it is the reason the accuracy curve turned over at twelve questions.

But the design effect governs the sampling variance of **an estimator of a population quantity from a clustered sample**. It answers: given that I interviewed m people per village, how precisely do I know the national mean? This model is doing something else entirely. It has one person, that person's own k answers, and a latent location to infer. The person is not a cluster drawn from a population whose mean is wanted; the person is the entire object of inference. Within-person correlation genuinely does break the likelihood factorisation and genuinely should be handled — but the repair is to model the dependence, with a person-level random effect integrated out, not to raise the whole likelihood to a power.

Tempering is a crude stand-in for that, and it is crude in exactly the way that hurts. It discounts all the evidence uniformly, including the part that was never redundant. It discounts by an amount that grows with question count whether or not the new questions overlap the old ones — even though the greedy selection has already been choosing questions specifically to minimise that overlap. And, as the experiment below shows, it corrects for something the posterior was already handling by itself.

There was also a second, weaker reason to distrust the number in this application. The two constructions it was measured on are *positive-anymore* and *come-with*, both syntactic acceptability judgments, which share an acquiescence factor and share Likert method variance. The quiz asks almost entirely lexical items, and there is no acquiescence axis in choosing what to call a crustacean. That argument suggested 0.177 was too high for the deployed items. It turns out not to matter, because the right value to deploy is zero across the whole plausible range and well past it.

## An error worth recording

Before any of that was understood, the discount had a bug of its own, and it is worth recording because it is the kind that hides.

The first version counted questions. That is correct only if every question carries the same information, and they do not: the first question in the ordering is worth 0.581 bits and the twentieth 0.094, a factor of six. Counting them equally means a weak question increments the discount by a full unit while contributing almost nothing. The effect was not subtle — effective information **peaked at seven questions** and was 18% lower at twenty than at seven.

The repair is Kish's effective sample size for unequal weights, (Σw)² / Σw², which counts near k when weights are even and much less when a few dominate. Substituting it for k moves the peak from seven to ten, and a monotonicity floor — more evidence may never leave the model with less effective evidence than it already had — then holds it flat from ten to twenty instead of letting it slide. Effective information at twenty goes from 17.6% below its own peak to sitting exactly on it, which is 31% more than the raw count allowed, and the median error at twenty fell from 800 km to 733 km.

All of which was a correct and careful repair to a mechanism that should not have been there. The residual peak-then-decline was then rationalised, in an earlier draft of this report, as the honest arithmetic of a 5.65-answer ceiling — the model telling the truth about what it had been told to believe. That reading was too charitable by half. The right response to a model that gets worse when you tell it more is not to explain why, it is to check whether the belief causing it survives contact with people who violate it. That check is the next section, and the belief does not survive.

## The experiment that settles it

Everything above prices the discount in the wrong world. `model/nullcheck.py` draws each answer independently given the true cell, so the within-person correlation in that population is zero by construction. Sweeping ρ against those people therefore does not ask what happens if people are correlated. It asks what it costs to insure against a risk that is provably absent, and the answer is always "more than nothing", monotonically. That sweep prices the insurance. It cannot price the risk.

`model/idiolect.py` supplies the missing half by simulating people who genuinely violate the model, in three different ways, each calibrated against something measured rather than guessed.

**An idiolect.** Each person gets a latent u ~ N(0,1) that tilts their answers toward or away from nationally marked variants: log P(a | cell) gains θ·u·z_a, where z_a is the national surprisal of choice a, centred and scaled to unit variance within each question so θ means the same thing whether the question has two choices or nine. A person with u > 0 is broadly regional and says *yinz* and *bubbler* and *pop*; a person with u < 0 is broadly standard and says *you guys* and *water fountain* and *soda*. Geography is untouched, so this violates conditional independence and nothing else. θ is not interpretable, so it is bisected until the realised correlation matches a target — measured with the same estimator that produced 0.177 from YGDP, imported from `model/ygdp_validation.py` rather than reimplemented, so the simulated number and the measured number are the same quantity computed by the same code. Targets of 0, 0.05, 0.10, 0.18 and 0.30 are hit to within 0.01.

**Mobility.** Fifteen percent of people are raised somewhere other than where they are recorded, so their speech comes from an independently drawn cell. This is the opposite kind of damage: every answer agrees, and they all agree on the wrong place, so the posterior is narrow and confidently wrong. It is also the most common real reason a person's answers will not match the hometown they type into the quiz.

**Wrong surfaces.** The perturbation no other simulation in this project contains. The people are drawn from surfaces that differ from the model's by a smooth, spatially correlated field, calibrated by bisection to a target mean absolute shift in choice probability. Spatial correlation is the right shape: a dot map read slightly wrong is read slightly wrong over a whole neighbourhood, because blur and colour bleed vary smoothly, and white noise would average out over the many cells a posterior touches and understate the damage badly. This is the error `TAU_BASE` exists to absorb. It is bracketed at 0, 10, 20 and 30 points, against the 9-to-28-point range the Cambridge comparison actually measured between two independent recoveries.

### The correlation discount is harmful in every world tested

At the measured ρ = 0.18, with 15% movers and 20 points of surface error — the most realistic cell in the design — holding `TAU_BASE` at its fitted 0.55:

| deployed ρ | k=5 | k=12 | k=20 |
|---|---|---|---|
| **0** | **833 km / 21%** | **497 km / 35%** | **329 km / 42%** |
| 0.177 | 979 km / 16% | 928 km / 20% | 1028 km / 18% |

Across the whole design — 5 true correlations from 0 to 0.30, with and without movers, four surface-error levels, four values of `TAU_BASE`, and question counts from 5 to 20 — deploying ρ = 0 beats every positive discount on median error in **253 of 253 matched comparisons**: 103 of them head-to-head against the deployed 0.177, the other 150 against intermediate values of 0.05, 0.09 and 0.13. Not one loss, not one tie. There is no corner of the design where the discount improves the estimate, including corners where the correlation it models is real, large, and nearly double the value deployed.

The single most adversarial cell is worth showing on its own, because it is the one the discount should win if it is ever going to. True ρ = 0.30, 15% movers, and 30 points of surface error — every assumption the model makes, broken at once, with the correlation the discount exists to handle running at nearly twice the deployed value. The tilt is calibrated to 0.303 on clean draws; once movers and wrong surfaces are layered on, the same estimator reads 0.245 on the population actually scored, so the honest description is "targeted at 0.30, verified at no less than 0.245".

| k | 5 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| ρ = 0 | **903 km / 18%** | **825 / 23%** | **586 / 29%** | **499 / 33%** | **457 / 36%** |
| ρ = 0.177 | 1039 / 14% | 1054 / 17% | 1033 / 17% | 1089 / 16% | 1138 / 15% |

Undiscounted, error falls at every single step and calibration error stays between 0.01 and 0.04. Discounted, the curve turns over and drifts back up to 1,138 km, and calibration error is *worse* as well, 0.07 against 0.04. The discount loses on distance, on log score and on calibration simultaneously, in the world built specifically to favour it.

On the log score, which is the proper scoring rule and the only metric a discount can legitimately improve, there is exactly one pocket where ρ = 0.177 wins — and it is the pocket that explains the whole result. Holding `TAU_BASE` at 0.40, 0.55 or 0.75, ρ = 0 wins the log score 12 times out of 12 at each. Set `TAU_BASE` to 1.0, removing the base temper entirely, and ρ = 0.177 wins 5 of 12. The correlation discount is not correcting for correlation. It is supplying tempering, badly and with a spurious dependence on question count, in a model that already has a tempering parameter fitted for the purpose. Where that parameter is present the discount is pure loss; where it is absent the discount partially stands in for it.

One metric is mixed overall, and it is the one where a discount ought to do best. On mean absolute calibration error, ρ = 0.177 is the better setting in 33 of the 103 head-to-head cells. The pattern is the same story told again: the wins cluster where the base temper is wrong — nine of twelve cells at `TAU_BASE` = 1.0, and three of four at 0.40 with only five questions asked. At the deployed 0.55 it wins 17 of 62, close enough to a coin flip to be noise, and it buys those coin flips at three times the median error. In the worst corner above it does not even manage the coin flip. A correction that sometimes nudges a calibration statistic while reliably tripling the distance is not a conservative choice.

Two further reasons it fails, both structural. The design-effect argument assumes each answer's evidence is uniformly inflated, but within-person correlation partly announces itself: a marked-speaking person's answers point to *different* regions — *yinz* to Pittsburgh, *bubbler* to Milwaukee and Boston, *hoagie* to Philadelphia — and Bayes widens the posterior on its own when they disagree. And tempering toward a population prior does not widen the credible region around the evidence; it slides it toward cities, so it fails to buy even the honesty it is supposed to buy.

### The base discount, by contrast, is doing exactly its job

The same runs sweep `TAU_BASE` while holding ρ at 0. Coverage after twenty questions, at 20 points of surface error, against nominal 50 / 80 / 95:

| `TAU_BASE` | 50% | 80% | 95% | log score | median km |
|---|---|---|---|---|---|
| 0.40 | 61.2% | 85.4% | 94.4% | −8.13 | 383 |
| **0.55** | **56.8%** | **80.7%** | **91.2%** | **−8.05** | 329 |
| 0.75 | 48.9% | 74.0% | 86.4% | −8.13 | 309 |
| 1.00 | 40.2% | 65.9% | 81.8% | −8.43 | 303 |

Untempered, the model is genuinely overconfident — it says 50% and is right 40% of the time, says 95% and is right 82%. Over-tempered at 0.40 it becomes timid in the other direction. The fitted 0.55 sits almost exactly on nominal at the 80% level and errs slightly cautious elsewhere, and it wins the log score.

The best value drifts in the direction theory demands. Averaged over question counts, the optimum is 0.75 at 0 and 10 points of surface error and 0.55 at 20 and 30. Less error in the surfaces, less tempering needed. That the fitted value lands on the optimum precisely in the 20-to-30-point band — the band the Cambridge cross-survey comparison independently measured — is a genuine triangulation. `TAU_BASE = 0.55` was fitted from YGDP by an entirely different argument, about differential coverage on a metro-skewed sample, and had no way to know what the Cambridge maps would say.

There is a second drift in the same table, and it is the grain of truth the design effect was reaching for. Holding surface error fixed, the best base also falls as the quiz lengthens: 0.75 at five questions, 0.55 at twenty, at every error level. So the intuition behind the discount — more answers, more tempering — is not wrong in direction. It is wrong in magnitude, and by a lot. What the data want is 0.75 → 0.55, a factor of 0.73 across a fourfold increase in k. What the deployed schedule delivers, once the Kish weights are folded in, is a multiplier of 0.607 on the base at five questions and 0.301 at twenty, a factor of 0.50 — twice as steep. And the level is worse than the slope: at twenty questions the deployed tau is 0.165 where the best available is 0.55, so the model was tempering **three times harder than anything in the design supports**. A correction can be pointed the right way and still be the largest error in the system.

### And the peak disappears

With ρ set to 0, median error falls monotonically with every question asked, in all ten combinations of true correlation and mobility, and at all four surface-error levels. The worst corner of the design was then run specifically to look for a turnaround — true ρ = 0.30, 15% movers, 30 points of surface error — and it still improves at every step, 903 km down to 457. The turnaround at twelve questions was never a property of respondents. It was the schedule discounting itself faster than the questions could inform it.

### What this experiment cannot say

The idiolect tilt is not perfectly marginal-preserving. Softmax is convex in the tilt direction, so pushing a person toward marked variants shifts the population marginals slightly too: measured on 8,000 simulated people, the mean total-variation distance from the model's own national marginal is 0.9 points at θ = 0, which is sampling noise, rising to 3.8 points at ρ = 0.18 and 6.0 at ρ = 0.30. So the high-correlation populations carry a few points of surface error on top of the correlation. That makes the conclusion about ρ stronger rather than weaker — those populations are doubly misspecified and the discount still does not help — but it does mean the ρ axis and the surface-error axis are not perfectly orthogonal above ρ ≈ 0.10.

The shape of the surface perturbation is an assumption, not a measurement. A 500 km smooth field is a reasonable stand-in for blur and colour bleed, but the real recovery error may be structured differently, concentrated at isogloss boundaries or in sparsely sampled regions. And all of this is one seed at n = 1,200; the margins are large enough that seed noise cannot flip the sign, but the third digit of any number here is not meaningful.

## What the questions are


Chosen greedily by mutual information against the running posterior, which is what lets the selection see redundancy: once *y'all* has moved the mass south, *fixin' to* is nearly constant across everything still plausible and its marginal value collapses.

The ordering is in `data/model/question_order.csv`. The first twelve, which is roughly as far as a party guest will sit still:

1. sweetened carbonated beverage (soda / pop / coke)
2. the small grey crustacean that rolls into a ball
3. what "the City" refers to
4. the small road parallel to the highway
5. covering a house in toilet paper
6. the night before Halloween
7. general term for a big fast road
8. the long sandwich with cold cuts
9. drive-through liquor store
10. the thing you drink water from at school
11. rubber-soled shoes worn in gym class
12. a sale of unwanted items in your yard

Note how few of these are the famous ones. *Soda/pop/coke* leads because it splits the country three ways along lines that cross other bundles. But *you all* does not appear until fifteenth, because by then the south has already been resolved by cheaper questions and it has little left to add. The dot maps' most-photographed isoglosses are not its most informative ones.

## What is validated and what is not

The distinction matters more here than the numbers. Simulating respondents from the same surfaces being tested, and letting them obey every assumption the model makes, guarantees a flattering answer and tests nothing — which is why the accuracy table at the top is drawn against people built to break the model instead. Even so, a simulation cannot promote itself to a validation, and the line below is drawn accordingly.

**Externally validated.** The recovered surfaces reproduce the survey's published state percentages at r = 0.955, on data the pixel recovery never saw. They were tuned against county-level *pop vs soda* returns, an entirely independent collection of 294,079 responses across 3,076 counties, cutting log-loss from **1.1608** — the national split with no geography at all — to **0.7234**. They reproduce 16 of 16 pre-registered isogloss locations. And on YGDP's real located respondents they rank correctly above chance: AUC 0.737, 0.659, 0.630 on positive-anymore and 0.605 on come-with, with q57 coming back 0.480, an honest null that was flagged low-confidence *before* the result was seen.

**Corroborated by a second survey.** The Cambridge Survey of World Englishes asked the Harvard question set again, of a differently recruited population several years later, and published its results the same way, as rendered dot maps. The re-run is verbatim and complete: Cambridge question IDs 241 through 362 are Harvard q1 through q122 in order, `cambridge_id = hds_q + 240`, and all 122 question texts match their twin with zero divergence. It is a world survey rather than a US one, but about 86% of the plotted ink falls in the lower 48.

Recovering it by the same pixel method and comparing on 25 high-confidence question pairs gives 3,930,194 per-cell comparisons. The two surveys agree on the locally most common answer in **68% of US land cells**, 73% density-weighted and 72% restricted to the cleanly recoverable answers, against a **54% chance baseline** computed from independent modal maps with the same marginals. Pooled cell-level correlation is 0.645, which is moderate rather than high.

Agreement tracks sharpness exactly as it should: 99% on milkshake versus frappe, 92% on what you call the midday meal, 87% on subway, 86% on the long sandwich, and worst where variants are nationally interspersed — 41% on trash versus garbage can, 46% on frosting versus icing. The documented isoglosses reproduce in both: *hoagie* in Philadelphia at 33% and 53%, *pop* in Buffalo at 53% and 63%, *frappe* in Boston at 37% and 55%, *bubbler* in eastern Massachusetts at 29% and 39%. *Yinz* cannot be checked this way at all, because Cambridge's version of the address-a-group question simply does not offer it.

The check that matters for the recommendation is narrower: are the twelve questions the quiz actually asks corroborated, or only the ones that happened to survive into the high-confidence tier? All twenty deployed questions have a Cambridge twin, thirteen of them high-confidence, and **every one of the first twelve beats chance**, by a mean of **+16.0 points** — from +4 on the long sandwich, where both surveys agree the country says *sub* almost everywhere and there is little left to agree about, to +37 on soda/pop/coke, the question the ordering opens with. The weakest link in the deployed set is q110, the night before Halloween, which sits sixth in the ordering: the two surveys agree on the modal answer in only 39% of cells against a 30% baseline, the cell-level correlation is 0.15, and it is mapped on 27,175 cells rather than the ~43,000 typical of the rest. It clears chance, but it is the one deployed question whose corroboration would not survive a stricter threshold. The reason is structural rather than a defect in recovery. Seventy percent of the country answers *I have no word for this*; the signal lives entirely in two small pockets, *devil's night* around Detroit and *mischief night* around Philadelphia and north Jersey, and outside them there is nothing for two surveys to agree about. That is also precisely why the greedy selection wants it sixth — a question that is silent nationally and loud in two specific places is exactly what a locator needs, and it is the same property that makes it hard to corroborate.

Extending the comparison to the 103 medium-confidence pairs adds 14.9 million cell comparisons and splits cleanly by question type, since Harvard's numbering is contiguous — q1 through q48 pronunciation, q49 through q57 syntax, q58 through q122 lexical. The margin over chance is +9 points on lexical items, +9 on phonetic and +8 on syntactic, with pooled correlations of 0.632, 0.560 and 0.647. That uniformity is the useful part. Had the corroboration lived only in the lexical items, the natural reading would be that the pixel recovery works for sharp word maps and not for gradient pronunciation maps. It does not: the margin is flat across classes, and the surfaces can be defended as a whole rather than only where they are easy.

What this does and does not establish. Both surfaces come from the same pixel-recovery method, so a systematic bias in that method would be shared and invisible here. What it does rule out is that the recovery is mostly noise, since two independent recoveries of two independently recruited populations would then agree at 54%, not 72%. It corroborates the surfaces at the surface level. It says nothing about how well the model locates an individual, because the Cambridge maps are aggregates too — they give dot locations, not per-person answer vectors. The wall described below is not breached by it.

**Measured but narrow.** ρ = 0.177, from 349 people on two constructions. It is the more trustworthy of the model's two fitted numbers because a within-person correlation does not care who was sampled. What it is not is a licence to temper the likelihood, which is a separate claim that failed on its own terms.

**Stress-tested, not validated.** The accuracy table at the top. The respondents are still simulated and the surfaces are still the model's own. What changed is that they now violate the model in three calibrated ways rather than obeying it perfectly, so the table is no longer circular in the way the earlier self-consistency runs were: it is an answer to "how badly can this be wrong and still work", which is a real question, rather than to "how well does the model do against people invented to agree with it", which is not. The magnitudes of the three violations are pinned to measurements — ρ from YGDP, surface error from the Cambridge comparison — but their shapes are assumptions, and no simulation can tell you the shape of your own blind spot. It still does not measure how well the model locates a human being.

**Trained on the simulator, and bounded by it.** The discriminative model in `model/neural.py` is the least externally grounded thing in this project, and the ordering here is deliberate: it sits below the stress test rather than beside it. Every person it has ever seen was generated by `model/idiolect.py` from the model's own surfaces. It cannot discover that a surface is wrong, cannot learn a dialect feature the Harvard survey did not ask about, and cannot correct a bias shared by the pixel recovery. What it *can* do — and what the measurements show it does — is read the same surfaces without assuming answers are conditionally independent, which is the assumption known to be false. Its advantage over Bayes is therefore an advantage at inference, not at evidence. Two specific weaknesses are worth carrying: it has never been trained against wrong surfaces, so it has had no opportunity to learn the caution `TAU_BASE` encodes, and its question ordering is inherited from the model it outperforms. Neither its accuracy numbers nor its confidence claims have touched a real person.

 YGDP is the only public source with real people, known raised locations and dialect answers. It overlaps the Harvard questions on exactly two distinct constructions, and two constructions cannot locate anyone: the model beats the population prior by 0.061 nats and 25 km on 1,450 of its respondents. Worse, the population prior *alone* covers YGDP respondents 63%, 88% and 97% of the time at nominal 50%, 80% and 95%. An online syntax survey is not a population sample; its respondents sit in large metros far more often than the population they are drawn from, and that skew is larger than the effect being measured.

What survives the skew is the change rather than the level. Adding answers with no discount moves coverage three to five points *below* where the prior alone sits on the same people, meaning the answers are making the posterior too narrow. Setting the base discount to 0.55 returns coverage to the prior's own level. On a biased sample that is the right target, because a calibrated model should inherit exactly the offset its prior inherits. That differential argument, not the absolute coverage, is the entire basis for 0.55 — and it is worth noting that the simulation later arrived at the same value from a completely unrelated direction, by asking which base minimises log score against people with wrong surfaces. Two arguments with no shared assumptions, one number.

Stratified calibration on the same set shows real confounding, exactly as a referee predicted before it was run: median error 1,099 km for White respondents, 1,207 for Black, 1,770 for Asian and 2,829 for Hispanic, with the probability-integral transform at 0.430 and 0.257 for the first and third. The model has no way to represent age, race or class, and it shows.

**Instrument controls.** The calibrator was run against a population simulated to obey the model exactly, where the correct answer is known to be no correction at all. It recovered exactly that: discount 1.0, contamination 0.0, coverage 0.510 / 0.800 / 0.955 against nominal 0.50 / 0.80 / 0.95, and a PIT mean of 0.497. Under the k-dependent parameterisation it pushes the base to the top of its grid, which is the same answer differently expressed. A procedure that manufactured corrections out of nothing would have failed here.

## Two objections that were tested and closed

A referee raised six concerns. Two were empirical claims about the measurement pipeline and both were settled by measurement rather than argument.

The saturation window is 114 km wide, far wider than a dot, and should blunt sharp features. Swept from 1 to 17 cells against the external county target, the log-loss is flat from 5 to 17 and *worse* at 1. The recommended narrow window is the bad one, because it saturates on a couple of dots, which costs more than the blunting it avoids. The deployed setting is within 0.0006 nats of optimal.

One global bandwidth cannot suit both a point feature like *yinz* and a national gradient like *you guys*. Spatial block cross-validation over 122 questions, twelve bandwidths, three block sizes and three pipeline variants confirms the sharpness is real — measured per choice, *yinz*, *yous*, *bubbler*, *grinder* and *hoagie* come out sharp and *you guys*, *soda*, *y'all*, *sub* and *coke* come out diffuse, all as predicted before fitting — but fitting a bandwidth per question buys at most 0.077% of the log-loss, thirteen times below the threshold set in advance. Both objections are correct in principle and negligible in practice.

One caveat on that second test, since it matters for reading its output. Block cross-validation scores prediction across a held-out void, which structurally rewards maximal smoothing, and its per-question optima pile up at the top of whatever sweep it is given. It cannot set the bandwidth's level. It can only settle whether per-question beats global, which it does.

## A bug in the answer, not the model

Worth recording because it was invisible in every aggregate metric. The model reports a named place, and place ranking scored only the grid cells that contained a town above the population floor. Those cells hold **16% of the posterior**, and the discarded 84% is not discarded evenly: it vanishes wherever people live in places too small to list.

Pittsburgh's metro population sits in dozens of sub-25,000 boroughs while New York's sits in large incorporated cities. So the model would report New York while its own modal cell sat 3 km from Pittsburgh. Every distance and coverage number was fine, because they are computed on cells; only the sentence the model actually says was wrong.

Assigning every cell to its nearest named place fixes it, and the ranking becomes a partition of the whole posterior. Towns are then agglomerated into metro areas within 60 km, since municipal boundaries are not dialect boundaries and a player from Pittsburgh hears "Bethel Park" as a miss. Clusters are ranked by summed probability and named for their most populous member.

The result on single answers, which is the useful sanity check:

| answers | top state | top metros |
|---|---|---|
| *yinz* | PA 20.6% | Pittsburgh 8.6%, New York 3.7% |
| *yinz, pop, water fountain, hoagie* | PA 33.0% | Pittsburgh 15.0%, Altoona 4.2% |
| *y'all, coke* | TX 20.1% | Houston 3.3%, Dallas 2.6% |
| *soda, you guys, sub* | CA 16.4% | New York 4.5%, Los Angeles 3.3% |
| *bubbler, pop, you guys* | WI 12.6% | Milwaukee, Worcester |

*Yinz* alone puts the mode 3 km from Pittsburgh, from a single answer. The last row is the pleasing one: *bubbler* has two homes, eastern Wisconsin and southern New England, and the model names both without being told they are related.

## A second model, and what it says about the number twelve

Everything above is one model — Bayes over 50,888 cells, with the likelihood factorised across questions. That model is *misspecified*, and the report has already established three of the ways. It assumes answers are conditionally independent given location, which the measured ρ = 0.177 says is false. The design-effect discount introduced to repair that is the wrong tool, as the previous sections show at length. And it cannot represent a mover at all: someone raised elsewhere produces a posterior that is narrow and confidently wrong, while the only dial available — a scalar temper — widens everything uniformly and cannot tell "unsure" apart from "sure and misled".

Those are not tuning problems. They are consequences of the factorisation. So the obvious question is what a model that never factorises would do with the same surfaces.

**Why this is not circular, and where it is.** Training a discriminative model on data simulated by the generative model would be pointless if the generative model were correct — the student could only recover the teacher. It is not pointless here precisely *because* the teacher is known to be wrong in three specified ways, and the simulator contains all three. `model/idiolect.py` draws people with a real idiolect, real mobility and optional surface error; Bayes must then read those people through an independence assumption it does not satisfy. A network reading the same answers has no factorisation to violate and no discount to undo. It can, in principle, learn to recognise a mover and widen appropriately rather than concentrating. **The ceiling is exact and worth stating plainly: it cannot learn anything the simulator does not contain.** It is a better reader of these surfaces, not a source of new information about American English.

**The reason it must be simulated at all.** There is no alternative, and this was checked rather than assumed. The Harvard survey survives only as aggregates — 30,197 rows of `state, question, choice, percent`. The Cambridge re-run is aggregate-only too. YGDP has 1,450 respondents with usable locations, but `data/ygdp/answers.csv` contains exactly five distinct questions, and the crosswalk notes that four of them (q54–57) are one construction. That is roughly two independent bits per person. **No public dataset pairs an answer vector with a hometown.** That absence is the binding constraint on this entire project, and it is why the model is trained against a simulator whose damage is calibrated to measurements.

**The model** (`model/neural.py`). Input is 802 bits: 680 for the chosen option of every answered question, plus a 122-bit mask saying which were asked, so one network serves any number of questions and any selection policy. Trunk is a 1024-wide linear layer, three residual blocks (`LayerNorm → Linear → GELU → Linear`), and a final projection — **8,177,664 parameters**. Output is a softmax over 1024 population-weighted k-means centroids rather than over all 50,888 cells, which keeps the output layer tractable; the quantisation costs a median of 39 km and a p90 of 82 km, comfortably inside the 200–500 km error scale being measured. Targets are not the centroid index but a distance-decayed distribution, `softmax(−d²/2σ²)` with σ = 100 km, so the loss rewards being *near* the right place instead of treating a neighbouring cluster as no better than the far side of the country. Question subsets during training are drawn by Gumbel top-k over question informativeness with a randomised sharpness, so the network sees everything from uniformly random subsets to the greedy ordering. For scoring, cluster probabilities are splatted back over all 50,888 cells in proportion to the prior, so both models are evaluated by identical code on identical support.

**An error worth recording, again.** The first training run overfitted: validation loss bottomed at epoch 36 and rose steadily for the next 44 epochs while training loss kept falling, and the saved checkpoint was from the worst part of that curve. This falsified something stated confidently at the time — that generated data makes overfitting impossible. It does not, if the pool is generated *once*. 400,000 people is a finite sample no matter where it came from, and the network memorised it. Redrawing the pool every epoch costs about 3 seconds against a 12-second epoch and removes the effect entirely: training and validation loss now track each other to within 0.01 nats, and validation was still improving at epoch 78 of 80.

**The result, and the answer to the question.** 2,000 held-out simulated people, ρ = 0.180, 15% movers, **no surface error**, all three models scored on the identical people and the identical question ordering. Median great-circle error:

| questions | discriminative net | Bayes, ρ = 0 | Bayes, ρ = 0.177 (deployed) |
|---|---|---|---|
| 1 | 1193 | 1123 | 1123 |
| 3 | 871 | 942 | 1058 |
| 5 | 654 | 685 | 897 |
| 8 | 514 | 567 | 842 |
| **12** | **343** | **444** | **760** |
| 14 | 281 | 384 | 789 |
| 16 | 237 | 330 | 794 |
| 20 | **199** | 264 | 847 |

**Twelve is not a fact about dialects. It is an artefact of the discount.** The deployed model bottoms at eleven-to-twelve questions and then actively degrades, losing 88 km between twelve and twenty; its marginal value per question is −8.4 km over 13–17 and −14.5 km over 17–20. Twelve is where that model stops, and it stops because the temper has finally outrun the evidence. Both correctly-specified models are still descending at twenty with no turnaround: the net gains 26.5 km per question over 13–17, and ρ = 0 Bayes is still gaining 16.6 km per question at the cap.

**Model choice dominates question count, by a wide margin.** At the same twelve questions the network more than halves the deployed error, 760 km to 343. The sharper comparison is that the network with **five** questions (654 km) beats the deployed model with *any* number of questions, whose best is 760 km. No quiz length rescues a model that has stopped learning, and no amount of extra asking substitutes for removing the discount.

**Where the returns actually stop.** For the network, 50% of the total achievable reduction arrives by five questions, 75% by ten, 90% by fourteen and 95% by sixteen. Priced explicitly — how many km must a question save to be worth asking — the optimum is seventeen questions at 10 km, seventeen at 20 km, **fourteen at 30 km**, and seven at 50 km.

**Four caveats, none of them small.** First, this population has **no surface error**, so these numbers are not comparable to the headline table above, which carried 20 points of it; that is why ρ = 0 Bayes reads 264 km here and 329 km there at twenty questions. The network has therefore never been trained against wrong surfaces, which is the exact failure `TAU_BASE` exists to absorb — this is the most important remaining gap in it. Second, the sweep stops at twenty because `question_order.csv` only has twenty rows, and both good models are still improving there, so the true optimum is beyond what was measured. Third, that ordering was derived greedily under *Bayes with the discount active*, so questions 17–20 are whatever a contaminated search ranked last; the network's apparent flattening over 17–20 (6.1 km per question) may be an artefact of a bad tail rather than real saturation. Fourth, the network wins on distance at every length past two but **loses on modal state at twenty**, 46.5% against 48.0%, which is the 1024-cluster quantisation blurring state boundaries that Bayes resolves at full cell resolution. It is better at *where*, slightly worse at *which state*.

## The recommendation

**Set `RHO` to 0.** This is a one-line change in `model/infer.py` and it is the largest single improvement available: three times the precision at twenty questions, from answers the model already had. It is not a close call — 253 of 253 matched comparisons, across correlations up to 0.30, with and without movers, at four levels of surface error. The measurement of ρ was sound; the design effect was the wrong tool to apply it with. **This change is recommended but not yet made**, because it alters a deployed constant.

**Keep `TAU_BASE` at 0.55.** It is the correction that works, and the experiment confirms it by a route entirely independent of how it was fitted. Untempered, the model says 50% and is right 40% of the time; at 0.55 it says 50% and is right 57%, says 80% and is right 81%. Its optimum tracks surface error in the direction theory demands, and lands on 0.55 exactly in the error band the Cambridge comparison independently measured.

**Ask fourteen, not twelve — and understand why twelve looked right.** With the discount off, error falls at essentially every question, in every world tested including the worst corner of the design. "Ask twelve" was never a fact about dialects, and the second model makes the mechanism unmistakable: the deployed configuration bottoms out at eleven-to-twelve questions and then *gets worse*, shedding 88 km between twelve and twenty, because the schedule discounts itself faster than the questions can inform it. Twelve is the correct stopping point for a broken model. For a correctly specified one, 90% of the achievable reduction arrives by fourteen questions and the priced optimum at a plausible boredom cost of 30 km per question is also fourteen; both models are still improving at the twenty-question measurement cap. Fourteen is the defensible number today, and it is provisional until the ordering is re-derived without the discount and extended past twenty.

**Consider replacing the model, not just the constant.** `model/neural.py` reads the same surfaces without factorising the likelihood, and on stress-tested people it halves the deployed error at equal quiz length — 343 km against 760 at twelve questions — and beats the deployed model's *best-ever* accuracy using five questions instead of twelve. It is not ready to ship: it has never been trained against wrong surfaces, its question ordering is inherited from the model it replaces, and it trades a little modal-state accuracy for its distance advantage. But the gap is large enough that closing those three is worth more than any further tuning of the Bayes path.

**You may now quote a confidence number, with one caveat.** Across every length and every misspecification constructed, the credible regions cover within four points of nominal, and usually within three. That is a real change from the previous position. The caveat is that this is still a simulation: the prior and the surfaces are the model's own, deliberately damaged in three ways whose *magnitudes* are calibrated against measurements but whose *shapes* are assumptions. Say "eighty percent" and mean it; do not say "eighty-three".

**Close the loop by playing.** `model/quiz.py` asks adaptively and records every game with the player's real hometown, and `model/export.py` turns that log into a validation set the calibrator already knows how to read. Fifty games with known truth would still be worth more than everything above, because it is the only evidence in the project that would come from real people answering the model's own questions.

## Reproducing

```
model/bandwidth.py                        bandwidth cross-validation
model/choose.py --questions 20            regenerate the ordering
model/nullcheck.py --n 1200 --questions … a population that obeys the model
model/validate.py --set selfconsistency   accuracy on people who obey it
model/validate.py --set selfconsistency --rho 0     ... with the discount off
model/validate.py --set selfconsistency --tau 1.0   ... with all tempering off

model/idiolect.py --n 1200 --true-rho 0.18 --mover 0.15 --surface-mae 20 \
    --deployed-rho 0,0.177 --ks 1,2,3,4,5,6,8,10,12,14,16,18,20 \
    --out headline                                       the headline tables
model/idiolect.py --n 1200 --true-rho 0.18 --mover 0.15 --surface-mae 20 \
    --deployed-rho 0,0.177 --ks 5,12,20 --base 0.4,0.55,0.75,1.0 \
    --out surf20                                         the TAU_BASE sweep
model/idiolect.py --n 1200 --true-rho 0,0.05,0.10,0.18,0.30 --mover 0.15 \
    --out idiolect_mover                                 the 5x5x5 surface
model/idiolect.py --n 1200 --true-rho 0.30 --mover 0.15 --surface-mae 30 \
    --deployed-rho 0,0.177 --ks 5,8,12,16,20 \
    --out worstcorner                                    everything broken at once

model/calibrate.py --set ygdp --auto      refit the base discount
model/isogloss.py                         the 16-case regression
model/quiz.py                             play; model/export.py turns it into data

model/neural.py prep                      1024 clusters + a 400k training pool
model/neural.py train --epochs 80         train the discriminative model
model/neural.py curve --n 2000 --kmax 20  the three-model question curve
```

`neural.py` needs `torch`, which is the only heavy dependency in the project. `prep` takes about five minutes, almost all of it k-means; `train` runs about twelve seconds per epoch on an Apple M3 Pro with MPS, redrawing the training pool every epoch and keeping the best-validation checkpoint; `curve` takes about twenty-five minutes, dominated by the 80,000 Bayes posteriors it computes for the two comparison arms. Everything lands in `data/model/neural_*`.

Note that `curve` scores a population with movers and a real idiolect but **no surface error**, while the headline table above carries 20 points of it. The two tables are not comparable, and the difference is the whole reason `TAU_BASE` exists.

Re-run the `surf20` line at `--surface-mae 0`, `10` and `30` for the surface-error axis, and the `idiolect_mover` line at `--mover 0` for the no-mobility half. Everything lands in `data/model/*_surface.csv`. A full 5×5×5 run is about forty minutes; four fit comfortably in parallel on a laptop.

Three switches that are easy to confuse. `--rho` changes the correlation discount the model deploys and leaves `TAU_BASE` alone. `--tau` is a flat override that removes both, and so additionally asserts the recovered surfaces are exactly right. `--true-rho` is not a model setting at all: it changes the *people*, bisecting the idiolect strength until the correlation they actually exhibit matches the target. Confusing the first two produced a mislabelled column in an earlier draft of this report.
