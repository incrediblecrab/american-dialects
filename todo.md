# todo

What is left to do. The evidence for each of these lives in [findings.md](findings.md) and is not restated here — this file says what to change, not what was measured.

## done

- **`RHO = 0`, the ordering re-derived, `N_QUESTIONS = 14`** — landed as one change, because any one of the three alone makes the model worse. `data/model/question_order.csv` now holds thirty questions searched with the discount off, `data/model/neural_curve.csv` runs to thirty, and every "provisional" that traced back to that file has been resolved or restated. Two things worth knowing about the outcome: the k = 20 figures came back identical to the kilometre on the new ordering, which is the best available evidence that nothing else moved; and fourteen survived, but the argument for it did not — the two stopping rules that used to agree on fourteen now bracket it, and `findings.md` says so.

- **The int8 quantisation penalty is measured, and it is free.** `model/export_web.py --measure` replays simulated games against full-precision and byte-quantised surfaces; they agree on the answer the large majority of the time, and every disagreement was a near-tie rather than an error. The payload can ship as bytes. The numbers are in [findings.md](findings.md) and `check.py` holds the prose to the artefact the site was built from.

## still capped, just further out

- **The sweep now stops at thirty rather than twenty, and both correct models are still improving there.** The cap moved; it did not go away. Extending `model/choose.py` again would refine `N_QUESTIONS`, but the return is small and the quiz-length decision is now dominated by how long a stranger will sit still rather than by accuracy.

## closed, negative

- **The network does not recognise a mover, and in this simulator it cannot.** The claim was that a model seeing the whole answer vector could widen for somebody raised elsewhere instead of concentrating confidently. It does not: the ratio of its 80% area on movers to the same area on non-movers is 1.01, its width carries no mover signal at all, and it degrades on movers slightly worse than the Bayesian model does. The reason is structural rather than a training failure — `model/idiolect.py` draws a mover's speech cell from the same prior it draws `home` from, so mover status is independent of the answers by construction, confirmed by a probe at 0.4993 against a positive control at 0.8500 on identical inputs. The auxiliary head was not built because its target is provably unpredictable from its input. Real movers plausibly blend two places and would be detectable; simulating that needs a per-question blend fraction and nothing in this project can calibrate one. Details in [findings.md](findings.md).

- **Ranking questions by their geography-to-idiolect ratio does not beat ranking them by information.** The split was measured per question and it is a good measurement — it puts *the City*, TP-ing and *pop/soda* at the top and the "s" in chromosome at the bottom, and it correctly separates the three pronunciation questions that are real isoglosses from the rest of the phonetics. But the ordering it produces is worse, by 64 km at the deployed quiz length, and the reason is that it correlates with mutual information at 0.945 and so buys cleanliness by giving up bits the model still needs. `model/signal_split.py` and `model/order_compare.py` are kept because the measurement is reusable; nothing is deployed from them. Details in [findings.md](findings.md).

- **The UWM Dialect Survey maps cannot be recovered, and the attempt is not worth repeating.** `scrape/uwm.py` pulled all 154 published questions and 700 heatmap images; `scrape/uwm_crosswalk.py` matched them against the Harvard question list; `data/uwm/inventory.md` records what was found. The artefacts are tracked so that the next person reads this instead of redoing it.

  Two blockers are structural and either one is fatal on its own. **There is no colour legend** — anywhere, on any image or in any post — so a pixel's colour cannot be turned into a proportion. The Harvard maps were recoverable because a dot is a countable object with a known meaning; a smooth diverging surface with no scale is somebody else's interpolation at an unknown bandwidth, read against an unknown range. And **86 of the 154 questions carry no answer-choice label**: the 2019–2020 posts name their images `heatmap.{Q}.{N}.png`, and neither the filename nor the post HTML says which answer choice image N shows. A surface whose answer is unknown is not evidence about anything. Two further problems compound these: the projection is conic rather than the plate carrée the project's grid uses, and the recovered surfaces correlate with published Harvard state percentages at a mean r near 0.53 against the 0.955 the Harvard recovery achieves — though that figure assumes the very colormap that cannot be verified, so it should be read as indicative rather than as a measurement.

  **One planning estimate was badly wrong and is worth recording as such.** The overlap with Harvard was scoped at "at least 79 questions, twelve verbatim", described as a floor depressed by WordPress truncating slugs. Matching on full question text rather than slugs, then hand-checking every candidate, gives **17 verbatim and 28 valid at a 0.70 threshold**, with 8 false positives removed from that band. The "floor" was an overestimate by a factor of nearly three, because slug matching had been scoring partial-word overlaps as hits. The lesson is the ordinary one: a number produced by fuzzy matching is a hypothesis until somebody reads the pairs.

  **What would reopen it.** Answer-choice labels for the 2019–2020 batch, or the underlying tabular data, from the blog's author. With those, the 17 verbatim pairs would be worth re-testing. Without them, no amount of image processing helps, because the missing information is not in the images.

  This leaves the project without the control arm it wanted. Cambridge-vs-Harvard still confounds time with population and method, and there is no near-zero-time comparison to read it against.

## the measurement the project still lacks

- **Fifty logged games.** `data/quiz/log.csv` is header-only. Until it is not, "80% confident" is an extrapolation from simulation, and it is the one claim the report cannot make. `model/quiz.py` and `site/server.py` both already write the format `calibrate.py --set quiz` reads, so this is a distribution problem rather than an engineering one.

## research worth doing

- **Harvard vs. Cambridge as a longitudinal panel, against the phonology.** The Cambridge survey re-ran all 122 Harvard questions verbatim, years later, on a different population, and both geographies are already recovered onto the same grid. `cambridge_validate.py` currently treats the second survey only as agreement evidence for the first, which throws away the interesting half. The literature sharpens the question: the big regional *phonological* shifts are documented as retreating fast — the Southern Vowel Shift has a Gen X cliff (Renwick et al. 2023), the Northern Cities Shift is in retreat across Michigan and upstate New York, both being absorbed into the continental Low-Back-Merger Shift — while the *lexical* variation these two surveys actually measure is reported as stable. So: does lexical variation level at the same rate as phonological variation? If these surfaces show lexical stability over an interval in which vowels demonstrably moved, that is a real result, and this may be the only data that can show it below the state. Look at Buffalo, Chicago and Detroit first: NCS retreat is centred exactly there, and they are also where the strongest within-state lexical contrasts in this project sit, so it is where the two surveys are most likely to genuinely diverge.

## leads worth an hour

- `osf.io/56umh` — possible rehost of Grieve's Word Mapper county matrices. All four original Drive links were re-confirmed dead on August 29, 2026. The OSF page renders nothing to a plain fetch, so this needs a real browser before the source is abandoned.
- **DialectGuessr** (`dialectguessr.com`) is collecting answer vectors paired with player-declared hometowns and has released none of it. That is precisely the resource whose absence forces this project to simulate. Worth an email: the assets are complementary, not competing.
- **Bert Vaux** directly, for whether Cambridge respondent-level data exists. The Oslo mirror serves maps; the `survey.johndal.com` results pages return only a copyright notice. He is at Cambridge, and he ran a new Texas survey with the Houston Chronicle in 2026, so he is active.
- ~~`dialectsurvey.wordpress.com/category/all-maps/`~~ — scraped, and closed as a negative result. See below.

