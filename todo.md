# todo

What is left to do. The evidence for each of these lives in [findings.md](findings.md) and is not restated here — this file says what to change, not what was measured.

## blocked on a decision, not on work

- **Set `RHO = 0`** in `model/infer.py`. `findings.md` makes the case and it is not close. One line, but it changes deployed behaviour and it invalidates the current question ordering, so do it together with the two items below rather than on its own.
- **Re-derive the question ordering with the discount off**, and extend it past twenty. `model/choose.py` writes `data/model/question_order.csv`, which currently has twenty rows because that is where the sweep stopped, and the ordering it holds was searched under the discount it is now known to be contaminated by. Every "provisional" in the report traces back to this file.
- **Then raise `N_QUESTIONS`.** Fourteen is the current recommendation and it is only valid once the two items above are done: raising the count while `RHO` is still 0.177 makes the model worse, because that configuration degrades past twelve.

Do the three as one change. `check.py` will fail on the prose until the documents are updated to match, which is the intended behaviour.

## the measurement the project still lacks

- **Fifty logged games.** `data/quiz/log.csv` is header-only. Until it is not, "80% confident" is an extrapolation from simulation, and it is the one claim the report cannot make. `model/quiz.py` and `site/server.py` both already write the format `calibrate.py --set quiz` reads, so this is a distribution problem rather than an engineering one.

## research worth doing

- **Harvard vs. Cambridge as a longitudinal panel, against the phonology.** The Cambridge survey re-ran all 122 Harvard questions verbatim, years later, on a different population, and both geographies are already recovered onto the same grid. `cambridge_validate.py` currently treats the second survey only as agreement evidence for the first, which throws away the interesting half. The literature sharpens the question: the big regional *phonological* shifts are documented as retreating fast — the Southern Vowel Shift has a Gen X cliff (Renwick et al. 2023), the Northern Cities Shift is in retreat across Michigan and upstate New York, both being absorbed into the continental Low-Back-Merger Shift — while the *lexical* variation these two surveys actually measure is reported as stable. So: does lexical variation level at the same rate as phonological variation? If these surfaces show lexical stability over an interval in which vowels demonstrably moved, that is a real result, and this may be the only data that can show it below the state. Look at Buffalo, Chicago and Detroit first: NCS retreat is centred exactly there, and they are also where the strongest within-state lexical contrasts in this project sit, so it is where the two surveys are most likely to genuinely diverge.
- **Give the network a mover head.** The argument for a non-factorised model was partly that it *could* learn to recognise someone raised elsewhere and widen rather than concentrate. That was never tested. `model/idiolect.py` already labels who moved.
- **Separate the geographic questions from the personal ones.** The within-person correlation is measured pooled. Per question, some answers must carry far more idiolect than region, and ranking them by that ratio would change which questions are worth asking.
- **Measure the quantisation penalty before designing any browser port.** Requantising the surfaces to int8 is what decides whether the model can ship as a static payload. The arithmetic suggests the penalty is negligible against the signal, but that is a guess, and re-running `neural.py curve` on quantised surfaces turns it into a number.

## leads worth an hour

- `osf.io/56umh` — possible rehost of Grieve's Word Mapper county matrices. All four original Drive links were re-confirmed dead on August 29, 2026. The OSF page renders nothing to a plain fetch, so this needs a real browser before the source is abandoned.
- **DialectGuessr** (`dialectguessr.com`) is collecting answer vectors paired with player-declared hometowns and has released none of it. That is precisely the resource whose absence forces this project to simulate. Worth an email: the assets are complementary, not competing.
- **Bert Vaux** directly, for whether Cambridge respondent-level data exists. The Oslo mirror serves maps; the `survey.johndal.com` results pages return only a copyright notice. He is at Cambridge, and he ran a new Texas survey with the Houston Chronicle in 2026, so he is active.
- `dialectsurvey.wordpress.com/category/all-maps/` — Vaux's own blog, never scraped. Everything else in the sources list has been.

