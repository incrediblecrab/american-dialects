# UWM Dialect Survey — Data Inventory

## Source

**Survey:** Vaux & Samuels (2004–2006), published online as the UWM Dialect Survey, subsequently cleaned and republished by Mørten Mørch Jøhndal starting 09/25/2018 at <https://dialectsurvey.wordpress.com>.

**Why this survey matters as a control arm:** The Harvard Dialect Survey (HDS) ran 2002–2003, and the Cambridge survey is undated (began ≈2006 with open-ended accumulation). Comparing Cambridge to HDS therefore confounds *time* with *population* and *method*. The UWM survey was conducted in precisely the 2004–2006 window, only 1–3 years after HDS, an interval over which real lexical change should be near zero. That makes UWM-vs-HDS a near-zero-time control arm, isolating method-plus-population noise so that Cambridge-vs-HDS can be read against that baseline rather than against zero.

---

## What was scraped

The sitemap at `https://dialectsurvey.wordpress.com/sitemap.xml` returned 158 URLs. Of those, 148 carried a `/qN-` slug (question post), 5 were blog posts whose Q-numbers appeared only inside the post's H1 heading, and 5 were static pages (about, survey-questions, etc.).

**Scraping outputs (`scrape/uwm.py`, cache at `data/raw/uwm/`):**
- **154 UWM questions** parsed and indexed in `data/uwm/questions.csv`.
- **700 heatmap PNG images** downloaded to `data/raw/uwm/images/`.
- Post dates span 09/25/2018 – 11/14/2020.
- The full UWM survey contains 544 questions; only 154 were published as maps on the blog (390 remain unpublished).
- One post (`/q94-97-usage-of-positive-anymore/`) covers four questions (q94–q97) in a single page; each has separate images.

**Image format split:**
- 68 questions use the **old image format** (2018 batch): filenames `heatmap-{Q}-{N}-{answer-label}.png`; the answer-choice label is embedded in both the filename and the HTML `data-image-title` attribute.
- 86 questions use the **new image format** (2019–2020 batch): filenames `heatmap.{Q}.{N}.png`; no answer-choice label anywhere in the filename or post HTML.

---

## HDS crosswalk (`data/uwm/hds_crosswalk.csv`)

Matching method: `difflib.SequenceMatcher` on normalised text (lowercase, punctuation stripped, stopwords removed), augmented by a containment bonus (score floor 0.82) when all normalised HDS tokens appear in the normalised UWM text. This handles the many HDS entries that are single-word keyword descriptions (e.g., HDS q4 "caramel" → UWM q102 "How do you pronounce caramel?").

**Counts at each threshold (after removing 8 confirmed false positives):**

| Threshold | Raw count | Valid after hand-verification | Precision |
|-----------|-----------|-------------------------------|-----------|
| ≥ 0.99 (verbatim) | 17 | 17 | 100% |
| ≥ 0.95 | 17 | 17 | 100% |
| ≥ 0.85 | 18 | 18 | 100% |
| ≥ 0.70 | 36 | 28 | 78% |

The 0.85–0.95 band adds one entry: HDS q5 "the vowel in the second syllable of 'cauliflower'" ↔ UWM q153, ratio=0.868, confirmed TRUE.

The 0.70–0.85 band contains 18 entries; 8 were hand-verified as false positives, all caused by the containment algorithm firing when a short HDS keyword appears incidentally in an unrelated UWM question:

- HDS q22 "poem" → UWM q123: "poem" appears in an example sentence ("I used to could recite the entire poem from memory"); UWM q123 is about the construction "used to could", not the word "poem".
- HDS q23 "really" → UWM q343: "really" appears as an intensifier adverb; UWM q343 asks about a rubber ball toy.
- HDS q36 "the c in grocery" → UWM q346: "grocery" is shared but UWM q346 asks about the functional difference between a supermarket and a grocery store, not pronunciation.
- HDS q40 "quarter" → UWM q391: "quarter" appears in context of a fitness activity; the questions are about unrelated topics.
- HDS q95 "What is the City?" → UWM q336: "city" appears in a question about passageways between buildings; different referents.
- HDS q97 "Which of these terms do you prefer?" → UWM q436: generic phrase overlap; unrelated questions.
- HDS q98 "Which of these terms do you prefer?" → UWM q436: same generic phrase; unrelated questions.
- HDS q28 "Do you pronounce cot and caught the same?" → UWM q25 "Do you pronounce which and witch the same?": structural template match but different phoneme contrasts tested.

**Coverage limitation:** Many HDS questions have no published UWM map counterpart. For example, HDS q64 (long sandwich), q105 (carbonated beverage), q73 (sneakers/gym shoes), and q105 (soda/pop) all correspond to UWM questions that were never published on the blog. The 94 HDS questions with no valid match at ≥0.70 are mostly in this category, not text-mismatch failures.

---

## Colour scale and image analysis

Each published map is a **pre-smoothed diverging heatmap**, not a dot plot or a discrete proportions-by-state choropleth. The colour scale runs from warm red (high proportion choosing a given answer) through white (midpoint) to cool blue (low proportion). The best-fit standard colormap is matplotlib `RdYlBu_r` (reverse Red-Yellow-Blue diverging).

**Legend status:** There is no colour legend inside any image, and no legend metadata in any post's HTML. The mapping from pixel colour to proportion value is therefore unknown without external calibration data.

**Image dimensions and borders:**
- Old format (2018): 493×317 px, map fills the entire image extent.
- New format (2019–2020): 640×480 px with white borders; map region at approximately x=79–577, y=64–436 (499×373 px).

**Projection:** The map projection is not plate-carree. The aspect ratio of the map region (499×373 ≈ 1.34:1) is consistent with Albers Equal Area Conic, the standard projection for CONUS reference maps. The HDS geographic pipeline uses plate-carree (456×200 pixels). Geographic extraction therefore requires either re-projection or an approximation (resize to HDS grid dimensions), which introduces spatial noise especially near the map borders.

---

## Validation: per-state signal vs HDS state percentages

To assess whether the colour signal carries usable geographic information, I resized each UWM image (cropping white borders for newer format) to the HDS plate-carree grid (456×200 px) and extracted the mean R−B (red minus blue) channel value for each state using the existing `data/model/state_raster.npy` grid. I then correlated those per-state means against `data/hds/state_pct.csv` for 13 verbatim-matched question pairs.

For questions with multiple images (one per answer choice), `r_best` records the highest correlation found across all images for that question (an upper bound on what is knowable if the correct image were identified); `r_img0` records the correlation for the first image in filename-sort order (closer to what a naive pipeline would use without answer-choice labels).

| UWM q | HDS q | Label | r_best | r_img0 | MAE | # images |
|-------|-------|-------|-------:|-------:|----:|---------:|
| 140 | 80 | sunshower | 0.814 | 0.814 | 0.133 | 8 |
| 264 | 61 | boulevard / median | 0.738 | 0.738 | 0.193 | 8 |
| 334 | 118 | brew-thru | 0.703 | 0.703 | 0.182 | 8 |
| 40 | 117 | basement | 0.661 | 0.661 | 0.255 | 6 |
| 229 | 52 | where-are-you-at | 0.615 | 0.615 | 0.359 | 3 |
| 397 | 87 | bear claw | 0.577 | 0.577 | 0.223 | 3 |
| 138 | 103 | drinking fountain | 0.576 | 0.284 | 0.240 | 5 |
| 92 | 76 | across both streets | 0.535 | 0.535 | 0.224 | 9 |
| 142 | 110 | mischief night | 0.534 | 0.183 | 0.258 | 10 |
| 333 | 102 | crane fly | 0.411 | 0.249 | 0.243 | 18 |
| 380 | 111 | end of bread | 0.404 | 0.285 | 0.223 | 7 |
| 141 | 106 | TP-the-house | 0.239 | 0.239 | 0.250 | 8 |
| 330 | 119 | take-out | 0.102 | −0.060 | 0.376 | 5 |

**Summary statistics:** Best-image r: mean = 0.531, median = 0.576, range [0.10, 0.81]. First-image r: mean = 0.448, median = 0.535, range [−0.06, 0.81].

**Notes on specific pairs:**
- For UWM q330 (take-out), images 2 and 4 correlate at r=−0.730 and r=−0.715 with HDS q119 choice 'a' (take-out). Negative correlation at that magnitude means those images are showing a competing answer — almost certainly "carry-out", the Midwestern term, which is geographically complementary to "take-out". This confirms the geographic signal is real; the difficulty is that without answer labels we cannot identify which image is which.
- For UWM q138 (drinking fountain), best_r=0.576 against HDS choice 'a' (bubbler, mean 6.7% nationally with a strong Wisconsin/Rhode Island peak), but img0_r=0.284. The large gap confirms image-choice ambiguity for this newer-format question.
- For UWM q142 (mischief night), best_r=0.534 but img0_r=0.183. Same ambiguity problem.
- The MAE figures (0.13–0.38) are computed by normalising the UWM R−B signal to [0, 1] before comparing to HDS proportions. Even at r=0.814 (sunshower), the absolute proportion errors reach 0.133 across states. Without calibration data the proportions themselves cannot be recovered.

---

## Verdict

**The UWM heatmap images are not usable as a geographic data source for this project in their current form.**

The geographic signal is real — Pearson r up to 0.814 for sunshower, r ≥ 0.70 for three others — but four independent problems prevent pipeline integration:

1. **No colour legend.** There is no way to convert pixel colour to a proportion value without external calibration that does not exist. The images show somebody else's interpolated surface, not the underlying per-respondent data. Any number extracted would represent the original authors' smoothing choices, not the survey responses directly.

2. **Unknown image-to-choice mapping for 86 of 154 published questions.** The newer-format posts (2019–2020) use unlabelled filenames. For questions with multiple images (typically 4–18 per question), it is not possible to know which image corresponds to which answer choice without either (a) external documentation from the blog author, or (b) brute-force cross-correlation against a known ground truth for every image — which requires exactly the clean data this pipeline would be intended to provide.

3. **Projection mismatch.** The UWM maps are likely Albers Equal Area Conic; the HDS geographic pipeline uses plate-carree. Resizing to the HDS grid without re-projection introduces systematic geographic distortion. State-level averages are approximately correct (the validation above uses this approximation), but any finer geographic inference would require proper GCP registration.

4. **Weak to moderate mean correlation.** Best-image mean r = 0.531 across 13 pairs, first-image mean r = 0.448. The HDS-pipeline standard for state-level inference is much closer to r ≥ 0.90. Questions like TP-the-house (r=0.239) and take-out (r=0.102 for best-known image) are too noisy to be informative.

**A clear stopping point:** If the blog author made the underlying tabular data available — or if the old-format image answer labels could be mapped to UWM answer choices — the 68 older-format questions would be worth re-evaluating, particularly for the 17–28 verbatim crosswalk matches. The 4 questions with r ≥ 0.70 (sunshower, boulevard/median, brew-thru, basement) carry a usable ordinal geographic signal if the correct image per choice were identified. But none of this applies to the data as currently available, and building a recovery pipeline on top of an unvalidated colour-to-proportion mapping would produce numbers inconsistent with this project's commitment to not shipping guesses.

---

## Files

| Path | Description |
|------|-------------|
| `data/uwm/questions.csv` | 154 rows: `question, slug, url, text, image_urls, n_images` |
| `data/uwm/hds_crosswalk.csv` | 121 rows: `uwm_question, hds_question, ratio, confidence, method, uwm_text, hds_text`; `confidence` ∈ {verbatim, high, medium, low, reject} |
| `data/raw/uwm/*.html` | Cached post HTML for all 153 scraped URLs |
| `data/raw/uwm/images/*.png` | 700 heatmap images |
| `scrape/uwm.py` | Scraper: sitemap → posts → images → questions.csv |
| `scrape/uwm_crosswalk.py` | Crosswalk builder: HDS × UWM text matching → hds_crosswalk.csv |
