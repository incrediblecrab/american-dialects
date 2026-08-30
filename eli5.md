# eli5

The plain-language version. [README.md](README.md) is the tour, [findings.md](findings.md) is the full report with all the numbers and arguments. This file is the story.

## what this is

You know the party trick where someone guesses where you grew up from how you talk? This is that, done with math.

You answer some questions — *soda, pop, or coke?*, *what do you call the night before Halloween?*, *y'all or you guys?* — and a program draws a map of where it thinks you were raised. Not a joke map. A real one, with an honest confidence number attached.

## why it works at all

The way you talk was mostly set before you were twelve. You didn't choose to say *sub* instead of *hoagie*; you absorbed it from whoever was around. So your vocabulary is a fingerprint of a place, and it's a stubborn one — people who moved at thirty usually still say the words they learned at eight.

The trick is that these words don't scatter randomly. They come in patches. *Hoagie* is basically a circle around Philadelphia. *Bubbler* means "water fountain" in exactly two places on Earth: eastern Wisconsin and Rhode Island. If you know the patches, and you know which words someone uses, you can intersect them.

## the part that shouldn't have been possible

Back in 2002, two Harvard linguists asked 30,788 people 122 questions like this. They wrote down each person's answers **and their ZIP code**.

Those ZIP codes were never made public. The New York Times built its famous 2013 dialect quiz on that private data, and the server that did the math has been switched off for years. What's left in public is state-level summaries — "34% of Pennsylvanians say *pop*" — which is far too coarse. Pennsylvania is not one dialect. Pittsburgh and Philadelphia disagree about nearly everything.

But the old survey website did put up a **picture** for every answer: a small map of the United States with a colored dot for each person who gave that answer, plotted at their ZIP code.

The data was thrown away. Photographs of the data survived.

So this project reads the photographs. It zooms into those images, finds every colored dot, works out exactly which latitude and longitude each pixel corresponds to, and reconstructs roughly where the respondents were. Three annoying problems had to be solved:

- **Where is this pixel?** Solved by finding borders whose real position is known exactly — the US/Canada line is at 49°N, the Colorado/Wyoming line at 41°N — and fitting the math until the map lines up. It lands within about 7 km, which is smaller than a single pixel.
- **The dots are blurry.** Images get smoothed at the edges, so a dot bleeds into its neighbors and comes out as a faded version of its color. But faded-toward-white is predictable, so you can undo it and figure out how much dot was really there.
- **Cities are a blob.** In New York, so many dots overlap that they merge into one solid mass, and you can't tell 500 people from 5,000. There's a standard statistical fix for "how many things are in there if they're piled on top of each other," and it gets density back out.

**Does it work?** Yes, and this is checkable, because the original survey *did* publish per-state percentages calculated from the real records. So: reconstruct the map from pixels, compute what percentage of each state it implies, and compare to the real published number. Across 28,513 comparisons the agreement is **r = 0.955** — very close. The pixels really do give the data back.

## how the guessing works

Chop the country into 50,888 little squares. Every square starts with a score based on how many people live there — before you say anything, you're probably from somewhere populated.

Then each answer you give multiplies those scores. Say *yinz* and every square near Pittsburgh gets multiplied way up while everywhere else gets multiplied way down. Say *bubbler* and Wisconsin and Rhode Island both light up. Normalize so everything adds to 100%, and that's your map.

That's it. That's the whole model. Each answer is a filter; stack the filters.

## the mistake, which is the best part of this project

Here's the problem with stacking filters. Suppose you say *y'all*, then *fixin' to*, then *coke*, then *crawfish*.

That looks like four pieces of evidence. It isn't. It's **one fact — "I'm Southern" — said four times.** If you treat it as four independent clues and multiply them all in, you become far more confident than you've earned. The model will announce a specific county when it has only really learned "the South."

This is a real problem, and it was measured properly: two answers from the same person really are correlated, by about 0.18, even after accounting for where they're from. That number is right.

The fix that was applied was: turn the volume down on *all* the evidence, and turn it down more the longer the quiz runs.

That fix is wrong, and it took a long time to see why. It's like noticing that one actor in a movie is too loud and responding by turning down the whole TV. You do fix the loud actor. You also make the dialogue you actually needed inaudible. Real repeated evidence gets quieted, but so does genuinely new evidence — and the questions had already been deliberately chosen *not* to repeat each other.

**The symptom was unmistakable once we looked for it: the model got worse the more you told it.** Past about thirteen questions, extra answers actively made the guess less accurate. A model that gets dumber when you give it more information is broken. That's not a subtle statistical concern; that's an alarm.

Turning that correction off entirely made the model roughly **three times more precise**. It was tested against simulated people deliberately built to break the model in three different ways at once, at many different severities: **253 comparisons, and turning it off won all 253.**

The lesson isn't "don't correct for things." The measurement that motivated the correction was completely correct. The correction was borrowed from a different problem — it's a formula for "I surveyed 10 people in each of 50 villages, how well do I know the national average?" — and this is not that problem. There's one person here, not a village.

## why "twelve questions" was wrong

The NYT quiz asked about twelve questions. This project inherited that number without ever checking it. Everyone just... kept using twelve.

So we finally measured it. Ask 1 question, 2, 3, up to 30, and see how the error actually falls.

It turns out **twelve was the right answer for the broken model, and only for the broken model.** The broken version stops improving at thirteen questions and then gets *worse* — by thirty questions it's 109 km worse than it was at its best. Of course it stops early. It's turning its own volume down faster than the questions can inform it.

Fix the model and the curve just keeps descending. At thirty questions — the furthest we can currently measure — it's still improving, with no flattening out.

Twelve was never a fact about how people talk. It was a fact about a bug.

The quiz now asks **fourteen**, and it's worth being straight about where that number comes from: it is a judgement, not a discovery. Because the fixed model never stops improving, there's no "best" length to find. Fourteen is what you get if you assume each extra question has to save you about 30 km to be worth the annoyance of answering it. If you instead ask where you've collected 90% of the available accuracy, the answer is nineteen. Fourteen is the shorter of the two defensible answers, and a quiz someone actually finishes is worth more than one they abandon.

## the second model

Everything above works by multiplying filters, which *requires* pretending your answers are independent. We know they aren't. That's the whole problem, and the volume-knob fix failed.

So: build something that never makes that assumption in the first place.

Instead of a filter per answer, show a neural network your entire answer sheet at once and let it output a location directly. It has no filters to multiply, so it never double-counts, so it never needs a volume knob to un-double-count.

The catch is that a neural network needs examples — thousands of people's answers paired with their real hometowns — and **that dataset does not exist anywhere public.** That was checked carefully, not assumed. So instead we *generate* fake people: pick a hometown, generate answers from the reconstructed maps, and add realistic damage. Some of them have a personal quirk that makes them consistently use unusual words. 15% of them were raised somewhere other than where they're recorded, because real people move.

Isn't that circular — training a model on data made by the other model? It would be, if the first model were right. It isn't. It's wrong in three specific known ways, and the fake people contain all three. So the network gets to learn how to read people the old model *misreads*.

The honest limit: **it can't learn anything the fake people don't contain.** It's a better reader of the same maps, not a source of new knowledge about American English.

**It works.** At the same twelve questions, it cuts the broken model's typical error from 891 km down to 347 km. The sharper way to say it: **the neural network using five questions beats the broken model using any number of questions at all.** Against the fixed model the margin is smaller but it holds everywhere — 291 km against 361 at the fourteen questions the quiz asks.

Picking the right model matters far more than asking more questions.

## what we actually know

This is the part that matters, and it's the part most projects skip.

**Solidly established:** the pixel reconstruction works (r = 0.955 against the survey's own published numbers). The reconstructed maps beat a no-geography baseline on 294,079 completely separate responses. A totally different survey, run years later on different people, asked the same 122 questions and published its own dot maps — reconstructing those independently and comparing agrees far above chance, and every famous dialect boundary shows up in both.

**Established but only in simulation:** everything about accuracy — the "291 km at fourteen questions" type of claim. Those are measured against people we invented. We invented them to be difficult, and their flaws are sized to match real measurements, but they're still invented.

**Not established at all: how well this works on a real human being.** No public dataset has real people, with known hometowns, answering many dialect questions. The closest one overlaps with our questions on exactly two topics, and two questions can't locate anybody.

So the accuracy numbers are a stress test, not a promise.

**Fixing that is easy and boring:** the quiz records every game along with where you actually grew up. About fifty real games would turn the whole confidence claim from an educated extrapolation into an actual measurement. That's worth more than any further math.

## the short version

Someone threw away the data. There were pictures of the data. We read the pictures back into numbers, checked them against the one thing that was published, and they held up.

Then we built a guesser, discovered its most carefully-reasoned feature was actively hurting it, and only noticed because we checked whether more information made it better — and it didn't.

Then we asked whether twelve questions was the right number, which nobody had ever checked, and it wasn't. Twelve was the point where a broken model gave up.
