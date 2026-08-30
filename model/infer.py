"""Turn a set of answers into a posterior over where someone grew up.

Bayes over grid cells: the prior says where people live, each answer multiplies
in the surface for that answer, and what survives is a probability map.

The estimand is deliberately narrow. This recovers the place whose 2003 survey
respondents your idiolect most resembles. Because dialect is acquired young that
is usually where you were raised, which is the target it is scored against, but
it is not birthplace and not current address, and for someone who has moved a
lot it is a blend of everywhere they have lived.

Two departures from textbook naive Bayes, and they fix different failures.

`tau` is an exponent on the whole log-likelihood. Naive Bayes assumes answers
are independent given location, which is plainly false here. A Southerner says
y'all, and fixin' to, and crawfish, and coke; those are four expressions of one
underlying fact about a person, not four independent measurements of it.
Multiplying them as though they were independent counts the same evidence over
and over, and the posterior collapses onto a single county with unearned
certainty. Raising the likelihood to tau < 1 discounts each answer to its share
of genuinely new information.

`eps` mixes each answer's surface with the national marginal for that question,
so the model always allows that a person answered a question for a reason that
has nothing to do with where they are from. It has to be separate from tau,
because tau scales every answer by the same factor and so cannot bound the
damage one answer does. Measured on the fitted tensor, the median answer swings
the log-posterior by 5.5 nats across cells and the worst by 9.1, which is about
13 bits, comparable to the entire entropy of the location itself. So a single
misclick, a bidialectal speaker, or an honest "I say both" really can outweigh
everything else combined.

eps does not cap that at log(1/eps). It floors the likelihood at eps * m(a), so
the most one answer can move the log-posterior is log(p_max / (eps * m(a))),
and for a rare answer m(a) is small enough that the floor barely rises. That is
the right behaviour rather than a defect: saying yinz genuinely is two orders of
magnitude more likely in Pittsburgh than in Texas, and a model that refused to
believe it would be worse. What eps buys is that the floor is now a stated
misclick rate times how often people say the thing at all, instead of the flat
1e-4 in tensor.py, which was picked to avoid log(0) and means nothing.

tau sets the overall width; eps protects the tails. Neither is set by hand. Both
are fitted in calibrate.py against people whose locations are known, so that a
stated 80% means right 80% of the time.
"""

import sys
from pathlib import Path

import numpy as np

from prior import population_prior
from tensor import Tensor, haversine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))
from common import DATA  # noqa: E402

EARTH_CELL_KM2 = None  # computed per cell; latitude changes cell width

RHO = 0.177
"""Within-person residual correlation between two distinct dialect features.

Measured on 349 YGDP respondents who answered two different constructions,
after subtracting what the model expects given where they were raised. The
raw correlation is 0.193 and conditioning on location brings it only to 0.177,
which is the important part: the dependence between a person's answers is
almost entirely NOT explained by geography. It is the person. 95% CI is about
plus or minus 0.11, and it is measured on two constructions, so treat it as an
order of magnitude rather than a precise figure.
"""

TAU_BASE = 0.55
"""Everything wrong with the model that is not double-counted evidence.

Surfaces recovered from pixels are imperfect, a Likert rating is not a forced
choice, people move, and the model has no way to represent age or race or class.
All of that widens the honest posterior by a factor that does not grow with the
number of questions, unlike redundancy, which does. Separating the two is what
lets tau extrapolate at all.

This number is weakly identified and should be treated as provisional. Fitted
against 1,450 YGDP respondents it came out at 0.55, but everything from 0.3 to
0.85 scores within 0.01 nats per person, because that validation set can only
test two distinct constructions and two constructions cannot locate anyone: the
model beats the population prior by 0.06 nats and 25 km on it.

The absolute coverage on that set cannot be read as calibration, because the
population prior ALONE covers YGDP respondents 63, 88 and 97 percent of the
time at nominal 50, 80 and 95. An online syntax survey is not a population
sample; its respondents sit in large metros far more often than the population
they are drawn from, and that skew is bigger than the effect being measured.

What survives the skew is the CHANGE. Adding answers at base=1 moves coverage
to 60/83/94, three to five points BELOW where the prior alone sits on the same
people: the answers are making the posterior too narrow. At base=0.55 coverage
returns to the prior's own level. On a biased sample that is the right target,
because a calibrated model should inherit exactly the offset its prior inherits.
That differential argument, not the absolute level, is why 0.55 is used.

RHO is the more trustworthy of the two numbers, because it is a within-person
quantity and so does not care who was sampled. Calibrating the rest honestly
needs real people, with known hometowns, answering many questions. No public
dataset has that. quiz.py records exactly that, so this can eventually be
measured rather than assumed.
"""

N_QUESTIONS = 12
"""How many questions the quiz asks, everywhere it is played.

Terminal and browser both read this, so the length is defined once. It is 12
because the 2013 New York Times quiz asked 12, which is the only reason anyone
has ever asked 12; it was never derived from anything about dialects.

findings.md recommends 14 and does not merely prefer it: with RHO at its
deployed 0.177 the error bottoms at eleven-to-twelve questions and then climbs
again, because the design-effect discount outruns the evidence faster than the
questions inform it. Twelve is the correct stopping point for that model and
for no other. With RHO = 0 both correctly specified models are still improving
at the twenty-question measurement cap, 90% of the achievable reduction has
arrived by 14, and 14 is also the optimum if a question is priced at 30 km of
error.

Left at 12 because raising it and leaving RHO at 0.177 would make the model
worse, not better. The two constants have to move together, and both are
waiting on a question ordering re-derived with the discount off.
"""


def tau_for(k, rho=None, base=None):
    """How much to discount the likelihood after k answers.

    A single scalar tau is wrong, and wrong in a direction that gets worse the
    more questions you ask. If answers carry pairwise residual correlation rho,
    then k of them are worth k / (1 + (k-1) rho) independent observations. That
    is the survey statistician's design effect, and it means the right exponent
    shrinks as the quiz lengthens: at two questions almost nothing is
    double-counted, at twenty-five most of it is.

    Fitting one tau on people who answered two questions and applying it to a
    twenty-five question quiz would therefore be badly overconfident, which is
    exactly the trap this project set for itself by having only a two-question
    validation set.

    The extrapolation to large k is NOT measured. No public dataset has real
    people, with known hometowns, answering many dialect questions; that is why
    quiz.py records answers, so this can eventually be checked rather than
    assumed.

    This counts questions, which is only right when they carry equal
    information. They do not, so prefer tau_for_weights. See its docstring for
    what goes wrong.
    """
    k = max(int(k), 1)
    rho = RHO if rho is None else rho
    base = TAU_BASE if base is None else base
    return float(base / (1.0 + (k - 1) * rho))


def tau_for_weights(w, rho=None, base=None):
    """The same design effect, but counting information instead of questions.

    Discounting by raw question count assumes every answer carries the same
    information. In this quiz they differ by a factor of six: the first question
    is worth 0.581 bits and the twentieth 0.094. Counting them equally means a
    weak question increments the discount by a whole unit while adding almost no
    signal, so it costs more than it contributes. The effect is not subtle.
    Measured over the deployed ordering, effective information PEAKS AT SEVEN
    QUESTIONS and then falls, and is 18% lower at twenty than at seven. A model
    that gets worse when you tell it more is not being conservative, it is
    misspecified.

    The standard repair is Kish's effective sample size for unequal weights,
    (sum w)^2 / sum w^2, which counts near-k when the weights are even and much
    less when a few dominate. Substituting it for k moves the peak from seven to
    ten and leaves 31% more effective information at twenty questions.

    A residual decline survives that, because even the Kish count keeps growing
    after the information has flattened. So the schedule is additionally held to
    a monotonicity floor: more evidence may never leave the model with less
    effective evidence than it already had. That is a constraint rather than a
    fitted parameter, it binds only past the peak, and it is worth about half a
    percent.

    Weights are sorted descending first so the prefixes the floor is computed
    over are the informative ones, which makes the result independent of the
    order the questions happened to be asked in.
    Both rho and base default to the module-level RHO and TAU_BASE, resolved at
    call time rather than at import, so a sweep can override them by rebinding
    the module attribute.
    """
    rho = RHO if rho is None else rho
    base = TAU_BASE if base is None else base
    w = np.sort(np.asarray([x for x in w if x > 0], dtype=float))[::-1]
    if w.size == 0:
        return float(base)

    cum = np.cumsum(w)
    cum_sq = np.cumsum(w * w)
    k_eff = np.where(cum_sq > 0, cum * cum / np.maximum(cum_sq, 1e-30), 1.0)
    total = base * cum / (1.0 + (k_eff - 1.0) * rho)
    return float(np.maximum.accumulate(total)[-1] / cum[-1])


class Geolocator:
    def __init__(self, tau="auto", eps=0.10, vintage="pop2003", tensor=None):
        self.t = tensor or Tensor()
        self.tau = tau
        self.eps = eps
        self.prior, _ = population_prior(self.t, vintage)
        self.log_prior = np.log(self.prior)
        self.index = {
            (q, c): i for i, (q, c) in enumerate(zip(self.t.question, self.t.choice))
        }
        self.cell_km2 = cell_areas(self.t)
        self._marginal = None
        self._bits = None

    @property
    def marginal(self):
        """P(answer) for a random person drawn from the prior.

        This is what an answer means once you strip out geography, so it is the
        right thing to mix toward when discounting an answer as uninformative.
        Computed from the tensor and the prior rather than the survey's
        published national percentages, so it is consistent with the surfaces
        and with the population the model is actually reasoning about.
        """
        if self._marginal is None:
            m = np.empty(self.t.logp.shape[0])
            for i in range(0, len(m), 64):
                block = np.exp(self.t.logp[i:i + 64])
                m[i:i + 64] = block @ self.prior
            self._marginal = np.maximum(m, 1e-9)
        return self._marginal

    @property
    def question_bits(self):
        """Mutual information in bits between each question and location.

        Marginal rather than conditional on other answers, so it does not depend
        on the order questions were asked in, which the adaptive quiz changes
        from player to player. It is a per-question property of the surfaces.
        """
        if self._bits is None:
            out = {}
            for q, rows in self.t.rows.items():
                p = np.exp(self.t.logp[rows])
                m = self.marginal[rows][:, None]
                out[q] = float(
                    (self.prior[None, :] * p * np.log2(np.maximum(p, 1e-12)
                                                       / m)).sum())
            self._bits = out
        return self._bits

    def tau_used(self, answers):
        """The tau posterior() would apply to this set of answers."""
        if self.tau != "auto":
            return float(self.tau)
        bits = self.question_bits
        w = [bits.get(str(q), 0.0) for q, c in answers
             if (str(q), str(c)) in self.index]
        return tau_for_weights(w)

    def loglik(self, i):
        """log P(answer i | cell), contaminated toward the national marginal."""
        if self.eps <= 0:
            return self.t.logp[i]
        return np.logaddexp(
            np.log1p(-self.eps) + self.t.logp[i],
            np.log(self.eps * self.marginal[i]),
        )

    def posterior(self, answers, tau=None, eps=None):
        """answers is a sequence of (question, choice); returns P(cell).

        tau defaults to "auto", which scales the discount to how many answers
        were actually used rather than applying one exponent regardless.
        """
        answers = list(answers)
        keep = self.eps
        if eps is not None:
            self.eps = eps
        try:
            lp = self.log_prior.copy()
            used = []
            for q, c in answers:
                i = self.index.get((str(q), str(c)))
                if i is not None:
                    lp += self.loglik(i)
                    used.append(str(q))
            t = self.tau if tau is None else tau
            if t == "auto":
                bits = self.question_bits
                t = tau_for_weights([bits.get(q, 0.0) for q in used])
            if used:
                lp = self.log_prior + t * (lp - self.log_prior)
        finally:
            self.eps = keep
        lp -= lp.max()
        p = np.exp(lp)
        return p / p.sum()

    def report(self, post, places=None, k=5):
        """Everything worth saying about a posterior."""
        best = int(np.argmax(post))
        out = {
            "map_lat": float(self.t.cell_lat[best]),
            "map_lon": float(self.t.cell_lon[best]),
            "map_state": str(self.t.state[best]),
            "entropy_bits": float(entropy_bits(post)),
            "states": top_states(self.t, post, k),
        }
        for level in (0.5, 0.8, 0.95):
            out[f"area{int(level * 100)}_km2"] = credible_area(
                post, self.cell_km2, level)
        if places is not None:
            out["places"] = places.rank(self.t, post, k)
            out["areas"] = places.areas(self.t, post, k)
        return out


def cell_areas(t):
    """km^2 per grid cell, which shrinks toward the poles."""
    dlat = abs(float(t.lats[1] - t.lats[0]))
    dlon = abs(float(t.lons[1] - t.lons[0]))
    h = dlat * 111.32
    w = dlon * 111.32 * np.cos(np.radians(t.cell_lat))
    return h * w


def entropy_bits(p):
    q = p[p > 0]
    return -float((q * np.log2(q)).sum())


def credible_area(post, cell_km2, level=0.8):
    """Area of the smallest set of cells holding `level` of the mass."""
    order = np.argsort(post)[::-1]
    cum = np.cumsum(post[order])
    n = int(np.searchsorted(cum, level) + 1)
    return float(cell_km2[order[:n]].sum())


def credible_cells(post, level=0.8):
    order = np.argsort(post)[::-1]
    cum = np.cumsum(post[order])
    return order[: int(np.searchsorted(cum, level) + 1)]


def top_states(t, post, k=5):
    totals = {}
    for s in np.unique(t.state):
        totals[str(s)] = float(post[t.state == s].sum())
    return sorted(totals.items(), key=lambda kv: -kv[1])[:k]


def error_km(t, post, lat, lon):
    """Great-circle distance from the posterior mode to the truth."""
    best = int(np.argmax(post))
    return float(haversine(lat, lon, t.cell_lat[best], t.cell_lon[best]))


def contains(t, post, lat, lon, level=0.8):
    """Did the credible region actually cover the true location?"""
    cells = set(credible_cells(post, level).tolist())
    return t.nearest(lat, lon) in cells


class Places:
    """Named towns, so the model can answer with a place instead of a bearing."""

    def __init__(self, min_pop=25000):
        import csv
        self.min_pop = int(min_pop)
        self.rows = []
        path = DATA / "census" / "places.csv"
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if int(r["pop"]) >= min_pop:
                    self.rows.append((r["name"], r["state"], float(r["lat"]),
                                      float(r["lon"]), int(r["pop"])))
        self.lat = np.array([r[2] for r in self.rows])
        self.lon = np.array([r[3] for r in self.rows])
        self.pop = np.array([r[4] for r in self.rows], dtype=float)
        self._cells = None
        self._catchment = None

    def cells(self, t):
        if self._cells is None:
            self._cells = np.array([t.nearest(la, lo)
                                    for la, lo in zip(self.lat, self.lon)])
        return self._cells

    def catchment(self, t):
        """Index of the nearest listed place for every cell in the grid.

        This exists because the obvious approach is badly wrong. Scoring only
        the cells that happen to contain a listed town throws away 84% of the
        posterior, and it does not throw it away evenly: it is discarded
        wherever people live in places too small to list. Pittsburgh's metro
        population sits in dozens of sub-25,000 boroughs while New York's sits
        in large incorporated cities, so the discard silently penalises
        Pittsburgh. The visible symptom was a model that put the state at PA
        20.6% with its modal cell 3 km from Pittsburgh while reporting New York
        as the most likely place.

        Assigning every cell to its nearest named place instead makes the town
        scores a partition of the whole posterior. Nothing is dropped and
        nothing is counted twice. Population is deliberately not a factor here,
        because the prior is already population-weighted and multiplying by it
        again counts it twice.
        """
        if self._catchment is None:
            cache = DATA / "model" / f"catchment_{self.min_pop}.npy"
            if cache.exists():
                self._catchment = np.load(cache)
            else:
                out = np.empty(t.n_cells, dtype=np.int32)
                for lo in range(0, t.n_cells, 4096):
                    hi = min(lo + 4096, t.n_cells)
                    d = haversine(t.cell_lat[lo:hi, None], t.cell_lon[lo:hi, None],
                                  self.lat[None, :], self.lon[None, :])
                    out[lo:hi] = np.argmin(d, axis=1)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, out)
                self._catchment = out
        return self._catchment

    def _scores(self, t, post):
        who = self.catchment(t)
        score = np.zeros(len(self.rows))
        np.add.at(score, who, post)
        total = score.sum()
        return score / total if total > 0 else score

    def rank(self, t, post, k=5):
        """Most probable named places, as a partition of the whole posterior.

        Each place gets the posterior mass of every cell closer to it than to
        any other listed place, so these are probabilities that a person is from
        that place's catchment, and they sum to one across the country.
        """
        score = self._scores(t, post)
        if not score.any():
            return []
        order = np.argsort(score)[::-1][:k]
        return [(self.rows[i][0], self.rows[i][1], float(score[i])) for i in order]

    def areas(self, t, post, k=5, radius_km=60.0):
        """Rank metropolitan areas rather than incorporated towns.

        A list of towns splits one answer across its own suburbs. Asked where a
        Pittsburgher is from, an unaggregated ranking says Pittsburgh, then
        Bethel Park, then Monroeville -- three separate ways of saying
        Pittsburgh, none of which sounds like the model knows anything.
        Municipal boundaries are not dialect boundaries, and at this model's
        12.7 km resolution they are far below what it can resolve anyway.

        So absorb each town into the highest-scoring town within radius_km and
        add the probabilities. Seeding greedily from the top score builds
        clusters around the places the posterior actually likes, but three
        things then have to be corrected. Clusters are ranked by their summed
        probability, not by the seed's, since a seed that beats another town
        individually can easily lose once both have absorbed their suburbs. Each
        cluster is named for its most populous member rather than its seed, so
        that a cluster covering Chicago is called Chicago even when the seed
        that formed it was Elk Grove Village. And each cluster is re-centred on
        that dominant city before absorbing, because a ball drawn around a
        suburb is off-centre by however far the suburb sits from downtown, so a
        metro wider than the radius gets split into two balls that both then
        take metro names -- Dallas and Grand Prairie, 27 km apart, were being
        reported as separate answers. Re-centring makes the ball the city's own.

        The result is a partition, so the numbers still sum to one and nothing
        is double counted.
        """
        score = self._scores(t, post)
        if not score.any():
            return []

        taken = np.zeros(len(score), dtype=bool)
        clusters = []
        for i in np.argsort(score)[::-1]:
            if taken[i]:
                continue
            free = ~taken
            near = (haversine(self.lat[i], self.lon[i],
                              self.lat, self.lon) <= radius_km) & free
            members = np.flatnonzero(near)
            hub = members[np.argmax(self.pop[members])]

            near = (haversine(self.lat[hub], self.lon[hub],
                              self.lat, self.lon) <= radius_km) & free
            near[i] = True
            taken |= near
            members = np.flatnonzero(near)
            label = members[np.argmax(self.pop[members])]
            clusters.append((self.rows[label][0], self.rows[label][1],
                             float(score[members].sum())))

        clusters.sort(key=lambda c: c[2], reverse=True)
        return clusters[:k]


if __name__ == "__main__":
    g = Geolocator()
    cases = [
        ("coherent Pittsburgher", [("105", "b"), ("50", "f"), ("64", "c")],
         "pop, yinz, hoagie"),
        ("with one answer from elsewhere",
         [("105", "b"), ("50", "f"), ("64", "c"), ("103", "a")],
         "pop, yinz, hoagie, bubbler"),
    ]
    for title, demo, gloss in cases:
        print(f"{title}: {gloss}")
        for eps in (0.0, 0.05):
            r = g.report(g.posterior(demo, eps=eps))
            top = ", ".join(f"{s} {v:.0%}" for s, v in r["states"][:3])
            print(f"  eps={eps:<5} {r['map_state']} "
                  f"({r['map_lat']:.1f}, {r['map_lon']:.1f})  "
                  f"area80 {r['area80_km2'] / 1000:,.0f}k km2  {top}")
        print()
    print("bubbler is Milwaukee and Providence; yinz is Pittsburgh. Nobody says")
    print("both, so the model splits the difference and lands in Indiana, which")
    print("is nowhere. A wider posterior is the honest response to a person who")
    print("does not fit, and that is what eps and tau are fitted to produce.")
