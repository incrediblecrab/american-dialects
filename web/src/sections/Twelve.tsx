import { useMemo } from "react";
import Curve from "../components/Curve";
import { constants, curveFor, int, kMax, km, MODELS } from "../content";
import s from "./Section.module.css";

/**
 * Act IV. Where the question count came from.
 *
 * Every figure here is derived from the curve rather than written down, so the
 * argument survives the curve being re-run. In particular the claim that the
 * discounted model turns upward is a computed fact about the data, not a
 * sentence that has to be kept in sync with it.
 */
export default function Twelve() {
  const facts = useMemo(() => {
    const discounted = curveFor(MODELS.discounted);
    const correct = curveFor(MODELS.deployed);
    const net = curveFor(MODELS.net);
    const best = discounted.reduce((a, b) => (b.medianKm < a.medianKm ? b : a));
    const end = discounted[discounted.length - 1];
    return {
      best,
      end,
      turnsUp: end.medianKm > best.medianKm,
      correctEnd: correct[correct.length - 1],
      netEnd: net[net.length - 1],
    };
  }, []);

  return (
    <section className={s.section} id="questions">
      <div className={s.head}>
        <p className={s.act}>Act IV</p>
        <h2 className={s.title}>
          Nobody ever worked out how many questions it should ask.
        </h2>
      </div>

      <div className={s.body}>
        <p>
          Twelve is not a finding about language. It is an interface decision
          from 2013 — a quiz that fits on a page and holds attention — and it
          has been inherited ever since as though it were a property of the
          problem. It is worth asking what the
          number should be, and the honest way to ask is to measure error
          against how many questions are asked and look at the curve.
        </p>
      </div>

      <div className={s.wide}>
        <div className={s.figure}>
          <Curve
            markK={constants.nQuestions}
            series={[
              {
                model: MODELS.deployed,
                label: "Bayes, no idiolect discount",
                tone: "accent",
              },
              { model: MODELS.net, label: "Discriminative network", tone: "ink" },
              {
                model: MODELS.discounted,
                label: `Bayes, discount \u03c1 = ${constants.legacyRho}`,
                tone: "warn",
              },
            ]}
          />
        </div>
      </div>

      <div className={s.body}>
        <p>
          Two of these curves do what you would expect: more questions, less
          error, still falling when the measurement runs out at{" "}
          <span className={s.stat}>{kMax}</span>. At that point the network is
          down to <span className={s.stat}>{km(facts.netEnd.medianKm)}</span> of
          median error and the undiscounted Bayesian model to{" "}
          <span className={s.stat}>{km(facts.correctEnd.medianKm)}</span>.
        </p>
        {facts.turnsUp ? (
          <p>
            The third does something a well-specified model cannot do. It
            reaches its best result at{" "}
            <span className={s.stat}>{facts.best.k}</span> questions —{" "}
            <span className={s.stat}>{km(facts.best.medianKm)}</span> — and then
            gets <em>worse</em>, ending at{" "}
            <span className={s.stat}>{km(facts.end.medianKm)}</span>. Feeding
            more evidence to a model and watching it move further from the truth
            is not a tuning problem. It is the signature of{" "}
            <a href="#mistake">the mistake in Act III</a>: the discount is
            applied so that additional answers pull the answer back toward the
            population prior faster than they push it toward the speaker.
          </p>
        ) : null}
        <p>
          So the question count and the discount cannot be settled separately.
          With the discount in place, asking more than about{" "}
          <span className={s.stat}>{facts.best.k}</span> questions is actively
          harmful, which makes twelve look like a reasonable choice for the
          wrong reason. Remove it and the ceiling disappears; the curve is still
          falling at the end of the measurement, and where to stop becomes a
          judgement about how long a stranger will keep answering, not about
          statistics.
        </p>
        <p>
          This quiz asks{" "}
          <span className={s.stat}>{int.format(constants.nQuestions)}</span>,
          and that is the judgement rather than a result. It is the point at
          which an extra question stops being worth about 30 km of accuracy —
          roughly what one more screen of a stranger&rsquo;s patience seems to
          cost. Ask instead where 90% of the available accuracy has arrived and
          the answer is further out. Both are defensible; neither is a fact
          about dialects, which is the entire point of this section.
        </p>
        <p className={s.note}>
          These are simulated speakers, built to violate the model's assumptions
          in three calibrated ways rather than to obey them. That makes the
          curve an answer to &ldquo;how wrong can this be and still work&rdquo;
          rather than a measurement of accuracy on real people, which this
          project does not have and{" "}
          <a href="#limits">says so plainly</a>.
        </p>
      </div>
    </section>
  );
}
