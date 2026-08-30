import Scrubber from "../components/Scrubber";
import { content, int, oneDp, recovery, recoveryStrip } from "../content";
import s from "./Section.module.css";

/**
 * Act II. The part that should not have worked.
 *
 * The load-bearing sentence is that the validation used numbers the recovery
 * never saw. Recovering geography from a picture is only interesting if it can
 * be checked against something independent, and the published state
 * percentages are exactly that: they were never an input.
 */
export default function Recovery() {
  const strip = recoveryStrip;
  return (
    <section className={s.section} id="recovery">
      <div className={s.head}>
        <p className={s.act}>Act II</p>
        <h2 className={s.title}>The data was never published. The pictures were.</h2>
      </div>

      <div className={s.body}>
        <p>
          The Harvard Dialect Survey collected where every respondent grew up,
          down to their ZIP code, and released none of it. What it released was
          one image per answer: a dot for each person, drawn at their ZIP
          centroid on a map of the lower 48. For twenty years those images have
          been the only surviving sub-state record of the survey.
        </p>
        <p>
          They are recoverable, because of how they were drawn. Every dot is a
          known colour over a white background, and the drawing was
          antialiased, so a pixel on the edge of a dot is a measurable blend of
          ink and paper. Solving that blend per pixel gives back not just{" "}
          <em>where</em> people were, but roughly <em>how many</em>.
        </p>
      </div>

      <div className={s.wide}>
        <div className={s.figure}>
          <Scrubber
            stages={strip.stages}
            dir="recovery"
            sourceLabel={`Harvard Dialect Survey question ${strip.question}, “${strip.answer}”,`}
          />
        </div>
      </div>

      <div className={s.body}>
        <p>
          The step above recovers{" "}
          <span className={s.stat}>{int.format(strip.inkedPixels)}</span> inked
          pixels from a single answer map, and the same procedure over every
          published map recovers the geography of the whole survey.
        </p>
        <h3>Checking it against numbers it never saw</h3>
        <p>
          A recovery like this is easy to believe and hard to trust. So it is
          checked against the one thing the survey did publish in numeric form:
          the percentage of each state choosing each answer. Those percentages
          were never an input to the recovery. If reading the pixels were
          fabricating structure, there is no reason the fabrication would agree
          with them.
        </p>
        <p>
          Over <span className={s.stat}>{int.format(recovery.comparisons)}</span>{" "}
          state-by-answer comparisons, it agrees at{" "}
          <strong>
            r = <span className={s.stat}>{recovery.r.toFixed(3)}</span>
          </strong>
          , with a mean absolute error of{" "}
          <span className={s.stat}>{oneDp.format(recovery.mae)}</span> percentage
          points, and it picks the same winning answer in a state{" "}
          <span className={s.stat}>{oneDp.format(recovery.modalAgreement)}%</span>{" "}
          of the time.
        </p>
        <p>
          A second, fully independent check comes from outside the survey
          entirely. The <em>Pop vs Soda</em> project collected the same question
          from{" "}
          <span className={s.stat}>{int.format(content.popVsSoda.categorised)}</span>{" "}
          people across{" "}
          <span className={s.stat}>{int.format(content.popVsSoda.counties)}</span>{" "}
          counties, with different respondents, a different decade and a
          different method. The recovered surface reproduces its geography too.
        </p>
        <p className={s.note}>
          Both checks are at the state level, because state percentages are the
          only published numbers to check against. Nothing here validates the
          recovery <em>within</em> a state, and the sub-state detail — which is
          the entire reason to do this — remains unvalidated. That is the honest
          limit of the claim, and{" "}
          <a href="#limits">it is stated again at the end</a>.
        </p>
      </div>
    </section>
  );
}
