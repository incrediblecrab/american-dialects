import {
  constants,
  content,
  curveAt,
  int,
  MODELS,
  oneDp,
  popVsSoda,
  quantisation,
  recovery,
  tuning,
} from "../content";
import s from "./Section.module.css";
import styles from "./Limits.module.css";

/**
 * Act V. What is and is not known.
 *
 * The ladder, strongest claim at the top, is the shape of the argument: a
 * simulation cannot promote itself to a validation, so the tiers are named for
 * what produced them rather than for how confident they sound. Ending here,
 * rather than on the accuracy figures, is what makes the accuracy figures
 * worth anything.
 */
export default function Limits() {
  const cov = curveAt(MODELS.deployed, constants.nQuestions);

  return (
    <section className={s.section} id="limits">
      <div className={s.head}>
        <p className={s.act}>Act V</p>
        <h2 className={s.title}>What is known, and what is only hoped.</h2>
      </div>

      <div className={s.body}>
        <p>
          Simulating people from the same surfaces you are testing, and letting
          them obey every assumption the model makes, guarantees a flattering
          answer and tests nothing. So the claims here are sorted by what
          produced them, and the line between evidence and self-agreement is
          drawn explicitly.
        </p>
      </div>

      <div className={s.wide}>
        <ol className={styles.ladder}>
          <li className={styles.rung}>
            <p className={styles.tier}>Externally validated</p>
            <h3 className={styles.claim}>
              The recovered surfaces reproduce numbers they never saw.
            </h3>
            <p className={styles.detail}>
              Against the survey's own published state percentages, r ={" "}
              <span className={s.stat}>{recovery.r.toFixed(3)}</span> over{" "}
              <span className={s.stat}>{int.format(recovery.comparisons)}</span>{" "}
              comparisons, mean absolute error{" "}
              <span className={s.stat}>{oneDp.format(recovery.mae)}</span> points.
              Independently, tuning against{" "}
              <span className={s.stat}>{int.format(popVsSoda.categorised)}</span>{" "}
              responses from a separate project across{" "}
              <span className={s.stat}>{int.format(popVsSoda.counties)}</span>{" "}
              counties cuts log-loss to{" "}
              <span className={s.stat}>{tuning.logloss.toFixed(4)}</span>.
            </p>
          </li>

          <li className={styles.rung}>
            <p className={styles.tier}>Corroborated by a second survey</p>
            <h3 className={styles.claim}>
              An independent population, recovered the same way, agrees.
            </h3>
            <p className={styles.detail}>
              The Cambridge Survey of World Englishes re-ran the Harvard
              question set years later on differently recruited respondents and
              published its results as dot maps too. Recovering those maps and
              comparing gives agreement well above the chance baseline, and the
              margin is flat across lexical, phonetic and syntactic questions
              rather than concentrated in the easy ones. Both recoveries use the
              same pixel method, so a bias shared by that method would be
              invisible here — what this rules out is that the recovery is
              mostly noise.
            </p>
          </li>

          <li className={styles.rung}>
            <p className={styles.tier}>Measured, but narrow</p>
            <h3 className={styles.claim}>
              The idiolect correlation is real. The use made of it was not.
            </h3>
            <p className={styles.detail}>
              &rho; = <span className={s.stat}>{constants.rho}</span> comes from
              real people answering two constructions, and a within-person
              correlation does not much care who was sampled. That is a
              different claim from the one the model made with it, which was
              that the whole likelihood should be tempered — and{" "}
              <a href="#mistake">that claim failed on its own terms</a>.
            </p>
          </li>

          <li className={styles.rung}>
            <p className={styles.tier}>Stress-tested, not validated</p>
            <h3 className={styles.claim}>
              Every accuracy figure on this page is from simulated speakers.
            </h3>
            <p className={styles.detail}>
              The simulated respondents are built to violate the model's
              assumptions in three calibrated ways, so the numbers answer
              &ldquo;how wrong can this be and still work&rdquo; rather than
              &ldquo;how well does it do against people invented to agree with
              it&rdquo;. The magnitudes of those violations are pinned to
              measurements; their shapes are assumptions, and no simulation can
              tell you the shape of your own blind spot.
              {cov ? (
                <>
                  {" "}
                  On that population the model&rsquo;s nominal 80% region
                  contains the truth{" "}
                  <span className={s.stat}>
                    {(cov.cover80 * 100).toFixed(1)}%
                  </span>{" "}
                  of the time.
                </>
              ) : null}
            </p>
          </li>

          <li className={styles.rung}>
            <p className={styles.tier}>Bounded by its own simulator</p>
            <h3 className={styles.claim}>
              The network never saw a person, or a surface that was wrong.
            </h3>
            <p className={styles.detail}>
              Every speaker the discriminative model has been trained on was
              generated from the model's own surfaces. It cannot discover that a
              surface is wrong, cannot learn a feature the survey never asked
              about, and inherits its question ordering from the model it beats.
              Its advantage is at inference — it does not assume answers are
              independent — not at evidence.
            </p>
          </li>

          <li className={`${styles.rung} ${styles.wall}`}>
            <p className={styles.tierWarn}>Not established at all</p>
            <h3 className={styles.claim}>
              How well this locates a real human being is unknown.
            </h3>
            <p className={styles.detail}>
              There is no public dataset of real people with both their dialect
              answers and where they grew up. Without one, every figure above is
              a statement about surfaces and simulations. Stratified checks on
              the one partial source available show exactly the confounding you
              would expect — error varies substantially by respondent
              demographics, and the model has no way to represent age, race or
              class. Sub-state geography, which is the entire point of
              recovering the pixels, is validated nowhere: the only published
              numbers to check against are state totals.
            </p>
          </li>
        </ol>
      </div>

      <div className={s.body}>
        <h3>Why no figure here was measured on a real player</h3>
        <p className={s.note}>
          Every confidence figure on this page — median error, state accuracy,
          the coverage of the 80% region — is simulated rather than measured on
          the people who play the quiz, and this site is not collecting what
          would change that. It is a static file: it can run the model in your
          browser, but it has nowhere to send the game afterwards. The
          workaround a static site permits is to invite the player to file a
          public report carrying their result and the town they grew up in.
          That was considered and declined. Such reports come only from people
          willing to publish where they were raised, and, among those,
          disproportionately from the ones whose guess was uncanny or absurd —
          selection on precisely the quantity being estimated, uncorrectable
          without knowing who chose not to write. A number built that way would
          read as a measurement while carrying less than this paragraph does.
          The instrument that would do it properly already exists:{" "}
          <code>site/server.py</code> logs whole games against a stated home
          town, and runs on a local machine under a consent conversation a
          share button cannot hold.
        </p>
      </div>

      <div className={s.body}>
        <h3>Two smaller things, for completeness</h3>
        <p className={s.note}>
          The surfaces are shipped to your browser as one byte per cell. Over{" "}
          <span className={s.stat}>{int.format(quantisation.games)}</span>{" "}
          simulated games that costs nothing measurable: the chosen cell is
          identical in{" "}
          <span className={s.stat}>{oneDp.format(quantisation.identicalPct)}%</span>{" "}
          of them, and in the rest the cell picked instead was within{" "}
          <span className={s.stat}>
            {oneDp.format(100 * (1 - quantisation.worstRatio))}%
          </span>{" "}
          of the true best cell's probability. Compression is breaking ties, not
          losing accuracy. Separately, the questions and answers here are the
          survey's own wording from 2003, reproduced unchanged, and{" "}
          <span className={s.stat}>{int.format(content.inventory.length)}</span>{" "}
          data files stand behind everything above.
        </p>
      </div>
    </section>
  );
}
