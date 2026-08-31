import { useEffect, useMemo, useState } from "react";
import MapView from "../components/Map";
import { usePayload } from "../model/usePayload";
import { loadSurface } from "../model/payload";
import type { Answer } from "../model/posterior";
import {
  argmax,
  credibleArea,
  posterior,
  tauUsed,
  topPlaces,
} from "../model/posterior";
import { constants, content, int, km2, pct } from "../content";
import s from "./Section.module.css";
import styles from "./Mistake.module.css";

const MAX_RHO = 0.25;

/**
 * The correction that was the largest source of error.
 *
 * The slider runs the real model, not a lookup between two stored pictures.
 * The claim being made is that one parameter moves the answer this much, and
 * the only convincing way to make it is to let the reader move the parameter.
 * The two endpoints are also exported from Python, so the numbers shown at
 * the deployed rho and at the discount that used to be deployed can be checked
 * against the research code. Note that the slider must be pinned to
 * constants.legacyRho rather than to constants.rho: rho is now zero, so
 * "deployed" and "corrected" would otherwise be the same picture.
 */
export default function Mistake() {
  const { payload } = usePayload();
  const fx = content.fixture;
  const [rho, setRho] = useState(constants.legacyRho);
  const [answers, setAnswers] = useState<Answer[] | null>(null);

  useEffect(() => {
    if (!payload) return;
    let live = true;
    const bits = new Map<string, number>(
      content.questions.map((q) => [q.id, q.bits]),
    );
    Promise.all(
      fx.answers.map(async (a) => ({
        question: a.question,
        choice: a.choice,
        bits: bits.get(a.question) ?? 0,
        surface: await loadSurface(
          payload.manifest,
          payload.cells,
          a.question,
          a.choice,
        ),
      })),
    ).then((rows) => live && setAnswers(rows));
    return () => {
      live = false;
    };
  }, [payload, fx]);

  const view = useMemo(() => {
    if (!payload || !answers) return null;
    const p = posterior(payload.cells, answers, rho, constants.tauBase);
    const cell = argmax(p);
    return {
      p,
      lat: payload.cells.lats[payload.cells.cellY[cell]],
      lon: payload.cells.lons[payload.cells.cellX[cell]],
      place: topPlaces(p, payload.cells, payload.manifest.places, 1)[0],
      area: credibleArea(p, payload.cells.km2, 0.8),
      tau: tauUsed(answers, rho, constants.tauBase),
    };
  }, [payload, answers, rho]);

  const discounted = Math.abs(rho - constants.legacyRho) < 0.004;
  const corrected = Math.abs(rho - constants.rho) < 0.004;

  return (
    <section className={s.section} id="mistake">
      <div className={s.body}>
        <p>
          Treating {int.format(constants.nQuestions)} answers from one person as{" "}
          {int.format(constants.nQuestions)} independent pieces of evidence is
          wrong, and obviously so. People are idiosyncratic: someone who says{" "}
          <em>yinz</em> is more likely to say <em>redd up</em> for reasons that
          have nothing to do with adding new information about where they live.
          Counting both at full strength overstates the case twice.
        </p>
        <p>
          So the model discounted for it. The size of the discount was not
          guessed — it was measured, from the residual correlation between
          answers after location is accounted for, and it came out at{" "}
          <span className={s.stat}>{constants.legacyRho}</span>. That
          measurement is not in dispute. What was wrong was applying it as if
          the discount scaled the whole likelihood.
        </p>
      </div>

      <div className={s.wide}>
        <div className={styles.panel}>
          <div className={styles.mapSide}>
            {payload ? (
              <MapView
                cells={payload.cells}
                posterior={view?.p ?? null}
                markers={
                  view ? [{ lat: view.lat, lon: view.lon, tone: "accent" }] : []
                }
                caption={`The same ${fx.answers.length} answers every time. A speaker whose every answer is the most common one in Pittsburgh.`}
              />
            ) : (
              <div className={styles.skeleton}>Loading the model…</div>
            )}
          </div>

          <div className={styles.readout}>
            <label className={styles.label} htmlFor="rho">
              Idiolect discount
            </label>
            <div className={styles.track}>
              <input
                id="rho"
                className={styles.slider}
                type="range"
                min={0}
                max={MAX_RHO}
                step={0.001}
                value={rho}
                onChange={(e) => setRho(Number(e.currentTarget.value))}
              />
            </div>
            <div className={styles.ticks}>
              <button
                className={styles.tick}
                onClick={() => setRho(constants.rho)}
              >
                {constants.rho} &middot; deployed now
              </button>
              <button
                className={styles.tick}
                onClick={() => setRho(constants.legacyRho)}
              >
                {constants.legacyRho} &middot; what shipped before
              </button>
            </div>

            <dl className={styles.stats}>
              <div>
                <dt>
                  <span className={styles.sym}>&rho;</span>
                </dt>
                <dd className={discounted ? styles.warnVal : undefined}>
                  {rho.toFixed(3)}
                </dd>
              </div>
              <div>
                <dt>
                  Effective <span className={styles.sym}>&tau;</span>
                </dt>
                <dd>{view ? view.tau.toFixed(3) : "—"}</dd>
              </div>
              <div>
                <dt>Best guess</dt>
                <dd className={styles.wide2}>
                  {view ? view.place.name : "—"}
                </dd>
              </div>
              <div>
                <dt>Confidence in it</dt>
                <dd
                  className={
                    discounted
                      ? styles.warnVal
                      : corrected
                        ? styles.goodVal
                        : undefined
                  }
                >
                  {view ? pct(view.place.p, 1) : "—"}
                </dd>
              </div>
              <div>
                <dt>80% area</dt>
                <dd
                  className={
                    discounted
                      ? styles.warnVal
                      : corrected
                        ? styles.goodVal
                        : undefined
                  }
                >
                  {view ? km2(view.area) : "—"}
                </dd>
              </div>
            </dl>

            <p className={styles.verdict}>
              {corrected
                ? "This is what runs now. The model concentrates where the evidence points."
                : discounted
                  ? "This is what shipped before. The discount has pulled the answer back toward the population prior, and the prior's largest mass is not Pittsburgh."
                  : "Drag to either end to see the two positions the project actually held."}
            </p>
          </div>
        </div>
      </div>

      <div className={s.body}>
        <p>
          The discount belongs on the <em>evidence</em>, not on the answer.
          Scaling the entire likelihood does not just weaken the update; past a
          point it drags the posterior back toward the prior, which is a
          statement about where people live rather than about how anyone talks.
          Watch the confidence rather than the name. The best guess survives the
          discount — it is the same town either way — but the belief behind it
          collapses to a fraction of its strength, and the strongest point on
          the map crosses two states to New York City. The model has not become
          appropriately unsure. It has been talked out of what it knew.
        </p>
        <p className={s.note}>
          This is stated plainly because it is the most useful thing here. The
          error was not carelessness; it came from doing the extra, correct
          thing of measuring a real effect, and then applying the measurement in
          the wrong place. A model with no idiolect correction at all was better
          than the one that shipped, and that is the model running here now.
        </p>
      </div>
    </section>
  );
}
