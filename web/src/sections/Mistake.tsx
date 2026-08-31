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
import { constants, content, km2, pct } from "../content";
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
      <div className={s.wide}>
        <div className={styles.panel}>
          <div className={styles.mapSide}>
            {payload ? (
              <MapView
                cells={payload.cells}
                posterior={view?.p ?? null}
                places={payload.manifest.places}
                legend={["less likely", "more likely"]}
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

    </section>
  );
}
