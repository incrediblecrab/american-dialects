import { useCallback, useEffect, useMemo, useState } from "react";
import MapView from "../components/Map";
import Share from "../components/Share";
import type { CardSpec } from "../components/ShareCard";
import { usePayload } from "../model/usePayload";
import { loadSurface, prefetchQuestion } from "../model/payload";
import type { Answer } from "../model/posterior";
import {
  argmax,
  credibleArea,
  posterior,
  topPlaces,
  topStates,
} from "../model/posterior";
import type { CurveRow, Named } from "../model/types";
import { asked, constants, curveAt, int, km, km2, MODELS, pct } from "../content";
import styles from "./Quiz.module.css";

type Stage = "intro" | "playing" | "result";

const SITE = "https://incrediblecrab.github.io/american-dialects/";
const CARD_FILE = "where-do-you-talk-like.png";

const names = (rows: Named[]) => rows.map((r) => r.name).join(" · ");
const withPct = (rows: Named[]) =>
  rows.map((r) => `${r.name} ${pct(r.p)}`).join(" · ");
const plural = (n: number) => `${int.format(n)} ${n === 1 ? "answer" : "answers"}`;

/**
 * What the guess is worth, in one sentence, in the player's own terms.
 *
 * This travels with every copied result and is printed on every shared card,
 * which is the whole reason it is written here rather than left to the reader
 * to infer from the page they are leaving. A picture of a map with a town name
 * on it makes a claim on its own; a picture that has to survive being pasted
 * into a group chat with no page around it has to carry its own caveat, or the
 * caveat is not being made.
 */
function caveat(row: CurveRow | undefined): string {
  if (!row) {
    return (
      "Nothing was answered, so this is only the population prior: where " +
      "people in the United States live, not where you talk like."
    );
  }
  return (
    `Measured on ${int.format(row.n)} simulated speakers rather than on real ` +
    `ones: at ${plural(row.k)} this model’s median error is ${km(row.medianKm)} ` +
    `and it names the right state ${pct(row.stateAcc)} of the time. This guess ` +
    `is one draw from that distribution, not a measurement.`
  );
}

/**
 * Act I. The quiz, run entirely in the browser.
 *
 * The questions come in the published ordering rather than being chosen
 * adaptively. That is a deliberate loss: adaptive selection is measurably
 * better, and site/server.py still does it. But every accuracy figure this
 * page quotes was measured on the fixed ordering, and a page that ran one
 * quiz while citing numbers from a different one would be quietly lying.
 */
export default function Quiz() {
  const { payload, error } = usePayload();
  const [stage, setStage] = useState<Stage>("intro");
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<Answer[]>([]);
  const [busy, setBusy] = useState(false);

  const question = asked[Math.min(step, asked.length - 1)];

  const post = useMemo(() => {
    if (!payload) return null;
    return posterior(payload.cells, answers, 0, constants.tauBase);
  }, [payload, answers]);

  useEffect(() => {
    if (!payload || stage !== "playing") return;
    prefetchQuestion(payload.manifest, payload.cells, question.id, question.choices);
    const next = asked[step + 1];
    if (next) {
      prefetchQuestion(payload.manifest, payload.cells, next.id, next.choices);
    }
  }, [payload, stage, step, question]);

  const advance = useCallback(() => {
    if (step + 1 >= asked.length) setStage("result");
    else setStep(step + 1);
  }, [step]);

  const answer = useCallback(
    async (choice: string) => {
      if (!payload || busy) return;
      setBusy(true);
      try {
        const surface = await loadSurface(
          payload.manifest,
          payload.cells,
          question.id,
          choice,
        );
        setAnswers((prev) => [
          ...prev,
          { question: question.id, choice, bits: question.bits, surface },
        ]);
        advance();
      } finally {
        setBusy(false);
      }
    },
    [payload, busy, question, advance],
  );

  const restart = () => {
    setAnswers([]);
    setStep(0);
    setStage("playing");
  };

  const result = useMemo(() => {
    if (!payload || !post || stage !== "result") return null;
    const cell = argmax(post);
    return {
      lat: payload.cells.lats[payload.cells.cellY[cell]],
      lon: payload.cells.lons[payload.cells.cellX[cell]],
      places: topPlaces(post, payload.cells, payload.manifest.places, 3),
      states: topStates(post, payload.cells, payload.manifest.states, 3),
      area: credibleArea(post, payload.cells.km2, 0.8),
    };
  }, [payload, post, stage]);

  const measured = curveAt(MODELS.deployed, answers.length);
  const said = caveat(measured);
  /* A player who skipped every question is looking at the population prior,
   * so the headline above the town name must not call it a guess about them. */
  const lede = answers.length ? "Best guess" : "Nothing answered";

  const rows = useMemo(
    () =>
      result
        ? [
            { label: "Runners-up", value: names(result.places.slice(1)) },
            { label: "Likely states", value: withPct(result.states) },
            { label: "80% area", value: km2(result.area) },
          ]
        : [],
    [result],
  );

  /**
   * The result as plain text.
   *
   * Plain rather than a link with parameters, because a link would encode the
   * player's answers into a URL they are about to paste somewhere public, and
   * a set of dialect answers is a reasonably distinctive thing to publish
   * about yourself. Text says the same thing and carries nothing extra.
   */
  const text = result
    ? [
        `Where do you talk like? — ${plural(answers.length)} to the Harvard Dialect Survey.`,
        "",
        `${lede}: ${result.places[0].name}`,
        `Runners-up: ${names(result.places.slice(1))}`,
        `Most likely states: ${withPct(result.states)}`,
        `80% of the model’s belief covers ${km2(result.area)}`,
        "",
        said,
        "",
        SITE,
      ].join("\n")
    : "";

  const buildCard = useCallback((): CardSpec => {
    if (!payload || !post || !result) throw new Error("the game is not finished");
    return {
      cells: payload.cells,
      posterior: post,
      marker: { lat: result.lat, lon: result.lon },
      kicker: "Where do you talk like?",
      strap:
        "Harvard Dialect Survey, 2003. Its geography was recovered from the " +
        "pixels of its own published maps, then run backwards.",
      mapNote: "Stronger blue is more likely. The dot is the single most likely cell.",
      headingLabel: answers.length
        ? `Best guess after ${plural(answers.length)}`
        : "Nothing answered — the population prior",
      heading: result.places[0].name,
      rows,
      honest: said,
      source: "incrediblecrab.github.io/american-dialects",
    };
  }, [payload, post, result, answers.length, rows, said]);

  return (
    <section
      className={stage === "intro" ? `${styles.section} ${styles.hero}` : styles.section}
      id="play"
    >
      <div className={styles.grid}>
        <div className={styles.mapCol}>
          {payload ? (
            <MapView
              cells={payload.cells}
              posterior={post}
              markers={
                result ? [{ lat: result.lat, lon: result.lon, tone: "accent" }] : []
              }
              caption={
                stage === "intro"
                  ? "The population prior. Before you answer anything, the best guess is simply where people live."
                  : answers.length === 0
                    ? "Nothing answered yet."
                    : `${int.format(answers.length)} ${answers.length === 1 ? "answer" : "answers"} in. Stronger blue is more likely.`
              }
            />
          ) : (
            <div className={styles.skeleton}>
              {error ? `Could not load the model: ${error}` : "Loading the model…"}
            </div>
          )}
        </div>

        <div className={styles.panel}>
          {stage === "intro" ? (
            <div className={styles.intro}>
              <p className={styles.kicker}>Harvard Dialect Survey · 2003</p>
              <p className={styles.lede}>
                In 2013 a newspaper quiz asked{" "}
                {int.format(constants.nQuestions)} questions and guessed where
                you grew up. Almost everyone who played remembers it. The data
                underneath it was never released. Only pictures of it were.
              </p>
              <button
                className={styles.primary}
                onClick={restart}
                disabled={!payload}
              >
                {payload
                  ? `Start the ${int.format(constants.nQuestions)} questions`
                  : "Loading…"}
              </button>
              <p className={styles.reassure}>
                Runs in this tab. No answer is sent anywhere.
              </p>
            </div>
          ) : null}

          {stage === "playing" ? (
            <div className={styles.play}>
              <div className={styles.progress}>
                <span className={styles.count}>
                  {step + 1} <span className={styles.of}>of</span>{" "}
                  {asked.length}
                </span>
                <div className={styles.bar}>
                  <div
                    className={styles.fill}
                    style={{ width: `${(step / asked.length) * 100}%` }}
                  />
                </div>
              </div>
              <h3 className={styles.question}>{question.text}</h3>
              <div className={styles.choices}>
                {question.choices.map((c) => (
                  <button
                    key={c.id}
                    className={styles.choice}
                    onClick={() => void answer(c.id)}
                    disabled={busy}
                  >
                    {c.text}
                  </button>
                ))}
              </div>
              <button className={styles.skip} onClick={advance} disabled={busy}>
                None of these
              </button>
            </div>
          ) : null}

          {stage === "result" && result ? (
            <div className={styles.result}>
              <p className={styles.kicker}>{lede}</p>
              <h3 className={styles.place}>{result.places[0].name}</h3>
              <dl className={styles.stats}>
                {rows.map((r) => (
                  <div key={r.label}>
                    <dt>{r.label}</dt>
                    <dd>{r.value}</dd>
                  </div>
                ))}
              </dl>
              <p className={styles.honest}>{said}</p>

              <Share text={text} buildCard={buildCard} fileName={CARD_FILE}>
                <p>
                  Nothing about this game is recorded. The page is a static
                  file with nowhere to send it, which is why the accuracy above
                  was measured on simulated speakers and not on the people who
                  play this —{" "}
                  <a href="#limits">what that does and does not establish</a>.
                </p>
              </Share>

              <button className={styles.secondary} onClick={restart}>
                Play again
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
