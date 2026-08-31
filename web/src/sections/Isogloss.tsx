import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import MapView from "../components/Map";
import type { Marker, Overlay } from "../components/Map";
import { usePayload } from "../model/usePayload";
import { loadSurface } from "../model/payload";
import { contour, logOdds, shade } from "../model/isogloss";
import { cssColour, parseRGB, rgba } from "../model/ramp";
import type { RGB } from "../model/ramp";
import { isoglosses, km, oneDp, pct } from "../content";
import s from "./Section.module.css";
import styles from "./Isogloss.module.css";

const TWEEN_MS = 420;

/**
 * The payoff of the recovery: a boundary, drawn from the pixels.
 *
 * Every other map on this page is a posterior over where one person grew up.
 * This one is a statement about the country: given that a speaker uses one of
 * two words, which one, and where does the answer change. It is the object
 * dialect geography was invented to draw, and it is the thing a table of state
 * percentages can never show, because the interesting lines run through states
 * rather than between them.
 *
 * The only motion is the field itself easing from one contrast to the next and
 * between the two ways of drawing it. Both are the argument: a boundary that
 * barely moves when the gradient is restored was always sharp, and one that
 * dissolves was never really a line.
 */
export default function Isogloss() {
  const { payload } = usePayload();
  const list = isoglosses.contrasts;
  /**
   * Open on the contrast that actually divides the country.
   *
   * The list is ordered by boundary width, and width is inversely related to
   * how lopsided a contrast is: the sharpest lines are sharp because one word
   * holds a small pocket. Opening on the first entry meant opening on yinz,
   * which puts 0.7% of the country on one side, so the map painted as a flat
   * wash with a dot in western Pennsylvania. That is a true picture and a
   * useless first impression, and no choice of palette repairs it, because the
   * fault is which surface is on screen rather than what colour it is.
   * Choosing by share rather than naming an id keeps this correct if the
   * published set ever changes.
   */
  const [at, setAt] = useState(() =>
    list.reduce(
      (best, c, i) =>
        Math.abs(c.shareA - 0.5) < Math.abs(list[best].shareA - 0.5) ? i : best,
      0,
    ),
  );
  const [flat, setFlat] = useState(false);
  const [failed, setFailed] = useState(false);
  const [tick, redraw] = useReducer((n: number) => n + 1, 0);
  const iso = list[at];

  const field = useRef<Float32Array | null>(null);
  const flatness = useRef(0);
  const inkA = useRef<HTMLSpanElement>(null);
  const inkB = useRef<HTMLSpanElement>(null);
  const [target, setTarget] = useState<Float32Array | null>(null);

  useEffect(() => {
    if (!payload) return;
    let live = true;
    Promise.all([
      loadSurface(payload.manifest, payload.cells, iso.question, iso.a.choice),
      loadSurface(payload.manifest, payload.cells, iso.question, iso.b.choice),
    ]).then(
      ([a, b]) => live && setTarget(logOdds(a, b)),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [payload, iso]);

  useEffect(() => {
    if (!target) return;
    const to = flat ? 1 : 0;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!field.current || field.current.length !== target.length || still) {
      field.current = Float32Array.from(target);
      flatness.current = to;
      redraw();
      return;
    }
    const was = Float32Array.from(field.current);
    const wasFlat = flatness.current;
    const t0 = performance.now();
    let raf = requestAnimationFrame(function step(now: number) {
      const u = Math.min(1, (now - t0) / TWEEN_MS);
      const e = 1 - (1 - u) ** 3;
      const cur = field.current!;
      for (let i = 0; i < cur.length; i++) cur[i] = was[i] + (target[i] - was[i]) * e;
      flatness.current = wasFlat + (to - wasFlat) * e;
      redraw();
      if (u < 1) raf = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(raf);
  }, [target, flat]);

  const view = useMemo(() => {
    const d = field.current;
    if (!payload || !d) return null;
    const k = flatness.current;
    const shading = shade(
      payload.cells,
      d,
      k,
      swatch(inkA.current, "--iso-a"),
      swatch(inkB.current, "--iso-b"),
      cssColour("--paper"),
    );
    const overlays: Overlay[] = [
      {
        segments: contour(payload.cells, d),
        width: 0.7 + 0.8 * k,
        colour: rgba(cssColour("--ink"), 0.5 + 0.4 * k),
      },
    ];
    return { shading, overlays };
    // tick advances once per animation frame.
  }, [payload, tick]);

  const markers = useMemo<Marker[]>(
    () =>
      iso.anchors.map((a, i) => ({
        lat: a.lat,
        lon: a.lon,
        tone: i === 0 ? "accent" : "ink",
      })),
    [iso],
  );

  const warm = (i: number) => {
    if (!payload) return;
    const c = list[i];
    void loadSurface(payload.manifest, payload.cells, c.question, c.a.choice).catch(
      () => {},
    );
    void loadSurface(payload.manifest, payload.cells, c.question, c.b.choice).catch(
      () => {},
    );
  };

  const tightest = list[0];
  const widest = list[list.length - 1];

  return (
    <section className={s.section} id="isoglosses">
      <div className={s.wide}>
        <div className={styles.picker} role="group" aria-label="Contrast">
          {list.map((c, i) => (
            <button
              key={c.id}
              className={i === at ? styles.wordOn : styles.word}
              aria-pressed={i === at}
              onClick={() => setAt(i)}
              onPointerEnter={() => warm(i)}
              onFocus={() => warm(i)}
            >
              {c.a.label}
              <span className={styles.wordKm}>{Math.round(c.widthKm)}</span>
            </button>
          ))}
        </div>

        <div className={styles.panel}>
          <div className={styles.mapSide}>
            {payload && view ? (
              <MapView
                cells={payload.cells}
                posterior={null}
                shading={view.shading}
                overlays={view.overlays}
                markers={markers}
                caption={iso.note}
              />
            ) : (
              <div className={styles.skeleton}>
                {failed
                  ? "The surfaces for this contrast could not be loaded."
                  : "Loading the surfaces…"}
              </div>
            )}

            <div className={styles.toggle} role="group" aria-label="How to draw it">
              <button
                className={flat ? styles.mode : styles.modeOn}
                aria-pressed={!flat}
                onClick={() => setFlat(false)}
              >
                As it was measured
              </button>
              <button
                className={flat ? styles.modeOn : styles.mode}
                aria-pressed={flat}
                onClick={() => setFlat(true)}
              >
                As an atlas draws it
              </button>
            </div>
            <p className={styles.toggleNote}>
              {flat
                ? "Solid either side of the line, which is how every published isogloss you have ever seen was printed."
                : "Colour drains where the odds are even, so the boundary appears as the bare ground showing through the ink."}
            </p>
          </div>

          <div className={styles.readout}>
            <dl className={styles.legend}>
              <div>
                <dt>
                  <span className={styles.swatchA} ref={inkA} aria-hidden="true" />
                  {iso.a.label}
                </dt>
                <dd>{oneDp.format(iso.a.national)}%</dd>
              </div>
              <div>
                <dt>
                  <span className={styles.swatchB} ref={inkB} aria-hidden="true" />
                  {iso.b.label}
                </dt>
                <dd>{oneDp.format(iso.b.national)}%</dd>
              </div>
            </dl>
            <p className={styles.legendNote}>
              Share of respondents nationally, as the survey published it.
              {iso.a.label !== iso.a.survey || iso.b.label !== iso.b.survey ? (
                <>
                  {" "}
                  The survey worded them <em>{iso.a.survey}</em> and{" "}
                  <em>{iso.b.survey}</em>.
                </>
              ) : null}
            </p>

            <div className={styles.headline}>
              <span className={styles.big}>{km(iso.widthKm)}</span>
              <span className={styles.bigLabel}>
                from {isoglosses.odds} to 1 for {iso.a.label} to{" "}
                {isoglosses.odds} to 1 for {iso.b.label}
              </span>
            </div>

            <dl className={styles.stats}>
              <div>
                <dt>Boundary, on land</dt>
                <dd>{km(iso.lineKm)}</dd>
              </div>
              <div>
                <dt>Country on the {iso.a.label} side</dt>
                <dd>{pct(iso.shareA, 1)}</dd>
              </div>
              {iso.anchors.map((a, i) => (
                <div key={a.name}>
                  <dt>
                    {a.name}, {a.state}
                  </dt>
                  <dd className={i === 0 ? styles.valA : styles.valB}>
                    {i === 0 ? iso.a.label : iso.b.label} {share(i === 0 ? a.p : 1 - a.p)}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </div>

      <div className={s.body}>
        <p>
          The number worth watching is the width. A boundary is not a line: it
          is the ground over which the odds turn over, and that ground is{" "}
          <span className={s.stat}>{km(tightest.widthKm)}</span> wide for{" "}
          <em>{tightest.a.label}</em> and{" "}
          <span className={s.stat}>{km(widest.widthKm)}</span> wide for{" "}
          <em>{widest.a.label}</em>. The same drawing convention flatters both
          of them equally. Switching to <em>as an atlas draws it</em> is
          therefore a different lie in each case: for{" "}
          <em>{tightest.a.label}</em> the printed line is very nearly true, and
          for <em>{widest.a.label}</em> it invents a border where the country
          has only a preference that drifts.
        </p>
        <p className={s.note}>
          Two limits. These widths are properties of the fitted surfaces, and
          those surfaces are smoothed at a bandwidth chosen by
          cross-validation — a boundary cannot come out of a smoothed surface
          sharper than the smoothing left it, so read a width as an upper bound
          on how abrupt the real thing is. And the recovery is validated only
          at the state level, so a line drawn inside Pennsylvania is exactly
          the claim this project cannot yet check.{" "}
          <a href="/american-dialects/limits/">That is taken up at the end</a>.
        </p>
      </div>
    </section>
  );
}

function share(p: number): string {
  if (p >= 0.99) return "over 99%";
  if (p <= 0.01) return "under 1%";
  return pct(p, 0);
}

/* The legend swatches are the source of truth for the two map inks: their
 * computed background is what the fill is painted with, so a change to the
 * stylesheet moves both together and they cannot drift apart. */
function swatch(el: HTMLElement | null, fallback: string): RGB {
  if (!el) return cssColour(fallback);
  const mixed = parseRGB(getComputedStyle(el).backgroundColor);
  return mixed ?? cssColour(fallback);
}
