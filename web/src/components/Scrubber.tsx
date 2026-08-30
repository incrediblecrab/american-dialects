import { useCallback, useEffect, useRef, useState } from "react";
import { buildRamp } from "../model/ramp";
import type { RecoveryStage } from "../model/types";
import styles from "./Scrubber.module.css";

interface Props {
  stages: RecoveryStage[];
  /** Directory under public/data holding the stage files. */
  dir: string;
  /** Provenance prefix. The native pixel size is measured and appended. */
  sourceLabel?: string;
  gamma?: number;
}

const BASE = import.meta.env.BASE_URL;

function reducedMotion() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * Four pictures of the same answer, one control to move between them.
 *
 * The single-channel stages are drawn through the same ramp as the live map,
 * so the recovery visibly ends where the quiz begins. Stage 0 is the published
 * GIF and is drawn as it is; recolouring somebody else's map would be a
 * misrepresentation of the source. It is mounted rather than stretched,
 * because it is a 456x200 artefact and there is no more resolution to find.
 *
 * Moving between stages crossfades. That is the one piece of motion this
 * component is entitled to: the claim of the section is that ink becomes a
 * surface, and a hard cut asserts it where a crossfade shows it.
 */
export default function Scrubber({
  stages,
  dir,
  sourceLabel,
  gamma = 0.55,
}: Props) {
  const [at, setAt] = useState(0);
  const [ghost, setGhost] = useState<number | null>(null);
  const [ghostOut, setGhostOut] = useState(false);
  const curRef = useRef<HTMLCanvasElement>(null);
  const ghostRef = useRef<HTMLCanvasElement>(null);
  const [images, setImages] = useState<ImageBitmap[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;
    Promise.all(
      stages.map(async (s) => {
        const res = await fetch(`${BASE}data/${dir}/${s.file}`);
        if (!res.ok) throw new Error(`${s.file}: ${res.status}`);
        return createImageBitmap(await res.blob());
      }),
    ).then(
      (bm) => (live ? setImages(bm) : bm.forEach((b) => b.close())),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [stages, dir]);

  const paint = useCallback(
    (canvas: HTMLCanvasElement | null, i: number) => {
      if (!canvas || !images) return;
      const bm = images[i];
      canvas.width = bm.width;
      canvas.height = bm.height;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;
      ctx.drawImage(bm, 0, 0);
      if (stages[i].kind === "rgb") return;

      const img = ctx.getImageData(0, 0, bm.width, bm.height);
      const px = img.data;
      const ramp = buildRamp();
      for (let k = 0; k < px.length; k += 4) {
        const v = (px[k] / 255) ** gamma;
        const s = Math.min(255, Math.round(v * 255)) * 3;
        px[k] = ramp[s];
        px[k + 1] = ramp[s + 1];
        px[k + 2] = ramp[s + 2];
        px[k + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    },
    [images, stages, gamma],
  );

  useEffect(() => {
    paint(curRef.current, at);
  }, [paint, at]);

  useEffect(() => {
    if (ghost === null) return;
    paint(ghostRef.current, ghost);
    const raf = requestAnimationFrame(() => setGhostOut(true));
    return () => cancelAnimationFrame(raf);
  }, [paint, ghost]);

  function select(i: number) {
    if (i === at) return;
    if (images && !reducedMotion()) {
      setGhost(at);
      setGhostOut(false);
    }
    setAt(i);
  }

  const stage = stages[at];
  const native = images?.[0];
  const source =
    sourceLabel && native
      ? `All four stages derive from ${sourceLabel} published at ${native.width} × ${native.height} pixels. No larger version exists.`
      : null;

  return (
    <figure className={styles.figure}>
      <div className={styles.mount}>
        <div className={styles.plate}>
          {images ? (
            <>
              <canvas
                ref={curRef}
                className={`${styles.canvas} ${
                  stage.kind === "rgb" ? styles.pixels : styles.smooth
                }`}
              />
              {ghost !== null && (
                <canvas
                  ref={ghostRef}
                  aria-hidden="true"
                  className={`${styles.canvas} ${styles.ghost} ${
                    ghostOut ? styles.ghostOut : ""
                  } ${
                    stages[ghost].kind === "rgb"
                      ? styles.pixels
                      : styles.smooth
                  }`}
                  onTransitionEnd={() => setGhost(null)}
                />
              )}
              <span className={styles.tag}>
                <span className={styles.tagN}>
                  {at + 1}/{stages.length}
                </span>
                {stage.name}
              </span>
            </>
          ) : (
            <div className={styles.loading}>
              {failed ? "Recovery images unavailable." : "Loading…"}
            </div>
          )}
        </div>
      </div>

      <div className={styles.controls} role="group" aria-label="Recovery stage">
        {stages.map((s, i) => (
          <button
            key={s.file}
            className={i === at ? styles.stepOn : styles.step}
            onClick={() => select(i)}
            aria-current={i === at}
          >
            <span className={styles.n}>{i + 1}</span>
            {s.name}
          </button>
        ))}
      </div>

      <figcaption className={styles.caption}>
        {stage.note}
        {source && <span className={styles.source}>{source}</span>}
      </figcaption>
    </figure>
  );
}
