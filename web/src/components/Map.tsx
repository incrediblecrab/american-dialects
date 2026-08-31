import { useEffect, useMemo, useRef } from "react";
import type { Cells } from "../model/payload";
import { buildBorders } from "../model/geometry";
import { buildLookup, makeAlbers } from "../model/albers";
import { buildRamp, inkColour } from "../model/ramp";
import { pickLabels, type Place } from "../model/labels";
import styles from "./Map.module.css";

/**
 * Resolution of the reprojected raster, independent of how big the map is
 * drawn. Fixing it means the lookup table survives every resize and is built
 * once for the life of the component; the canvas then scales this up, which
 * is the same thing the unprojected version did with the raw grid.
 */
const RASTER_W = 912;

export interface Marker {
  lat: number;
  lon: number;
  label?: string;
  tone?: "accent" | "warn" | "ink";
}

/** Extra segments to stroke over the fill, in the same grid coordinates. */
export interface Overlay {
  segments: Float32Array;
  width: number;
  colour: string;
}

interface Props {
  cells: Cells;
  /** Per-cell probability. Null draws the empty country. */
  posterior: Float64Array | null;
  /**
   * Per-cell RGB triples, count * 3. When given, this is the fill and the
   * posterior ramp is not used -- the isogloss map colours by which of two
   * words leads rather than by how much probability sits in a cell.
   */
  shading?: Uint8ClampedArray | null;
  overlays?: Overlay[];
  markers?: Marker[];
  /** Cities to name. Passing the full population-sorted list is correct; the
   * map decides how many of them fit at the width it turns out to have. */
  places?: Place[];
  /** Ends of the scale, e.g. ["less likely", "more likely"]. Omitted on maps
   * whose fill is not a scale, such as the isogloss plates. */
  legend?: [string, string];
  /** Lifts the shoulders of a very peaked posterior into view. */
  gamma?: number;
  caption?: string;
  className?: string;
}

/**
 * The posterior, drawn.
 *
 * Canvas rather than SVG or a mapping library: there are 50,888 cells, which
 * is 50,888 DOM nodes as SVG and a dependency plus a tile server as a map.
 * As a raster it is one ImageData write and it redraws in about a millisecond,
 * which is what makes the question-by-question animation affordable.
 *
 * The raw posterior is far too peaked to look at. One cell can hold a thousand
 * times another, and drawn linearly the map reads as a single dot on an empty
 * country. Dividing by the maximum and raising to a fractional power lifts the
 * shoulders back into view without changing the ordering, so the picture shows
 * the shape of the belief rather than only its argmax. This matches the
 * rendering in site/server.py, so the published map and the research tool draw
 * the same posterior the same way.
 *
 * A caller that has already decided what colour every cell should be passes
 * `shading` instead and skips all of that. The projection, the coastline and
 * the device-pixel handling are the same either way, which is the reason
 * there is one map component here rather than two.
 */
export default function Map({
  cells,
  posterior,
  shading = null,
  overlays = [],
  markers = [],
  places = [],
  legend,
  gamma = 0.4,
  caption,
  className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const albers = useMemo(() => makeAlbers(cells), [cells]);
  const aspect = albers.aspect;
  const raster = useMemo(() => {
    const h = Math.round(RASTER_W / albers.aspect);
    return { w: RASTER_W, h, lut: buildLookup(cells, albers, RASTER_W, h) };
  }, [cells, albers]);

  /**
   * Borders, projected once into [0,1] and kept there.
   *
   * Their geometry never changes, only the size they are drawn at, so putting
   * the trigonometry in a memo means a redraw is a multiply per endpoint
   * rather than a projection per endpoint.
   */
  const borders = useMemo(() => {
    const grid = buildBorders(cells);
    const conv = (seg: Float32Array) => {
      const out = new Float32Array(seg.length);
      for (let i = 0; i < seg.length; i += 2) {
        const p = albers.fromGrid(seg[i], seg[i + 1]);
        out[i] = p.x;
        out[i + 1] = p.y;
      }
      return out;
    };
    return { state: conv(grid.state), coast: conv(grid.coast) };
  }, [cells, albers]);

  /** Callers hand overlays in grid coordinates, same as the borders were. */
  const projectedOverlays = useMemo(
    () =>
      overlays.map((o) => {
        const out = new Float32Array(o.segments.length);
        for (let i = 0; i < o.segments.length; i += 2) {
          const p = albers.fromGrid(o.segments[i], o.segments[i + 1]);
          out[i] = p.x;
          out[i + 1] = p.y;
        }
        return { ...o, segments: out };
      }),
    [overlays, albers],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const draw = () => {
      const cssW = wrap.clientWidth;
      const cssH = cssW / aspect;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      canvas.style.height = `${cssH}px`;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const { w, h, lut } = raster;
      const img = new ImageData(w, h);
      const px = img.data;

      if (shading) {
        for (let p = 0; p < lut.length; p++) {
          const i = lut[p];
          if (i < 0) continue;
          const o = p * 4;
          px[o] = shading[i * 3];
          px[o + 1] = shading[i * 3 + 1];
          px[o + 2] = shading[i * 3 + 2];
          px[o + 3] = 255;
        }
      } else {
        const ramp = buildRamp();
        let max = 0;
        if (posterior) {
          for (let i = 0; i < posterior.length; i++) {
            if (posterior[i] > max) max = posterior[i];
          }
        }
        // Resolve each cell's ramp position once. Several output pixels fall
        // in the same cell, and the power is far more expensive than a lookup.
        const idx = new Uint8Array(cells.count);
        if (posterior && max > 0) {
          for (let i = 0; i < cells.count; i++) {
            const v = (posterior[i] / max) ** gamma;
            idx[i] = Math.min(255, Math.max(0, Math.round(v * 255)));
          }
        }
        for (let p = 0; p < lut.length; p++) {
          const i = lut[p];
          if (i < 0) continue;
          const s = idx[i] * 3;
          const o = p * 4;
          px[o] = ramp[s];
          px[o + 1] = ramp[s + 1];
          px[o + 2] = ramp[s + 2];
          px[o + 3] = 255;
        }
      }

      const small = document.createElement("canvas");
      small.width = w;
      small.height = h;
      small.getContext("2d")!.putImageData(img, 0, 0);

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(small, 0, 0, canvas.width, canvas.height);

      const sx = canvas.width;
      const sy = canvas.height;
      const line = (seg: Float32Array, width: number, colour: string) => {
        ctx.beginPath();
        for (let i = 0; i < seg.length; i += 4) {
          ctx.moveTo(seg[i] * sx, seg[i + 1] * sy);
          ctx.lineTo(seg[i + 2] * sx, seg[i + 3] * sy);
        }
        ctx.lineWidth = width * dpr;
        ctx.strokeStyle = colour;
        ctx.stroke();
      };
      line(borders.state, 0.5, inkColour(0.22));
      line(borders.coast, 0.9, inkColour(0.5));
      for (const o of projectedOverlays) line(o.segments, o.width, o.colour);

      /**
       * City names.
       *
       * How many fit is a property of the rendered width, not of the data, so
       * the choice is made here rather than by the caller: the same map in a
       * phone column and in a full-bleed figure wants seven names and sixteen.
       * Text is stroked in the paper colour before it is filled, which is the
       * cheap version of a halo and the only thing that keeps a name legible
       * where it crosses from pale fill onto the dark end of the ramp.
       */
      if (places.length) {
        const narrow = cssW < 380;
        const mid = cssW < 560;
        const css = getComputedStyle(document.documentElement);
        const fontSize = Math.max(9, Math.min(12.5, cssW / 48)) * dpr;
        ctx.font = `500 ${fontSize}px ${css.getPropertyValue("--font-ui").trim()}`;
        ctx.textBaseline = "middle";
        ctx.lineJoin = "round";
        const chosen = pickLabels(
          places,
          albers.forward,
          narrow ? 7 : mid ? 11 : 16,
          narrow ? 0.16 : mid ? 0.115 : 0.085,
          narrow ? 0.08 : mid ? 0.06 : 0.046,
          markers.map((m) => albers.forward(m.lat, m.lon)),
        );
        const dot = Math.max(1.4, fontSize * 0.16);
        const gap = dot * 2.4;
        const paper = css.getPropertyValue("--paper").trim();
        for (const l of chosen) {
          const x = l.x * sx;
          const y = l.y * sy;
          ctx.beginPath();
          ctx.arc(x, y, dot, 0, Math.PI * 2);
          ctx.fillStyle = inkColour(0.55);
          ctx.fill();
          // Flip to the left rather than run a name off the edge. Measured,
          // not guessed from the position: "Jacksonville" and "Reno" overflow
          // from very different places.
          const width = ctx.measureText(l.name).width;
          const right = x + gap + width < sx - 4 * dpr;
          ctx.textAlign = right ? "left" : "right";
          const tx = right ? x + gap : x - gap;
          ctx.lineWidth = 3 * dpr;
          ctx.strokeStyle = paper;
          ctx.strokeText(l.name, tx, y);
          ctx.fillStyle = inkColour(0.75);
          ctx.fillText(l.name, tx, y);
        }
      }

      for (const m of markers) {
        const p = albers.forward(m.lat, m.lon);
        const x = p.x * sx;
        const y = p.y * sy;
        const r = 4.5 * dpr;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = getComputedStyle(document.documentElement)
          .getPropertyValue(
            m.tone === "warn" ? "--warn" : m.tone === "ink" ? "--ink" : "--accent",
          )
          .trim();
        ctx.fill();
        ctx.lineWidth = 1.6 * dpr;
        ctx.strokeStyle = getComputedStyle(document.documentElement)
          .getPropertyValue("--paper")
          .trim();
        ctx.stroke();
      }
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [
    cells,
    posterior,
    shading,
    projectedOverlays,
    markers,
    places,
    gamma,
    borders,
    aspect,
    albers,
    raster,
  ]);

  return (
    <figure className={`${styles.figure} ${className ?? ""}`}>
      <div ref={wrapRef} className={styles.wrap}>
        <canvas ref={canvasRef} className={styles.canvas} />
      </div>
      {legend ? (
        <div className={styles.legend}>
          <span>{legend[0]}</span>
          <span className={styles.ramp} aria-hidden="true" />
          <span>{legend[1]}</span>
        </div>
      ) : null}
      {caption ? <figcaption className={styles.caption}>{caption}</figcaption> : null}
    </figure>
  );
}
