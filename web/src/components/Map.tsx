import { useEffect, useMemo, useRef } from "react";
import type { Cells } from "../model/payload";
import { buildBorders, displayAspect, project } from "../model/geometry";
import { buildRamp, inkColour } from "../model/ramp";
import styles from "./Map.module.css";

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
  gamma = 0.4,
  caption,
  className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const borders = useMemo(() => buildBorders(cells), [cells]);
  const aspect = useMemo(() => displayAspect(cells), [cells]);

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

      const { rows, cols } = cells;
      const img = new ImageData(cols, rows);
      const px = img.data;

      if (shading) {
        for (let i = 0; i < cells.count; i++) {
          const o = (cells.cellY[i] * cols + cells.cellX[i]) * 4;
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
        for (let i = 0; i < cells.count; i++) {
          const v = posterior && max > 0 ? (posterior[i] / max) ** gamma : 0;
          const s = Math.min(255, Math.max(0, Math.round(v * 255))) * 3;
          const o = (cells.cellY[i] * cols + cells.cellX[i]) * 4;
          px[o] = ramp[s];
          px[o + 1] = ramp[s + 1];
          px[o + 2] = ramp[s + 2];
          px[o + 3] = 255;
        }
      }

      const small = document.createElement("canvas");
      small.width = cols;
      small.height = rows;
      small.getContext("2d")!.putImageData(img, 0, 0);

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(small, 0, 0, canvas.width, canvas.height);

      const sx = canvas.width / cols;
      const sy = canvas.height / rows;
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
      for (const o of overlays) line(o.segments, o.width, o.colour);

      for (const m of markers) {
        const p = project(cells, m.lat, m.lon);
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
  }, [cells, posterior, shading, overlays, markers, gamma, borders, aspect]);

  return (
    <figure className={`${styles.figure} ${className ?? ""}`}>
      <div ref={wrapRef} className={styles.wrap}>
        <canvas ref={canvasRef} className={styles.canvas} />
      </div>
      {caption ? <figcaption className={styles.caption}>{caption}</figcaption> : null}
    </figure>
  );
}
