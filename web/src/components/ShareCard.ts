/**
 * The player's result, drawn as a card they can keep.
 *
 * This is a second renderer of the same posterior, not a screenshot of the
 * first. Map.tsx draws into a canvas that is sized by its container, styled by
 * the surrounding CSS and cropped by the layout; lifting a shareable picture
 * out of it would mean inheriting whatever width the reader's window happened
 * to be, with no title, no scale and no caption. A card that leaves the page
 * has to carry its own context, because the place it lands will not supply it.
 *
 * Everything that differs between the two renderers is framing. The pixels of
 * the map itself -- the ramp, the gamma lift, the borders derived from the
 * cells, the marker -- come from the same modules Map.tsx uses, so the map on
 * the card is the map on the page.
 *
 * Colour is read from tokens.css at draw time, which means the card follows
 * the reader's light or dark theme. That is deliberate: the alternative is a
 * second palette hardcoded here, and a second copy of the palette is exactly
 * the kind of duplicate this project spends a check.py rule preventing.
 */

import type { Cells } from "../model/payload";
import { buildBorders, displayAspect, project } from "../model/geometry";
import { buildRamp, inkColour } from "../model/ramp";

export interface CardRow {
  label: string;
  value: string;
}

export interface CardSpec {
  cells: Cells;
  posterior: Float64Array;
  marker: { lat: number; lon: number };
  /** Small tracked capitals at the top. */
  kicker: string;
  /** One line of prose under the kicker, saying what this is. */
  strap: string;
  /** Caption under the map, explaining how to read it. */
  mapNote: string;
  /** Label above the headline, e.g. how many answers produced it. */
  headingLabel: string;
  heading: string;
  rows: CardRow[];
  /** What the guess is and is not. This is not optional. */
  honest: string;
  source: string;
}

/* Card geometry, in card units. Every literal here stays well under a
 * thousand so that none of them can ever collide with a published figure --
 * see check_site_quotes_no_typed_numbers in check.py. The pixel size is
 * SCALE times this, applied as a transform rather than baked into the numbers. */
const SCALE = 2;
const W = 640;
const PAD = 34;
const INNER = W - PAD * 2;
const LABEL_COL = 124;
const GAMMA = 0.4;

const TYPE = {
  kicker: 11,
  strap: 13,
  mapNote: 10,
  headingLabel: 10,
  heading: 34,
  rowLabel: 10,
  rowValue: 13,
  honest: 11,
  source: 10,
};

function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function wrap(ctx: CanvasRenderingContext2D, text: string, max: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > max) {
      out.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) out.push(line);
  return out;
}

/**
 * The posterior, at whatever size the card gives it.
 *
 * Same two-step as Map.tsx: write one ImageData at grid resolution, then let
 * the browser resample it up. Drawing 50,888 cells as rectangles would be
 * slower and would leave seams between them.
 */
function drawMap(
  ctx: CanvasRenderingContext2D,
  spec: CardSpec,
  x: number,
  y: number,
  w: number,
  h: number,
) {
  const { cells, posterior } = spec;
  const { rows, cols } = cells;
  const ramp = buildRamp();
  const img = new ImageData(cols, rows);
  const px = img.data;

  let max = 0;
  for (let i = 0; i < posterior.length; i++) {
    if (posterior[i] > max) max = posterior[i];
  }
  for (let i = 0; i < cells.count; i++) {
    const v = max > 0 ? (posterior[i] / max) ** GAMMA : 0;
    const s = Math.min(255, Math.max(0, Math.round(v * 255))) * 3;
    const o = (cells.cellY[i] * cols + cells.cellX[i]) * 4;
    px[o] = ramp[s];
    px[o + 1] = ramp[s + 1];
    px[o + 2] = ramp[s + 2];
    px[o + 3] = 255;
  }

  const small = document.createElement("canvas");
  small.width = cols;
  small.height = rows;
  small.getContext("2d")!.putImageData(img, 0, 0);

  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, x, y, w, h);

  const borders = buildBorders(cells);
  const sx = w / cols;
  const sy = h / rows;
  const line = (seg: Float32Array, width: number, colour: string) => {
    ctx.beginPath();
    for (let i = 0; i < seg.length; i += 4) {
      ctx.moveTo(x + seg[i] * sx, y + seg[i + 1] * sy);
      ctx.lineTo(x + seg[i + 2] * sx, y + seg[i + 3] * sy);
    }
    ctx.lineWidth = width;
    ctx.strokeStyle = colour;
    ctx.stroke();
  };
  line(borders.state, 0.4, inkColour(0.22));
  line(borders.coast, 0.7, inkColour(0.5));

  const p = project(cells, spec.marker.lat, spec.marker.lon);
  ctx.beginPath();
  ctx.arc(x + p.x * sx, y + p.y * sy, 4, 0, Math.PI * 2);
  ctx.fillStyle = token("--accent");
  ctx.fill();
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = token("--paper");
  ctx.stroke();
  ctx.restore();
}

/** Renders the card and returns the canvas holding it. */
export function renderCard(spec: CardSpec): HTMLCanvasElement {
  const paper = token("--paper");
  const ink = token("--ink");
  const ink2 = token("--ink-2");
  const ink3 = token("--ink-3");
  const rule = token("--rule");
  const ruleStrong = token("--rule-strong");
  const prose = token("--font-prose");
  const ui = token("--font-ui");
  const data = token("--font-data");

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  const face = (weight: string, size: number, family: string) => {
    ctx.font = `${weight} ${size}px ${family}`;
  };

  // Measuring pass. The honest paragraph wraps to an unknown number of lines,
  // so the card's height is a result rather than a constant.
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  face("400", TYPE.honest, ui);
  const honestLines = wrap(ctx, spec.honest, INNER);
  face("400", TYPE.strap, prose);
  const strapLines = wrap(ctx, spec.strap, INNER);
  face("400", TYPE.heading, prose);
  const headingLines = wrap(ctx, spec.heading, INNER);

  const mapW = INNER;
  const mapH = Math.round(mapW / displayAspect(spec.cells));
  const honestH = honestLines.length * (TYPE.honest + 5);
  const strapH = strapLines.length * (TYPE.strap + 4);
  const headingH = headingLines.length * (TYPE.heading + 4);
  const rowsH = spec.rows.length * 26;
  const H =
    PAD + 16 + strapH + 20 + mapH + 8 + 16 + 20 + 16 + headingH + 12 + rowsH +
    20 + honestH + 8 + 16 + PAD;

  canvas.width = W * SCALE;
  canvas.height = Math.round(H) * SCALE;
  ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
  ctx.textBaseline = "top";

  ctx.fillStyle = paper;
  ctx.fillRect(0, 0, W, H);

  let y = PAD;

  const tracked = (on: boolean) => {
    // Supported in current Chromium and Safari; older engines ignore it and
    // simply draw untracked capitals, which is a legible fallback.
    if ("letterSpacing" in ctx) ctx.letterSpacing = on ? "0.09em" : "0px";
  };

  face("600", TYPE.kicker, ui);
  tracked(true);
  ctx.fillStyle = ink3;
  ctx.fillText(spec.kicker.toUpperCase(), PAD, y);
  tracked(false);
  y += 16;

  face("400", TYPE.strap, prose);
  ctx.fillStyle = ink2;
  for (const l of strapLines) {
    ctx.fillText(l, PAD, y);
    y += TYPE.strap + 4;
  }

  y += 20 - 4;
  drawMap(ctx, spec, PAD, y, mapW, mapH);
  ctx.strokeStyle = rule;
  ctx.lineWidth = 0.5;
  ctx.strokeRect(PAD, y, mapW, mapH);
  y += mapH + 8;

  face("400", TYPE.mapNote, ui);
  ctx.fillStyle = ink3;
  ctx.fillText(spec.mapNote, PAD, y);
  y += 16 + 20;

  ctx.beginPath();
  ctx.moveTo(PAD, y);
  ctx.lineTo(W - PAD, y);
  ctx.strokeStyle = ruleStrong;
  ctx.lineWidth = 0.5;
  ctx.stroke();
  y += 16;

  face("600", TYPE.headingLabel, ui);
  tracked(true);
  ctx.fillStyle = ink3;
  ctx.fillText(spec.headingLabel.toUpperCase(), PAD, y);
  tracked(false);
  y += 16;

  face("400", TYPE.heading, prose);
  ctx.fillStyle = ink;
  for (const l of headingLines) {
    ctx.fillText(l, PAD, y);
    y += TYPE.heading + 4;
  }
  y += 12;

  for (const r of spec.rows) {
    ctx.beginPath();
    ctx.moveTo(PAD, y);
    ctx.lineTo(W - PAD, y);
    ctx.strokeStyle = rule;
    ctx.lineWidth = 0.5;
    ctx.stroke();
    face("600", TYPE.rowLabel, ui);
    tracked(true);
    ctx.fillStyle = ink3;
    ctx.fillText(r.label.toUpperCase(), PAD, y + 8);
    tracked(false);
    face("400", TYPE.rowValue, data);
    ctx.fillStyle = ink;
    ctx.fillText(r.value, PAD + LABEL_COL, y + 6, INNER - LABEL_COL);
    y += 26;
  }
  ctx.beginPath();
  ctx.moveTo(PAD, y);
  ctx.lineTo(W - PAD, y);
  ctx.strokeStyle = rule;
  ctx.lineWidth = 0.5;
  ctx.stroke();

  y += 20;
  face("400", TYPE.honest, ui);
  ctx.fillStyle = ink2;
  for (const l of honestLines) {
    ctx.fillText(l, PAD, y);
    y += TYPE.honest + 5;
  }

  y += 8;
  face("400", TYPE.source, data);
  ctx.fillStyle = ink3;
  ctx.fillText(spec.source, PAD, y);

  ctx.strokeStyle = ruleStrong;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  return canvas;
}

export function toPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("the card could not be encoded"))),
      "image/png",
    );
  });
}
