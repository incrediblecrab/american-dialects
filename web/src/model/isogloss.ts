/**
 * What an isogloss is, computed from two surfaces.
 *
 * A single answer's surface is a heat map: it says where a word is common. An
 * isogloss is a statement about two words at once -- the line where one
 * overtakes the other -- so it needs both, and it needs them as a difference
 * rather than as two pictures side by side.
 *
 * The surfaces the exporter ships are log P(answer | place), so subtracting
 * one from the other gives log-odds of the first word against the second,
 * given that the speaker used one of them. Where that field is zero, the two
 * words are equally likely, and that contour is the isogloss. Everything else
 * here follows from that one field: its sign is which side you are on, and its
 * magnitude is how firmly.
 */

import type { Cells } from "./payload";
import type { RGB } from "./ramp";

/** Signed log-odds of the first variant against the second, per land cell. */
export function logOdds(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

/**
 * Per-cell RGB, count * 3, for the map fill.
 *
 * Colour is which word leads; how much colour is by how much. At equal odds a
 * cell is left as bare paper, so the boundary appears as the page showing
 * through the ink rather than as something drawn on top of it. That is the
 * whole argument of this section in one rule: a sharp isogloss is a thin pale
 * seam, a fuzzy one is a broad washed-out band, and both are the same
 * rendering of the same quantity.
 *
 * `flat` collapses that gradient toward the way an atlas prints it: solid
 * colour either side, no admission that anything is uncertain. Holding the two
 * against each other is the point of the control that drives it.
 */
export function shade(
  cells: Cells,
  d: Float32Array,
  flat: number,
  a: RGB,
  b: RGB,
  paper: RGB,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(cells.count * 3);
  for (let i = 0; i < cells.count; i++) {
    const v = d[i];
    const side = v >= 0 ? a : b;
    // |2p - 1| for p the probability of the leading variant, written so that
    // a large |v| cannot overflow the exponential.
    const e = Math.exp(-Math.abs(v));
    const lead = (1 - e) / (1 + e);
    const k = flat + (1 - flat) * lead;
    const o = i * 3;
    out[o] = paper[0] + (side[0] - paper[0]) * k;
    out[o + 1] = paper[1] + (side[1] - paper[1]) * k;
    out[o + 2] = paper[2] + (side[2] - paper[2]) * k;
  }
  return out;
}

/**
 * The zero contour of the field, in the grid coordinates the map draws in.
 *
 * Marching squares over cell centres rather than the cell-edge staircase the
 * state borders use, because a boundary between two words is not a boundary
 * between two cells: it falls somewhere inside the gap, and where inside is
 * exactly what the log-odds either side say. Interpolating for it is the
 * difference between a contour and a flight of stairs.
 *
 * Squares touching the sea are skipped, which clips the line to land for free.
 */
export function contour(cells: Cells, d: Float32Array): Float32Array {
  const { rows, cols } = cells;
  const grid = new Float32Array(rows * cols).fill(NaN);
  for (let i = 0; i < cells.count; i++) grid[cells.cellY[i] * cols + cells.cellX[i]] = d[i];

  const seg: number[] = [];
  const push = (x1: number, y1: number, x2: number, y2: number) =>
    seg.push(x1 + 0.5, y1 + 0.5, x2 + 0.5, y2 + 0.5);

  for (let y = 0; y + 1 < rows; y++) {
    for (let x = 0; x + 1 < cols; x++) {
      const tl = grid[y * cols + x];
      const tr = grid[y * cols + x + 1];
      const bl = grid[(y + 1) * cols + x];
      const br = grid[(y + 1) * cols + x + 1];
      if (Number.isNaN(tl) || Number.isNaN(tr) || Number.isNaN(bl) || Number.isNaN(br)) {
        continue;
      }
      const code =
        (tl >= 0 ? 1 : 0) | (tr >= 0 ? 2 : 0) | (br >= 0 ? 4 : 0) | (bl >= 0 ? 8 : 0);
      if (code === 0 || code === 15) continue;

      const top = x + tl / (tl - tr);
      const bottom = x + bl / (bl - br);
      const left = y + tl / (tl - bl);
      const right = y + tr / (tr - br);

      switch (code) {
        case 1:
        case 14:
          push(x, left, top, y);
          break;
        case 2:
        case 13:
          push(top, y, x + 1, right);
          break;
        case 3:
        case 12:
          push(x, left, x + 1, right);
          break;
        case 4:
        case 11:
          push(x + 1, right, bottom, y + 1);
          break;
        case 6:
        case 9:
          push(top, y, bottom, y + 1);
          break;
        case 7:
        case 8:
          push(x, left, bottom, y + 1);
          break;
        // The two saddles, where the square holds both words twice over.
        // Split it rather than guess which pairing is real.
        case 5:
          push(x, left, top, y);
          push(x + 1, right, bottom, y + 1);
          break;
        case 10:
          push(top, y, x + 1, right);
          push(x, left, bottom, y + 1);
          break;
      }
    }
  }
  return Float32Array.from(seg);
}
