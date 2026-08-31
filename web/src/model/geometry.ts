/**
 * Coastline and state borders, derived from the cell data already loaded.
 *
 * The alternative is shipping a GeoJSON of US state boundaries, which would be
 * a second, differently sourced geography sitting on top of the first. Every
 * cell already carries the state it belongs to, so the borders are exactly the
 * edges where that value changes, or where land meets everything else. Drawing
 * them this way guarantees the outline agrees with the data underneath it: if
 * a cell is attributed to Ohio, it is inside the Ohio drawn on the map.
 *
 * Segments are in grid coordinates, where (0,0) is the top-left corner of the
 * north-west cell, so a renderer scales them by its pixel size per cell.
 */

import type { Cells } from "./payload";

export interface Borders {
  /** Flat [x1, y1, x2, y2, ...] of edges between two different states. */
  state: Float32Array;
  /** Flat [x1, y1, x2, y2, ...] of edges between land and not-land. */
  coast: Float32Array;
}

const NONE = -1;

export function buildBorders(cells: Cells): Borders {
  const { rows, cols } = cells;
  const grid = new Int16Array(rows * cols).fill(NONE);
  for (let i = 0; i < cells.count; i++) {
    grid[cells.cellY[i] * cols + cells.cellX[i]] = cells.stateIdx[i];
  }

  const state: number[] = [];
  const coast: number[] = [];
  const edge = (a: number, b: number, x1: number, y1: number, x2: number, y2: number) => {
    if (a === b) return;
    if (a === NONE || b === NONE) coast.push(x1, y1, x2, y2);
    else state.push(x1, y1, x2, y2);
  };

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const here = grid[y * cols + x];
      // Right edge of this cell, against the cell east of it.
      const east = x + 1 < cols ? grid[y * cols + x + 1] : NONE;
      edge(here, east, x + 1, y, x + 1, y + 1);
      // Bottom edge, against the cell south of it.
      const south = y + 1 < rows ? grid[(y + 1) * cols + x] : NONE;
      edge(here, south, x, y + 1, x + 1, y + 1);
      // The far west and north sides are only drawn where land starts there.
      if (x === 0) edge(NONE, here, 0, y, 0, y + 1);
      if (y === 0) edge(NONE, here, x, 0, x + 1, 0);
    }
  }
  return { state: Float32Array.from(state), coast: Float32Array.from(coast) };
}
