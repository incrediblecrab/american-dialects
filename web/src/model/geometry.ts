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

/**
 * Display aspect ratio: width divided by height.
 *
 * The grid is stored on equal steps of latitude and longitude, so drawing it
 * one cell to one pixel is a plate carree projection, which stretches the
 * United States about 29% too wide. Correcting by the cosine of the middle
 * latitude costs nothing and is the difference between a map that looks right
 * and one that looks subtly wrong in a way most people feel but cannot name.
 */
export function displayAspect(cells: Cells): number {
  const north = cells.lats[0];
  const south = cells.lats[cells.lats.length - 1];
  const west = cells.lons[0];
  const east = cells.lons[cells.lons.length - 1];
  const mid = ((north + south) / 2) * (Math.PI / 180);
  return (Math.abs(east - west) * Math.cos(mid)) / Math.abs(north - south);
}

/** Grid coordinates of a latitude and longitude, for placing a marker. */
export function project(
  cells: Cells,
  lat: number,
  lon: number,
): { x: number; y: number } {
  const n = cells.lats.length;
  const m = cells.lons.length;
  const top = cells.lats[0];
  const dLat = cells.lats[1] - cells.lats[0];
  const left = cells.lons[0];
  const dLon = cells.lons[1] - cells.lons[0];
  return {
    x: Math.min(m, Math.max(0, (lon - left) / dLon + 0.5)),
    y: Math.min(n, Math.max(0, (lat - top) / dLat + 0.5)),
  };
}
