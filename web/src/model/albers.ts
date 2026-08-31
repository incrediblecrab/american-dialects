/**
 * Albers Equal Area Conic, the projection US thematic maps are drawn in.
 *
 * The cells are stored on a regular grid of latitude and longitude, so the
 * cheapest thing to do is draw one cell per pixel. That is a plate carree,
 * and it is wrong in a way readers feel without being able to name: the
 * parallels come out straight and horizontal, so the country reads as a
 * rectangle with a ragged edge rather than the fan shape every atlas prints.
 * Correcting only the width, which is what this project did before, fixes the
 * proportions but leaves the parallels flat.
 *
 * Equal area matters here beyond appearance. The map's whole subject is how
 * much probability sits in how much ground -- the headline number is an area
 * in square kilometres -- so a projection that inflates the north would make
 * the northern half of every posterior look more important than it is. Albers
 * preserves area exactly, which means a region that looks twice as big on
 * screen really does hold twice as much land.
 *
 * Standard parallels 29.5N and 45.5N with a central meridian of 96W are the
 * USGS and Census values for the contiguous states. The grid stops at the
 * borders -- there is no Alaska or Hawaii in the survey -- so this is the
 * single conic, not the composite with insets.
 *
 * Everything is on a unit sphere. Only ratios are ever used, so the radius
 * would cancel; leaving it out keeps the numbers small and the code short.
 */

import type { Cells } from "./payload";

const RAD = Math.PI / 180;

const LON0 = -96 * RAD;
const LAT0 = 37.5 * RAD;
const PHI1 = 29.5 * RAD;
const PHI2 = 45.5 * RAD;

const N = (Math.sin(PHI1) + Math.sin(PHI2)) / 2;
const C = Math.cos(PHI1) ** 2 + 2 * N * Math.sin(PHI1);
const RHO0 = Math.sqrt(C - 2 * N * Math.sin(LAT0)) / N;

/** Projected position on the unit sphere. y increases north. */
function project(latDeg: number, lonDeg: number): { x: number; y: number } {
  const theta = N * (lonDeg * RAD - LON0);
  const rho = Math.sqrt(Math.max(0, C - 2 * N * Math.sin(latDeg * RAD))) / N;
  return { x: rho * Math.sin(theta), y: RHO0 - rho * Math.cos(theta) };
}

/** The inverse of `project`, in degrees. */
function unproject(x: number, y: number): { lat: number; lon: number } {
  const dy = RHO0 - y;
  const rho = Math.hypot(x, dy);
  const theta = Math.atan2(x, dy);
  const s = (C - rho * rho * N * N) / (2 * N);
  const lat = Math.asin(Math.max(-1, Math.min(1, s))) / RAD;
  return { lat, lon: (theta / N + LON0) / RAD };
}

export interface Albers {
  /** Width divided by height of the projected grid. */
  aspect: number;
  /** Latitude and longitude to [0,1] square coordinates, y pointing down. */
  forward(lat: number, lon: number): { x: number; y: number };
  /** [0,1] square coordinates back to latitude and longitude. */
  inverse(u: number, v: number): { lat: number; lon: number };
  /** Grid coordinates, as `buildBorders` emits them, to [0,1] square. */
  fromGrid(gx: number, gy: number): { x: number; y: number };
}

/**
 * Fit the projection to the grid's extent.
 *
 * The projected outline of a latitude/longitude rectangle is a curved
 * trapezoid, so its bounding box is not simply the projection of the four
 * corners -- under a conic the top edge bows and the extreme northing sits at
 * the central meridian, not at a corner. Walking the perimeter finds the true
 * extent without having to reason about which edge happens to be extremal for
 * this particular set of parallels.
 */
export function makeAlbers(cells: Cells): Albers {
  const north = cells.lats[0];
  const south = cells.lats[cells.lats.length - 1];
  const west = cells.lons[0];
  const east = cells.lons[cells.lons.length - 1];

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  const STEPS = 256;
  const consider = (lat: number, lon: number) => {
    const p = project(lat, lon);
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  };
  for (let i = 0; i <= STEPS; i++) {
    const f = i / STEPS;
    const lat = north + (south - north) * f;
    const lon = west + (east - west) * f;
    consider(north, lon);
    consider(south, lon);
    consider(lat, west);
    consider(lat, east);
  }

  const spanX = maxX - minX;
  const spanY = maxY - minY;

  const top = cells.lats[0];
  const dLat = cells.lats[1] - cells.lats[0];
  const left = cells.lons[0];
  const dLon = cells.lons[1] - cells.lons[0];

  const forward = (lat: number, lon: number) => {
    const p = project(lat, lon);
    return { x: (p.x - minX) / spanX, y: (maxY - p.y) / spanY };
  };

  return {
    aspect: spanX / spanY,
    forward,
    inverse(u: number, v: number) {
      return unproject(minX + u * spanX, maxY - v * spanY);
    },
    fromGrid(gx: number, gy: number) {
      return forward(top + (gy - 0.5) * dLat, left + (gx - 0.5) * dLon);
    },
  };
}

/**
 * Which cell, if any, each pixel of a projected raster falls in.
 *
 * Reprojecting by walking the output and asking what is underneath, rather
 * than walking the cells and scattering them forward, is what keeps the result
 * free of the gaps that scattering leaves wherever the projection stretches.
 * The answer only changes when the raster is resized, so it is computed once
 * and reused: a redraw then costs a table lookup per pixel, which is what
 * makes the map cheap enough to animate on every answer.
 *
 * -1 means the pixel is off the grid or over water.
 */
export function buildLookup(
  cells: Cells,
  albers: Albers,
  width: number,
  height: number,
): Int32Array {
  const { rows, cols } = cells;
  const cellAt = new Int32Array(rows * cols).fill(-1);
  for (let i = 0; i < cells.count; i++) {
    cellAt[cells.cellY[i] * cols + cells.cellX[i]] = i;
  }

  const top = cells.lats[0];
  const dLat = cells.lats[1] - cells.lats[0];
  const left = cells.lons[0];
  const dLon = cells.lons[1] - cells.lons[0];

  const out = new Int32Array(width * height).fill(-1);
  for (let py = 0; py < height; py++) {
    const v = (py + 0.5) / height;
    for (let px = 0; px < width; px++) {
      const u = (px + 0.5) / width;
      const { lat, lon } = albers.inverse(u, v);
      const gx = Math.floor((lon - left) / dLon);
      const gy = Math.floor((lat - top) / dLat);
      if (gx < 0 || gx >= cols || gy < 0 || gy >= rows) continue;
      out[py * width + px] = cellAt[gy * cols + gx];
    }
  }
  return out;
}
