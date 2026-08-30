/**
 * Loading the payload the Python exporter wrote.
 *
 * Three things come over the wire. `manifest.json` describes the grid, the
 * cells, the named places and every available answer surface. `cells.bin` is
 * six parallel typed arrays holding per-cell geography and the prior.
 * `surfaces/<question>_<choice>.png` is one quantised log-likelihood surface.
 *
 * Surfaces are fetched lazily and cached, because the published quiz asks a
 * fixed ordering. A game needs exactly the surface for each answer the player
 * actually gave -- roughly 18 KB per question -- rather than every surface of
 * every question. The whole set is 4.7 MB; a game costs a few hundred KB.
 */

import type { Manifest, SurfaceMeta } from "./types";

const BASE = import.meta.env.BASE_URL;

function url(path: string): string {
  return `${BASE}data/${path}`.replace(/([^:]\/)\/+/g, "$1");
}

export interface Cells {
  count: number;
  rows: number;
  cols: number;
  /** Row of each cell in the grid, 0..rows-1. */
  cellY: Uint8Array;
  /** Column of each cell in the grid, 0..cols-1. */
  cellX: Uint16Array;
  /** log P(cell) under the population prior. */
  logPrior: Float32Array;
  stateIdx: Uint8Array;
  placeIdx: Uint16Array;
  /** Area of each cell, which shrinks toward the poles. */
  km2: Float32Array;
  lats: Float64Array;
  lons: Float64Array;
}

export interface Payload {
  manifest: Manifest;
  cells: Cells;
}

const CTORS = {
  uint8: Uint8Array,
  uint16: Uint16Array,
  float32: Float32Array,
} as const;

export async function loadPayload(signal?: AbortSignal): Promise<Payload> {
  const [manifest, buf] = await Promise.all([
    fetch(url("manifest.json"), { signal }).then((r) => {
      if (!r.ok) throw new Error(`manifest.json: ${r.status}`);
      return r.json() as Promise<Manifest>;
    }),
    fetch(url("cells.bin"), { signal }).then((r) => {
      if (!r.ok) throw new Error(`cells.bin: ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);

  const view: Record<string, ArrayBufferView> = {};
  for (const a of manifest.cells.arrays) {
    const Ctor = CTORS[a.dtype as keyof typeof CTORS];
    if (!Ctor) throw new Error(`cells.bin: unknown dtype ${a.dtype}`);
    view[a.name] = new Ctor(buf, a.offset, a.length);
  }

  const cells: Cells = {
    count: manifest.cells.count,
    rows: manifest.grid.rows,
    cols: manifest.grid.cols,
    cellY: view.cellY as Uint8Array,
    cellX: view.cellX as Uint16Array,
    logPrior: view.logPrior as Float32Array,
    stateIdx: view.stateIdx as Uint8Array,
    placeIdx: view.placeIdx as Uint16Array,
    km2: view.km2 as Float32Array,
    lats: Float64Array.from(manifest.grid.lats),
    lons: Float64Array.from(manifest.grid.lons),
  };
  return { manifest, cells };
}

/**
 * One answer's log-likelihood surface, decoded back to nats.
 *
 * The PNG carries the surface on the full grid as a single byte per pixel,
 * rescaled onto 0..255 with its own offset and scale. Only land pixels mean
 * anything, so decoding walks the cell index rather than the image.
 */
const cache = new Map<string, Promise<Float32Array>>();

export function surfaceKey(question: string, choice: string): string {
  return `${question}:${choice}`;
}

export function loadSurface(
  manifest: Manifest,
  cells: Cells,
  question: string,
  choice: string,
): Promise<Float32Array> {
  const key = surfaceKey(question, choice);
  const hit = cache.get(key);
  if (hit) return hit;

  const meta: SurfaceMeta | undefined = manifest.surfaces[key];
  if (!meta) {
    return Promise.reject(new Error(`no surface published for ${key}`));
  }

  const job = (async () => {
    const res = await fetch(url(`surfaces/${meta.file}`));
    if (!res.ok) throw new Error(`${meta.file}: ${res.status}`);
    const bitmap = await createImageBitmap(await res.blob(), {
      premultiplyAlpha: "none",
      colorSpaceConversion: "none",
    });
    const w = bitmap.width;
    const h = bitmap.height;
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("no 2d context available to decode surfaces");
    ctx.drawImage(bitmap, 0, 0);
    const data = ctx.getImageData(0, 0, w, h).data;
    bitmap.close();

    if (w !== cells.cols || h !== cells.rows) {
      throw new Error(
        `${meta.file}: ${w}x${h} does not match the ${cells.cols}x${cells.rows} grid`,
      );
    }

    const out = new Float32Array(cells.count);
    for (let i = 0; i < cells.count; i++) {
      const px = (cells.cellY[i] * w + cells.cellX[i]) * 4;
      out[i] = meta.lo + data[px] * meta.scale;
    }
    return out;
  })();

  cache.set(key, job);
  return job;
}

/** Warm the cache for every choice of a question, so answering feels instant. */
export function prefetchQuestion(
  manifest: Manifest,
  cells: Cells,
  question: string,
  choices: readonly { id: string }[],
): void {
  for (const c of choices) {
    if (manifest.surfaces[surfaceKey(question, c.id)]) {
      void loadSurface(manifest, cells, question, c.id).catch(() => {});
    }
  }
}

/**
 * The payload, fetched at most once per page load.
 *
 * Under the single React tree this was loaded in App and passed down as a
 * prop, which guaranteed one fetch by construction. Islands have no common
 * parent to hold that state, so the guarantee has to live somewhere both of
 * them can see. A module-scope promise is that place: every island imports
 * this same module instance, so the first caller starts the request and the
 * rest await the result of it.
 *
 * There is deliberately no AbortSignal here. A signal belongs to one consumer,
 * and cancelling a shared request on behalf of every other consumer is how the
 * quiz would end up with no surfaces because the isogloss plate unmounted.
 */
let shared: Promise<Payload> | null = null;

export function sharedPayload(): Promise<Payload> {
  if (!shared) {
    shared = loadPayload().catch((e) => {
      // Let a later island retry rather than caching the failure forever.
      shared = null;
      throw e;
    });
  }
  return shared;
}
