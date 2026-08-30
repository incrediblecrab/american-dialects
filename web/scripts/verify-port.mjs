/**
 * Prove the browser model reproduces the Python model, exactly.
 *
 * The page runs a TypeScript port of model/infer.py, and quotes accuracy
 * figures that were measured on the Python original. If the port drifted, the
 * page would show one model and cite another, and nothing else in the build
 * would notice. So this replays the worked example that model/export_web.py
 * records in generated.json, through the same PNG surfaces the browser
 * fetches, and fails loudly on any disagreement.
 *
 * Run with:  npm run verify
 *
 * It imports posterior.ts directly. Node 23 strips the types, and the only
 * import that file has is `import type`, which disappears, so there is no
 * bundler in the path and no second copy of the maths to keep in step.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { PNG } from "pngjs";

import {
  argmax,
  credibleArea,
  posterior,
  tauForWeights,
  topPlaces,
  topStates,
} from "../src/model/posterior.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, "..");
const DATA = join(WEB, "public", "data");

const content = JSON.parse(
  readFileSync(join(WEB, "src", "content", "generated.json"), "utf8"),
);
const manifest = JSON.parse(readFileSync(join(DATA, "manifest.json"), "utf8"));

const raw = readFileSync(join(DATA, "cells.bin"));
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const CTORS = { uint8: Uint8Array, uint16: Uint16Array, float32: Float32Array };
const arr = {};
for (const a of manifest.cells.arrays) {
  arr[a.name] = new CTORS[a.dtype](buf, a.offset, a.length);
}

const cells = {
  count: manifest.cells.count,
  rows: manifest.grid.rows,
  cols: manifest.grid.cols,
  cellY: arr.cellY,
  cellX: arr.cellX,
  logPrior: arr.logPrior,
  stateIdx: arr.stateIdx,
  placeIdx: arr.placeIdx,
  km2: arr.km2,
  lats: Float64Array.from(manifest.grid.lats),
  lons: Float64Array.from(manifest.grid.lons),
};

/** Decode one surface PNG the same way the browser does, via the cell index. */
function surface(question, choice) {
  const meta = manifest.surfaces[`${question}:${choice}`];
  if (!meta) throw new Error(`no surface for ${question}:${choice}`);
  const png = PNG.sync.read(readFileSync(join(DATA, "surfaces", meta.file)));
  const out = new Float32Array(cells.count);
  for (let i = 0; i < cells.count; i++) {
    const px = (cells.cellY[i] * png.width + cells.cellX[i]) * 4;
    out[i] = meta.lo + png.data[px] * meta.scale;
  }
  return out;
}

const bits = new Map(content.questions.map((q) => [q.id, q.bits]));
const fx = content.fixture;
const answers = fx.answers.map((a) => ({
  question: a.question,
  choice: a.choice,
  bits: bits.get(a.question),
  surface: surface(a.question, a.choice),
}));

let failures = 0;
function check(name, got, want, tol) {
  const ok =
    typeof want === "number" ? Math.abs(got - want) <= tol : got === want;
  if (!ok) failures++;
  const shown = typeof got === "number" ? got.toFixed(6) : got;
  const wanted = typeof want === "number" ? want.toFixed(6) : want;
  console.log(
    `${ok ? "ok  " : "FAIL"}  ${name.padEnd(46)}  ${shown}${ok ? "" : `   expected ${wanted}`}`,
  );
}

console.log(
  `replaying ${answers.length} answers through ${manifest.cells.count.toLocaleString()} cells\n`,
);

for (const [label, v] of Object.entries(fx.variants)) {
  console.log(`-- ${label} (rho = ${v.rho})`);
  const tau = tauForWeights(
    answers.map((a) => a.bits),
    v.rho,
    content.constants.tauBase,
  );
  check(`${label}: tau`, tau, v.tau, 5e-6);

  const p = posterior(cells, answers, v.rho, content.constants.tauBase);

  let sum = 0;
  for (let i = 0; i < p.length; i++) sum += p[i];
  check(`${label}: posterior sums to one`, sum, 1, 1e-5);

  const map = argmax(p);
  check(`${label}: MAP cell`, map, v.mapCell, 0);
  check(`${label}: MAP state`, manifest.states[cells.stateIdx[map]], v.mapState);
  check(`${label}: MAP latitude`, cells.lats[cells.cellY[map]], v.mapLat, 1e-3);
  check(`${label}: MAP longitude`, cells.lons[cells.cellX[map]], v.mapLon, 1e-3);

  // The quantisation the surfaces carry is worth a few thousandths of a nat,
  // so the probabilities are compared with a relative tolerance rather than
  // an absolute one. The cell ordering above is the strict test.
  for (let i = 0; i < v.topP.length; i++) {
    check(
      `${label}: p of cell ${v.topCells[i]}`,
      p[v.topCells[i]],
      v.topP[i],
      Math.max(v.topP[i] * 0.02, 1e-9),
    );
  }

  check(
    `${label}: 80% credible area`,
    credibleArea(p, cells.km2, 0.8),
    v.area80Km2,
    Math.max(v.area80Km2 * 0.005, 1),
  );

  const places = topPlaces(p, cells, manifest.places, 3);
  const states = topStates(p, cells, manifest.states, 3);
  check(`${label}: top place`, places[0].name, v.topPlaces[0].name);
  check(`${label}: top place mass`, places[0].p, v.topPlaces[0].p, 0.002);
  check(`${label}: top state`, states[0].name, v.topStates[0].name);
  console.log("");
}

if (failures) {
  console.log(
    `${failures} check(s) failed. The browser model and model/infer.py disagree.\n` +
      "Fix the port, or regenerate the fixture if infer.py changed on purpose.",
  );
  process.exit(1);
}
console.log("the browser model reproduces model/infer.py on every check");
