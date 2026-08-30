/**
 * The model, in the browser.
 *
 * A direct port of `posterior` and `tau_for_weights` from model/infer.py.
 * It has to be a port rather than an approximation: every accuracy figure the
 * page quotes was measured on the Python implementation, so if this one
 * disagreed, the page would be showing one model and citing another.
 *
 * Two things are already folded in by the exporter and so are absent here.
 * The eps contamination is baked into each surface, because eps is frozen at
 * the deployed value and pre-mixing spares the client the national marginal
 * and a log-sum-exp. And the surfaces arrive as log-likelihoods, so combining
 * evidence is addition.
 */

import type { Cells } from "./payload";

/**
 * How much to discount the likelihood, counting information rather than
 * questions.
 *
 * Discounting by raw question count assumes every answer carries the same
 * information, and in this quiz they differ by about a factor of six. Kish's
 * effective sample size, (sum w)^2 / sum w^2, counts near-k when the weights
 * are even and much less when a few dominate.
 *
 * The running maximum is not a fitted parameter but a constraint: more
 * evidence may never leave the model with less effective evidence than it
 * already had. It binds only past the peak. Weights are sorted descending
 * first so the prefixes it accumulates over are the informative ones, which
 * makes the result independent of the order questions happened to be asked in.
 */
export function tauForWeights(weights: number[], rho: number, base: number): number {
  const w = weights.filter((x) => x > 0).sort((a, b) => b - a);
  if (w.length === 0) return base;

  let cum = 0;
  let cumSq = 0;
  let running = -Infinity;
  let lastCum = 0;
  for (const x of w) {
    cum += x;
    cumSq += x * x;
    const kEff = cumSq > 0 ? (cum * cum) / cumSq : 1;
    const total = (base * cum) / (1 + (kEff - 1) * rho);
    if (total > running) running = total;
    lastCum = cum;
  }
  return running / lastCum;
}

export interface Answer {
  question: string;
  choice: string;
  /** Mutual information in bits between this question and location. */
  bits: number;
  surface: Float32Array;
}

/**
 * P(cell) given a set of answers.
 *
 * `rho` and `tauBase` are arguments rather than constants so the page can show
 * the same answers scored under the discount that was deployed and under the
 * one the findings recommend. That comparison is the whole of the third act,
 * and it has to be the real model on both sides of the slider.
 */
export function posterior(
  cells: Cells,
  answers: Answer[],
  rho: number,
  tauBase: number,
): Float64Array {
  const n = cells.count;
  const out = new Float64Array(n);
  if (answers.length === 0) {
    let max = -Infinity;
    for (let i = 0; i < n; i++) if (cells.logPrior[i] > max) max = cells.logPrior[i];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const v = Math.exp(cells.logPrior[i] - max);
      out[i] = v;
      sum += v;
    }
    for (let i = 0; i < n; i++) out[i] /= sum;
    return out;
  }

  const tau = tauForWeights(answers.map((a) => a.bits), rho, tauBase);

  // lp = log_prior + tau * sum(loglik), which is what
  //   lp = log_prior + tau * ((log_prior + sum) - log_prior)
  // reduces to. Accumulate the sum first so tau is applied once.
  const acc = new Float64Array(n);
  for (const a of answers) {
    const s = a.surface;
    for (let i = 0; i < n; i++) acc[i] += s[i];
  }

  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = cells.logPrior[i] + tau * acc[i];
    acc[i] = v;
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.exp(acc[i] - max);
    out[i] = v;
    sum += v;
  }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

export function tauUsed(answers: Answer[], rho: number, tauBase: number): number {
  return tauForWeights(answers.map((a) => a.bits), rho, tauBase);
}

export function argmax(p: Float64Array): number {
  let best = 0;
  let bestV = -Infinity;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > bestV) {
      bestV = p[i];
      best = i;
    }
  }
  return best;
}

/**
 * Area of the smallest set of cells holding `level` of the mass.
 *
 * This is the number that makes the design-effect mistake legible. Two models
 * can put their modal cell in the same town and still differ by a factor of
 * four in how much of the country they are unwilling to rule out.
 */
export function credibleArea(
  p: Float64Array,
  km2: Float32Array,
  level = 0.8,
): number {
  const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]);
  let cum = 0;
  let area = 0;
  for (const i of order) {
    cum += p[i];
    area += km2[i];
    if (cum >= level) break;
  }
  return area;
}

export interface Ranked {
  name: string;
  p: number;
}

/**
 * The most likely named places.
 *
 * Every cell is assigned to its nearest listed place, so the town scores are a
 * partition of the whole posterior. Scoring only the cells that happen to
 * contain a town would discard most of the mass, and would discard it unevenly:
 * metros whose population sits in many small boroughs would be penalised
 * against metros of the same size built from large incorporated cities.
 */
export function topPlaces(
  p: Float64Array,
  cells: Cells,
  places: { name: string; state: string }[],
  k = 3,
): Ranked[] {
  const mass = new Float64Array(places.length);
  for (let i = 0; i < p.length; i++) mass[cells.placeIdx[i]] += p[i];
  return rank(
    mass,
    (i) => `${places[i].name}, ${places[i].state}`,
    k,
  );
}

export function topStates(
  p: Float64Array,
  cells: Cells,
  states: string[],
  k = 3,
): Ranked[] {
  const mass = new Float64Array(states.length);
  for (let i = 0; i < p.length; i++) mass[cells.stateIdx[i]] += p[i];
  return rank(mass, (i) => states[i], k);
}

function rank(mass: Float64Array, name: (i: number) => string, k: number): Ranked[] {
  return Array.from(mass.keys())
    .sort((a, b) => mass[b] - mass[a])
    .slice(0, k)
    .map((i) => ({ name: name(i), p: mass[i] }));
}

const EARTH_KM = 6371.0088;

export function haversine(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
): number {
  const r = Math.PI / 180;
  const p1 = lat1 * r;
  const p2 = lat2 * r;
  const dp = p2 - p1;
  const dl = (lon2 - lon1) * r;
  const a =
    Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return EARTH_KM * 2 * Math.asin(Math.sqrt(Math.min(1, Math.max(0, a))));
}
