/**
 * Every number the page says out loud, and where it came from.
 *
 * generated.json is written by model/export_web.py from the CSVs in data/, and
 * nothing in src/ may hardcode a figure that lives in it. That is not a style
 * rule: r = 0.955 already appears in three documents, and a fourth hand-typed
 * copy in a component is a fourth thing that can quietly go stale. check.py
 * enforces both halves, that the JSON matches the data and that the components
 * do not duplicate it.
 *
 * This module exists so the rest of the code imports a typed value rather than
 * a bare JSON blob, and so the formatting helpers live next to the data they
 * format.
 */

import raw from "./generated.json";
import type { Content, CurveRow } from "../model/types";

export const content = raw as unknown as Content;

export const { constants, recovery, recoveryStrip, popVsSoda, tuning, quantisation } =
  content;

/** Published contrasts, already sorted by measured width. */
export const isoglosses = content.isoglosses;

/** The published question ordering, longest first. */
export const questions = content.questions;

/** The questions the deployed quiz actually asks. */
export const asked = content.questions.slice(0, constants.nQuestions);

/**
 * Curve arm names, exported by model/neural.py rather than rebuilt here.
 *
 * They used to be reconstructed from the rho values, and that was a bug
 * waiting for RHO to become zero: the leading-zero strip that turns 0.177 into
 * ".177" turns 0 into "", so the deployed arm resolved to "bayes(rho=)",
 * matched no curve row, and crashed the page. A name derived from a value in
 * two languages is two names.
 */
export const MODELS = content.models;

export function curveFor(model: string): CurveRow[] {
  return content.curve.filter((r) => r.model === model).sort((a, b) => a.k - b.k);
}

export function curveAt(model: string, k: number): CurveRow | undefined {
  return content.curve.find((r) => r.model === model && r.k === k);
}

/** Largest k measured, so the page never claims a point off the end. */
export const kMax = Math.max(...content.curve.map((r) => r.k));

export const int = new Intl.NumberFormat("en-US");
export const oneDp = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

export function km(n: number): string {
  return `${int.format(Math.round(n))} km`;
}

export function km2(n: number): string {
  return `${int.format(Math.round(n))} km²`;
}

export function pct(x: number, dp = 0): string {
  return `${(x * 100).toFixed(dp)}%`;
}
