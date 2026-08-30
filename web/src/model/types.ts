/** Shapes of the two files model/export_web.py writes. */

export interface SurfaceMeta {
  file: string;
  lo: number;
  scale: number;
}

export interface PlaceRow {
  name: string;
  state: string;
  lat: number;
  lon: number;
  pop: number;
}

export interface Manifest {
  grid: { rows: number; cols: number; lats: number[]; lons: number[] };
  cells: {
    count: number;
    file: string;
    arrays: { name: string; dtype: string; offset: number; length: number }[];
  };
  states: string[];
  places: PlaceRow[];
  surfaces: Record<string, SurfaceMeta>;
}

export interface Choice {
  id: string;
  text: string;
}

export interface Question {
  n: number;
  id: string;
  text: string;
  bits: number;
  choices: Choice[];
}

export interface CurveRow {
  model: string;
  k: number;
  /** Simulated speakers the row was measured on. */
  n: number;
  medianKm: number;
  p90Km: number;
  within150: number;
  stateAcc: number;
  /** Share of speakers whose true home fell inside the 80% credible set. */
  cover80: number;
  calibErr: number;
}

export interface Named {
  name: string;
  p: number;
}

/** One reading of the fixture's answers, under one value of the discount. */
export interface FixtureVariant {
  rho: number;
  tau: number;
  mapCell: number;
  mapLat: number;
  mapLon: number;
  mapState: string;
  area80Km2: number;
  topCells: number[];
  topP: number[];
  topPlaces: Named[];
  topStates: Named[];
}

export interface Fixture {
  answers: { question: string; choice: string }[];
  cell: Record<string, number>;
  variants: { deployed: FixtureVariant; discounted: FixtureVariant };
}

export interface RecoveryStage {
  file: string;
  kind: "rgb" | "mask";
  name: string;
  note: string;
}

export interface RecoveryStrip {
  question: string;
  choice: string;
  answer: string;
  colour: string;
  inkedPixels: number;
  stages: RecoveryStage[];
}

/** One side of a contrast: the answer, as the survey worded it, and its share. */
export interface Variant {
  choice: string;
  /** How the page names it, which is not always how the survey spelled it. */
  label: string;
  survey: string;
  /** Percent of all respondents nationally, as published. */
  national: number;
}

export interface Anchor {
  name: string;
  state: string;
  lat: number;
  lon: number;
  /** Probability of the first variant here, given one of the two. */
  p: number;
}

export interface Isogloss {
  id: string;
  question: string;
  questionText: string;
  a: Variant;
  b: Variant;
  /** Mean thickness of the transition zone, in kilometres. */
  widthKm: number;
  /** Length of the boundary itself, over land. */
  lineKm: number;
  /** Share of the country's land area on the first variant's side. */
  shareA: number;
  anchors: Anchor[];
  note: string;
}

export interface Content {
  constants: {
    rho: number;
    legacyRho: number;
    tauBase: number;
    nQuestions: number;
    eps: number;
  };
  /** Curve arm names, spelled by model/neural.py rather than rebuilt here. */
  models: {
    net: string;
    deployed: string;
    discounted: string;
  };
  recovery: {
    comparisons: number;
    r: number;
    mae: number;
    modalAgreement: number;
  };
  recoveryStrip: RecoveryStrip;
  popVsSoda: {
    categorised: number;
    sourceTotal: number;
    counties: number;
    offByOne: number;
  };
  tuning: { logloss: number; gridMinimum: number };
  curve: CurveRow[];
  questions: Question[];
  /** Published contrasts, sorted by measured width: sharpest first. */
  isoglosses: {
    /** Odds either way that bound the transition zone the width measures. */
    odds: number;
    contrasts: Isogloss[];
  };
  fixture: Fixture;
  inventory: { path: string; rows: number }[];
  quantisation: {
    games: number;
    questionsAsked: number;
    moved: number;
    identicalPct: number;
    worstRatio: number;
    maxKm: number;
  };
}
