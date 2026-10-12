import type { ComponentKey, Severity } from "./types";

export const COMPONENTS: readonly ComponentKey[] = ["cooler", "valve", "pump", "accumulator"];

/** Short names used in controls; the API supplies the full display names. */
export const COMPONENT_SHORT: Record<ComponentKey, string> = {
  cooler: "Cooler",
  valve: "Valve",
  pump: "Pump",
  accumulator: "Accumulator",
};

export const MODEL_NAMES: Record<string, string> = {
  logistic_regression: "Logistic regression",
  random_forest: "Random forest",
  gradient_boosting: "Gradient boosting",
  svm: "SVM (RBF)",
};

export const MODEL_SHORT: Record<string, string> = {
  logistic_regression: "LR",
  random_forest: "RF",
  gradient_boosting: "XGB",
  svm: "SVM",
};

/** Fixed categorical slot per model family - color follows the entity. */
export const MODEL_ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "svm"] as const;
export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"] as const;
export const SERIES_HEX = ["#3987e5", "#d95926", "#199e70", "#c98500"] as const;

export const modelColor = (model: string): string => {
  const i = MODEL_ORDER.indexOf(model as (typeof MODEL_ORDER)[number]);
  return SERIES_HEX[i] ?? SERIES_HEX[0];
};

export const FEATURE_SET_ORDER = ["baseline", "engineered", "selected"] as const;
export const FEATURE_SET_NAMES: Record<string, string> = {
  baseline: "Baseline",
  engineered: "Engineered",
  selected: "Selected",
};

export const BLOCK_NAMES: Record<string, string> = {
  statistical: "Statistical",
  time_domain: "Time domain",
  rolling: "Rolling window",
  frequency: "Frequency",
  domain: "Hydraulic domain",
};

export const SEVERITY_COLOR: Record<Severity, string> = {
  Normal: "var(--ok)",
  Degraded: "var(--warn)",
  Critical: "var(--crit)",
  Unknown: "var(--unknown)",
};

/**
 * One-hue sequential ramp for magnitude on the dark surface: low values recede
 * toward the plate, high values are brightest (blue ramp steps 700 -> 250).
 */
const RAMP = ["#1b2a3d", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"] as const;
export const rampColor = (t: number): { bg: string; fg: string } => {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const i = Math.min(RAMP.length - 1, Math.round(clamped * (RAMP.length - 1)));
  // Text flips to dark from step 4 so every cell clears 4.5:1 (measured).
  return { bg: RAMP[i] ?? RAMP[0], fg: i >= 4 ? "#0e1a24" : "#e6ebf0" };
};
export const RAMP_STOPS = RAMP;

export const SEVERITY_ORDER: Record<Severity, number> = { Normal: 0, Degraded: 1, Critical: 2, Unknown: 1 };

const nf = (digits: number) =>
  new Intl.NumberFormat("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const fmt = {
  int: (n: number) => new Intl.NumberFormat("en-US").format(n),
  fixed: (n: number, d = 3) => nf(d).format(n),
  pct: (n: number, d = 1) => `${nf(d).format(n * 100)}%`,
  signed: (n: number, d = 3) => `${n >= 0 ? "+" : "−"}${nf(d).format(Math.abs(n))}`,
  sig: (n: number, d = 4) => (Math.abs(n) >= 1e5 || (Math.abs(n) < 1e-3 && n !== 0) ? n.toExponential(2) : Number(n.toPrecision(d)).toString()),
};

export const valueWithUnit = (value: number, unit: string) => {
  if (unit === "%") return `${value}%`;
  if (unit === "level") return `level ${value}`;
  return `${value} ${unit}`;
};

export const classLabel = (value: number, unit: string, label: string) => `${valueWithUnit(value, unit)} (${label})`;
