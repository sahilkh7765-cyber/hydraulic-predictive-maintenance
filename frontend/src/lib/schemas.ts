/**
 * Runtime schemas for every API response. They mirror api/schemas.py one to
 * one; types in lib/types.ts are inferred from these, so each shape is
 * defined exactly once on the frontend and validated at the network boundary.
 */
import { z } from "zod/mini";

export const ComponentKey = z.enum(["cooler", "valve", "pump", "accumulator"]);
export const Severity = z.enum(["Normal", "Degraded", "Critical", "Unknown"]);

export const ApiErrorBody = z.object({ error: z.string(), detail: z.string() });

export const ClassInfo = z.object({ value: z.int(), label: z.string(), severity: Severity });
export const ComponentInfo = z.object({
  key: ComponentKey,
  display_name: z.string(),
  unit: z.string(),
  classes: z.array(ClassInfo),
});

// ---------------------------------------------------------------- health
export const Health = z.object({
  status: z.enum(["warming", "ready", "unavailable"]),
  message: z.nullable(z.string()),
  steps: z.array(z.object({ name: z.string(), done: z.boolean() })),
  checks: z.object({
    models: z.record(z.string(), z.boolean()),
    feature_table: z.boolean(),
    raw_data: z.boolean(),
    metrics: z.boolean(),
  }),
  versions: z.record(z.string(), z.string()),
});

// ---------------------------------------------------------------- overview
export const Headline = z.object({
  component: ComponentKey,
  display_name: z.string(),
  model: z.string(),
  feature_set: z.string(),
  n_features: z.int(),
  test_accuracy: z.number(),
  test_f1_macro: z.number(),
  test_roc_auc: z.nullable(z.number()),
  cv_f1_macro: z.number(),
  block_cv_f1_macro: z.number(),
  optimism: z.number(),
});

export const Overview = z.object({
  components: z.array(ComponentInfo),
  dataset: z.object({
    n_cycles: z.int(),
    n_sensors: z.int(),
    n_virtual_sensors: z.int(),
    cycle_duration_s: z.int(),
    raw_values_per_cycle: z.int(),
    total_raw_values: z.int(),
    sampling_rates_hz: z.array(z.int()),
  }),
  run: z.object({
    random_state: z.int(),
    test_size: z.number(),
    cv_folds: z.int(),
    block_size: z.int(),
    xgboost_available: z.boolean(),
    feature_sets: z.array(z.string()),
    models: z.array(z.string()),
    n_configurations: z.int(),
    n_train_cycles: z.int(),
    n_test_cycles: z.int(),
  }),
  feature_counts: z.object({
    raw: z.int(),
    engineered: z.int(),
    cleaned: z.int(),
    selected_min: z.int(),
    selected_max: z.int(),
  }),
  headline: z.array(Headline),
});

// ---------------------------------------------------------------- sensors / labels / cycles
export const Sensor = z.object({
  name: z.string(),
  quantity: z.string(),
  unit: z.string(),
  sampling_rate_hz: z.int(),
  n_samples: z.int(),
  group: z.string(),
  virtual: z.boolean(),
});
export const SensorsResponse = z.object({
  sensors: z.array(Sensor),
  groups: z.array(z.object({ key: z.string(), label: z.string(), sensors: z.array(z.string()) })),
});

export const LabelCount = z.object({
  value: z.int(),
  label: z.string(),
  severity: Severity,
  count: z.int(),
  share: z.number(),
});
export const LabelsResponse = z.object({
  n_cycles: z.int(),
  components: z.array(
    z.object({
      component: ComponentKey,
      display_name: z.string(),
      unit: z.string(),
      counts: z.array(LabelCount),
      imbalance_ratio: z.number(),
    }),
  ),
  stable_flag: z.record(z.string(), z.int()),
});

export const TrueLabel = z.object({ value: z.int(), label: z.string(), severity: Severity });
export const Signal = z.object({
  sensor: z.string(),
  quantity: z.string(),
  unit: z.string(),
  group: z.string(),
  sampling_rate_hz: z.int(),
  n_samples: z.int(),
  n_points: z.int(),
  downsampled: z.boolean(),
  t: z.array(z.number()),
  v: z.array(z.number()),
  mean: z.number(),
  min: z.number(),
  max: z.number(),
});
export const CycleResponse = z.object({
  cycle_id: z.int(),
  n_cycles: z.int(),
  labels: z.record(z.string(), TrueLabel),
  stable_flag: z.nullable(z.int()),
  split: z.record(z.string(), z.enum(["train", "test"])),
  signals: z.array(Signal),
});

// ---------------------------------------------------------------- metrics / features
export const CvRow = z.object({
  component: ComponentKey,
  feature_set: z.string(),
  model: z.string(),
  accuracy: z.number(),
  precision_macro: z.number(),
  recall_macro: z.number(),
  f1_macro: z.number(),
  f1_weighted: z.number(),
  training_time: z.number(),
  f1_macro_std: z.number(),
  n_input_features: z.int(),
});
export const TestRow = z.object({
  component: ComponentKey,
  display_name: z.string(),
  model: z.string(),
  feature_set: z.string(),
  n_train_cycles: z.int(),
  n_test_cycles: z.int(),
  accuracy: z.number(),
  precision_macro: z.number(),
  recall_macro: z.number(),
  f1_macro: z.number(),
  f1_weighted: z.number(),
  roc_auc_ovr_macro: z.nullable(z.number()),
  cv_f1_macro: z.number(),
  block_cv_f1_macro: z.number(),
  n_selected_features: z.int(),
});
export const ExperimentRow = z.object({
  component: ComponentKey,
  baseline: z.number(),
  engineered: z.number(),
  selected: z.number(),
  gain_engineered_vs_baseline: z.number(),
  gain_selected_vs_baseline: z.number(),
  gain_selected_vs_engineered: z.number(),
});
export const MetricsResponse = z.object({
  cv: z.array(CvRow),
  test: z.array(TestRow),
  experiment_effect: z.array(ExperimentRow),
  feature_set_sizes: z.record(z.string(), z.int()),
});

export const ImportanceItem = z.object({
  rank: z.int(),
  feature: z.string(),
  importance: z.number(),
  block: z.nullable(z.string()),
});
export const ImportanceResponse = z.object({
  component: ComponentKey,
  impurity: z.array(ImportanceItem),
  permutation: z.array(ImportanceItem),
});

export const SelectedFeature = z.object({ name: z.string(), block: z.nullable(z.string()) });
export const FeaturesResponse = z.object({
  raw_dimensions: z.int(),
  blocks: z.array(z.object({ key: z.string(), count: z.int(), examples: z.array(z.string()) })),
  cleaning: z.object({
    n_input: z.int(),
    n_output: z.int(),
    dropped_constant: z.array(z.string()),
    dropped_near_zero_variance: z.array(z.string()),
    dropped_duplicate: z.array(z.string()),
  }),
  selection: z.array(
    z.object({
      component: ComponentKey,
      display_name: z.string(),
      model: z.string(),
      feature_set: z.string(),
      cv_f1_macro: z.number(),
      n_selected: z.int(),
      features: z.array(SelectedFeature),
    }),
  ),
});

// ---------------------------------------------------------------- prediction
export const ClassProbability = z.object({
  value: z.int(),
  label: z.string(),
  severity: Severity,
  p: z.number(),
});
export const ComponentPrediction = z.object({
  component: ComponentKey,
  display_name: z.string(),
  unit: z.string(),
  predicted_value: z.int(),
  predicted_label: z.string(),
  severity: Severity,
  confidence: z.nullable(z.number()),
  margin: z.nullable(z.number()),
  entropy: z.nullable(z.number()),
  probabilities: z.array(ClassProbability),
  model: z.string(),
  feature_set: z.string(),
  n_features: z.int(),
  explanation: z.array(
    z.object({
      rank: z.int(),
      feature: z.string(),
      value: z.nullable(z.number()),
      importance: z.nullable(z.number()),
      block: z.nullable(z.string()),
    }),
  ),
  neighbours: z.array(
    z.object({ cycle: z.int(), distance: z.number(), true_value: z.int(), agrees: z.boolean() }),
  ),
  reliability: z.nullable(z.object({ lo: z.number(), hi: z.number(), n: z.int(), accuracy: z.nullable(z.number()) })),
  true_value: z.nullable(z.int()),
  true_label: z.nullable(z.string()),
  correct: z.nullable(z.boolean()),
  in_test_set: z.nullable(z.boolean()),
});
export const PredictionResponse = z.object({
  source: z.object({
    kind: z.enum(["dataset", "upload"]),
    cycle_id: z.nullable(z.int()),
    filename: z.nullable(z.string()),
  }),
  system_status: Severity,
  driving_component: z.string(),
  mean_confidence: z.nullable(z.number()),
  n_engineered_features: z.int(),
  components: z.array(ComponentPrediction),
});

// ---------------------------------------------------------------- models and evaluation
export const PipelineResponse = z.object({
  component: ComponentKey,
  model: z.string(),
  feature_set: z.string(),
  selection_rule: z.string(),
  cv_f1_macro: z.number(),
  cv_f1_macro_std: z.number(),
  block_cv_f1_macro: z.number(),
  steps: z.array(
    z.object({ name: z.string(), type: z.string(), n_in: z.nullable(z.int()), n_out: z.nullable(z.int()) }),
  ),
  estimator: z.string(),
  params: z.array(z.object({ key: z.string(), value: z.string() })),
  funnel: z.array(z.object({ stage: z.string(), features: z.int() })),
  balanced: z.boolean(),
  imbalance_ratio: z.number(),
  class_weights: z.array(
    z.object({ value: z.int(), label: z.string(), count: z.int(), weight: z.number() }),
  ),
  selected_features: z.array(SelectedFeature),
});

const Point = z.object({ x: z.number(), y: z.number() });
const Curve = z.object({ value: z.int(), label: z.string(), score: z.number(), points: z.array(Point) });

export const EvaluationResponse = z.object({
  component: ComponentKey,
  n_test: z.int(),
  accuracy: z.number(),
  f1_macro: z.number(),
  roc_auc: z.nullable(z.number()),
  ece: z.number(),
  n_errors: z.int(),
  classes: z.array(ClassInfo),
  confusion: z.array(z.array(z.int())),
  report: z.array(
    z.object({
      value: z.int(),
      label: z.string(),
      precision: z.number(),
      recall: z.number(),
      f1: z.number(),
      support: z.int(),
    }),
  ),
  roc: z.array(Curve),
  pr: z.array(Curve),
  calibration: z.array(z.object({ mean_confidence: z.number(), accuracy: z.number(), count: z.int() })),
  confidence_hist: z.array(
    z.object({ lo: z.number(), hi: z.number(), correct: z.int(), wrong: z.int() }),
  ),
  misclassified: z.array(
    z.object({
      cycle: z.int(),
      true_value: z.int(),
      predicted_value: z.int(),
      confidence: z.number(),
      p_true: z.number(),
      stable_flag: z.nullable(z.int()),
    }),
  ),
  saved_report_text: z.nullable(z.string()),
  parity: z.object({
    saved_accuracy: z.nullable(z.number()),
    saved_f1_macro: z.nullable(z.number()),
    matches: z.nullable(z.boolean()),
  }),
});

export const GeneralizationResponse = z.object({
  block_size: z.int(),
  rows: z.array(
    z.object({
      component: ComponentKey,
      display_name: z.string(),
      block_cv_f1_macro: z.number(),
      cv_f1_macro: z.number(),
      test_f1_macro: z.number(),
      optimism: z.number(),
    }),
  ),
});

export const TimelineResponse = z.object({
  component: ComponentKey,
  start: z.int(),
  end: z.int(),
  n_cycles: z.int(),
  classes: z.array(ClassInfo),
  points: z.array(z.object({ cycle: z.int(), value: z.int(), split: z.enum(["train", "test"]) })),
});

// ---------------------------------------------------------------- robustness and errors
export const ErrorCasesResponse = z.object({
  component: ComponentKey,
  mode: z.enum(["errors", "least_confident"]),
  n_test: z.int(),
  n_errors: z.int(),
  cases: z.array(
    z.object({
      cycle: z.int(),
      true_value: z.int(),
      predicted_value: z.int(),
      confidence: z.number(),
      p_true: z.number(),
      margin: z.number(),
    }),
  ),
});

export const ErrorProfileResponse = z.object({
  component: ComponentKey,
  cycle: z.int(),
  true_value: z.int(),
  predicted_value: z.int(),
  margin: z.number(),
  stable_flag: z.nullable(z.int()),
  in_test_set: z.boolean(),
  probabilities: z.array(ClassProbability),
  features: z.array(
    z.object({
      feature: z.string(),
      z_cycle: z.number(),
      z_true_mean: z.number(),
      z_pred_mean: z.nullable(z.number()),
    }),
  ),
});

export const WhatIfOptions = z.object({
  component: ComponentKey,
  cycle: z.int(),
  true_value: z.int(),
  features: z.array(z.object({ name: z.string(), current: z.number(), p01: z.number(), p99: z.number() })),
});

export const WhatIfResponse = z.object({
  component: ComponentKey,
  cycle: z.int(),
  feature: z.string(),
  original_value: z.number(),
  value: z.number(),
  true_value: z.int(),
  original: z.array(ClassProbability),
  modified: z.array(ClassProbability),
  original_prediction: z.int(),
  modified_prediction: z.int(),
  changed: z.boolean(),
});

export const NoiseResponse = z.object({
  component: ComponentKey,
  repeats: z.int(),
  seed: z.int(),
  points: z.array(z.object({ level: z.number(), f1_macro: z.number(), min: z.number(), max: z.number() })),
});

export const SelectiveResponse = z.object({
  component: ComponentKey,
  n_test: z.int(),
  n_errors: z.int(),
  grid: z.array(
    z.object({
      threshold: z.number(),
      coverage: z.number(),
      answered: z.int(),
      accuracy: z.nullable(z.number()),
      errors_answered: z.int(),
    }),
  ),
});

export const FiguresResponse = z.object({
  figures: z.array(z.object({ key: z.string(), title: z.string(), url: z.string() })),
});
