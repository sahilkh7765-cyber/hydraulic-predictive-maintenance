"""
schemas.py
==========
Response shapes of every endpoint. The frontend mirrors these one-to-one as zod
schemas in frontend/src/lib/schemas.ts; change both together.

Conventions
    * Component keys are "cooler" | "valve" | "pump" | "accumulator".
    * A condition class is identified by its documented integer value
      (e.g. valve 73/80/90/100) - never by the label-encoder index.
    * Probabilities and scores are floats in [0, 1]. Missing values are null.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

ComponentKey = Literal["cooler", "valve", "pump", "accumulator"]
Severity = Literal["Normal", "Degraded", "Critical", "Unknown"]
HealthStatus = Literal["warming", "ready", "unavailable"]


class ApiErrorBody(BaseModel):
    error: str
    detail: str


# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------
class ClassInfo(BaseModel):
    value: int
    label: str
    severity: Severity


class ComponentInfo(BaseModel):
    key: ComponentKey
    display_name: str
    unit: str
    classes: List[ClassInfo]


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
class WarmupStep(BaseModel):
    name: str
    done: bool


class HealthChecks(BaseModel):
    models: Dict[str, bool]
    feature_table: bool
    raw_data: bool
    metrics: bool


class Health(BaseModel):
    status: HealthStatus
    message: Optional[str]
    steps: List[WarmupStep]
    checks: HealthChecks
    versions: Dict[str, str]


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
class DatasetFacts(BaseModel):
    n_cycles: int
    n_sensors: int
    n_virtual_sensors: int
    cycle_duration_s: int
    raw_values_per_cycle: int
    total_raw_values: int
    sampling_rates_hz: List[int]


class RunFacts(BaseModel):
    random_state: int
    test_size: float
    cv_folds: int
    block_size: int
    xgboost_available: bool
    feature_sets: List[str]
    models: List[str]
    n_configurations: int
    n_train_cycles: int
    n_test_cycles: int


class FeatureCounts(BaseModel):
    raw: int
    engineered: int
    cleaned: int
    selected_min: int
    selected_max: int


class Headline(BaseModel):
    component: ComponentKey
    display_name: str
    model: str
    feature_set: str
    n_features: int
    test_accuracy: float
    test_f1_macro: float
    test_roc_auc: Optional[float]
    cv_f1_macro: float
    block_cv_f1_macro: float
    optimism: float


class Overview(BaseModel):
    components: List[ComponentInfo]
    dataset: DatasetFacts
    run: RunFacts
    feature_counts: FeatureCounts
    headline: List[Headline]


# --------------------------------------------------------------------------
# Sensors, labels, cycles
# --------------------------------------------------------------------------
class Sensor(BaseModel):
    name: str
    quantity: str
    unit: str
    sampling_rate_hz: int
    n_samples: int
    group: str
    virtual: bool


class SensorGroup(BaseModel):
    key: str
    label: str
    sensors: List[str]


class SensorsResponse(BaseModel):
    sensors: List[Sensor]
    groups: List[SensorGroup]


class LabelCount(BaseModel):
    value: int
    label: str
    severity: Severity
    count: int
    share: float


class ComponentLabels(BaseModel):
    component: ComponentKey
    display_name: str
    unit: str
    counts: List[LabelCount]
    imbalance_ratio: float


class LabelsResponse(BaseModel):
    n_cycles: int
    components: List[ComponentLabels]
    stable_flag: Dict[str, int]


class TrueLabel(BaseModel):
    value: int
    label: str
    severity: Severity


class Signal(BaseModel):
    sensor: str
    quantity: str
    unit: str
    group: str
    sampling_rate_hz: int
    n_samples: int
    n_points: int
    downsampled: bool
    t: List[float]
    v: List[float]
    mean: float
    min: float
    max: float


class CycleResponse(BaseModel):
    cycle_id: int
    n_cycles: int
    labels: Dict[str, TrueLabel]
    stable_flag: Optional[int]
    split: Dict[str, Literal["train", "test"]]
    signals: List[Signal]


# --------------------------------------------------------------------------
# Metrics and features
# --------------------------------------------------------------------------
class CvRow(BaseModel):
    component: ComponentKey
    feature_set: str
    model: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    training_time: float
    f1_macro_std: float
    n_input_features: int


class TestRow(BaseModel):
    component: ComponentKey
    display_name: str
    model: str
    feature_set: str
    n_train_cycles: int
    n_test_cycles: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    roc_auc_ovr_macro: Optional[float]
    cv_f1_macro: float
    block_cv_f1_macro: float
    n_selected_features: int


class ExperimentRow(BaseModel):
    component: ComponentKey
    baseline: float
    engineered: float
    selected: float
    gain_engineered_vs_baseline: float
    gain_selected_vs_baseline: float
    gain_selected_vs_engineered: float


class MetricsResponse(BaseModel):
    cv: List[CvRow]
    test: List[TestRow]
    experiment_effect: List[ExperimentRow]
    feature_set_sizes: Dict[str, int]


class ImportanceItem(BaseModel):
    rank: int
    feature: str
    importance: float
    block: Optional[str]


class ImportanceResponse(BaseModel):
    component: ComponentKey
    impurity: List[ImportanceItem]
    permutation: List[ImportanceItem]


class FeatureBlock(BaseModel):
    key: str
    count: int
    examples: List[str]


class Cleaning(BaseModel):
    n_input: int
    n_output: int
    dropped_constant: List[str]
    dropped_near_zero_variance: List[str]
    dropped_duplicate: List[str]


class SelectedFeature(BaseModel):
    name: str
    block: Optional[str]


class ComponentSelection(BaseModel):
    component: ComponentKey
    display_name: str
    model: str
    feature_set: str
    cv_f1_macro: float
    n_selected: int
    features: List[SelectedFeature]


class FeaturesResponse(BaseModel):
    raw_dimensions: int
    blocks: List[FeatureBlock]
    cleaning: Cleaning
    selection: List[ComponentSelection]


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
class ClassProbability(BaseModel):
    value: int
    label: str
    severity: Severity
    p: float


class ExplanationItem(BaseModel):
    rank: int
    feature: str
    value: Optional[float]
    importance: Optional[float]
    block: Optional[str]


class Neighbour(BaseModel):
    cycle: int
    distance: float
    true_value: int
    agrees: bool


class Reliability(BaseModel):
    lo: float
    hi: float
    n: int
    accuracy: Optional[float]


class ComponentPrediction(BaseModel):
    component: ComponentKey
    display_name: str
    unit: str
    predicted_value: int
    predicted_label: str
    severity: Severity
    confidence: Optional[float]
    margin: Optional[float]
    entropy: Optional[float]
    probabilities: List[ClassProbability]
    model: str
    feature_set: str
    n_features: int
    explanation: List[ExplanationItem]
    neighbours: List[Neighbour]
    reliability: Optional[Reliability]
    true_value: Optional[int]
    true_label: Optional[str]
    correct: Optional[bool]
    in_test_set: Optional[bool]


class PredictionSource(BaseModel):
    kind: Literal["dataset", "upload"]
    cycle_id: Optional[int]
    filename: Optional[str]


class PredictionResponse(BaseModel):
    source: PredictionSource
    system_status: Severity
    driving_component: str
    mean_confidence: Optional[float]
    n_engineered_features: int
    components: List[ComponentPrediction]


# --------------------------------------------------------------------------
# Models and evaluation
# --------------------------------------------------------------------------
class PipelineStep(BaseModel):
    name: str
    type: str
    n_in: Optional[int]
    n_out: Optional[int]


class Param(BaseModel):
    key: str
    value: str


class FunnelStage(BaseModel):
    stage: str
    features: int


class ClassWeight(BaseModel):
    value: int
    label: str
    count: int
    weight: float


class PipelineResponse(BaseModel):
    component: ComponentKey
    model: str
    feature_set: str
    selection_rule: str
    cv_f1_macro: float
    cv_f1_macro_std: float
    block_cv_f1_macro: float
    steps: List[PipelineStep]
    estimator: str
    params: List[Param]
    funnel: List[FunnelStage]
    balanced: bool
    imbalance_ratio: float
    class_weights: List[ClassWeight]
    selected_features: List[SelectedFeature]


class Point(BaseModel):
    x: float
    y: float


class Curve(BaseModel):
    value: int
    label: str
    score: float
    points: List[Point]


class ReportRow(BaseModel):
    value: int
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class CalibrationBin(BaseModel):
    mean_confidence: float
    accuracy: float
    count: int


class ConfidenceBin(BaseModel):
    lo: float
    hi: float
    correct: int
    wrong: int


class Misclassified(BaseModel):
    cycle: int
    true_value: int
    predicted_value: int
    confidence: float
    p_true: float
    stable_flag: Optional[int]


class Parity(BaseModel):
    saved_accuracy: Optional[float]
    saved_f1_macro: Optional[float]
    matches: Optional[bool]


class EvaluationResponse(BaseModel):
    component: ComponentKey
    n_test: int
    accuracy: float
    f1_macro: float
    roc_auc: Optional[float]
    ece: float
    n_errors: int
    classes: List[ClassInfo]
    confusion: List[List[int]]
    report: List[ReportRow]
    roc: List[Curve]
    pr: List[Curve]
    calibration: List[CalibrationBin]
    confidence_hist: List[ConfidenceBin]
    misclassified: List[Misclassified]
    saved_report_text: Optional[str]
    parity: Parity


class GeneralizationRow(BaseModel):
    component: ComponentKey
    display_name: str
    block_cv_f1_macro: float
    cv_f1_macro: float
    test_f1_macro: float
    optimism: float


class GeneralizationResponse(BaseModel):
    block_size: int
    rows: List[GeneralizationRow]


class TimelinePoint(BaseModel):
    cycle: int
    value: int
    split: Literal["train", "test"]


class TimelineResponse(BaseModel):
    component: ComponentKey
    start: int
    end: int
    n_cycles: int
    classes: List[ClassInfo]
    points: List[TimelinePoint]


# --------------------------------------------------------------------------
# Robustness and errors
# --------------------------------------------------------------------------
class ErrorCase(BaseModel):
    cycle: int
    true_value: int
    predicted_value: int
    confidence: float
    p_true: float
    margin: float


class ErrorCasesResponse(BaseModel):
    component: ComponentKey
    mode: Literal["errors", "least_confident"]
    n_test: int
    n_errors: int
    cases: List[ErrorCase]


class ZRow(BaseModel):
    feature: str
    z_cycle: float
    z_true_mean: float
    z_pred_mean: Optional[float]


class ErrorProfileResponse(BaseModel):
    component: ComponentKey
    cycle: int
    true_value: int
    predicted_value: int
    margin: float
    stable_flag: Optional[int]
    in_test_set: bool
    probabilities: List[ClassProbability]
    features: List[ZRow]


class WhatIfFeature(BaseModel):
    name: str
    current: float
    p01: float
    p99: float


class WhatIfOptions(BaseModel):
    component: ComponentKey
    cycle: int
    true_value: int
    features: List[WhatIfFeature]


class WhatIfRequest(BaseModel):
    cycle: int
    feature: str
    value: float


class WhatIfResponse(BaseModel):
    component: ComponentKey
    cycle: int
    feature: str
    original_value: float
    value: float
    true_value: int
    original: List[ClassProbability]
    modified: List[ClassProbability]
    original_prediction: int
    modified_prediction: int
    changed: bool


class NoisePoint(BaseModel):
    level: float
    f1_macro: float
    min: float
    max: float


class NoiseResponse(BaseModel):
    component: ComponentKey
    repeats: int
    seed: int
    points: List[NoisePoint]


class SelectivePoint(BaseModel):
    threshold: float
    coverage: float
    answered: int
    accuracy: Optional[float]
    errors_answered: int


class SelectiveResponse(BaseModel):
    component: ComponentKey
    n_test: int
    n_errors: int
    grid: List[SelectivePoint]


class Figure(BaseModel):
    key: str
    title: str
    url: str


class FiguresResponse(BaseModel):
    figures: List[Figure]
