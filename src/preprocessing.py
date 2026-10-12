"""
preprocessing.py
================
Feature-quality auditing and leakage-safe preprocessing.

WHAT THIS MODULE DOES
    1. Produces the feature-quality report (missing values, dtypes, ranges).
    2. Removes features that are structurally useless: all-NaN, constant,
       near-zero variance, exactly duplicated.
    3. Builds the scikit-learn Pipelines used by every experiment.

WHY IT IS NEEDED - THE LEAKAGE ARGUMENT
    Every step that *learns something from data* - the median used for
    imputation, the mean and scale used for standardisation, the variance
    threshold, the correlation filter, the mutual-information ranking, the tree
    importances - must be estimated from training data only. If any of them is
    estimated on the full dataset before splitting, information from the test
    cycles has leaked into the model and the reported accuracy is optimistic.

    The defence used throughout this project is structural rather than manual:
    all learned steps live inside a single `sklearn.pipeline.Pipeline`, and only
    that pipeline object is passed to `cross_val_score` / `fit`. scikit-learn
    then re-fits every step on each training fold automatically, so it is
    impossible to leak by accident.

    The only transformation applied before the split is the *deterministic*
    removal of columns that are constant or duplicated across the entire
    dataset. That is a schema decision, not a learned statistic - it uses no
    label information and no test-set distribution. Where even that is
    undesirable, `VarianceThreshold` inside the pipeline repeats the check on
    each fold anyway.

INPUT  : the engineered feature DataFrame
OUTPUT : cleaned DataFrame, quality report DataFrame, fitted-on-train pipelines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import RANDOM_STATE, print_header, print_kv

# Columns that describe a cycle but are never model inputs.
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "cycle_id",
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure",
    "stable_flag",
)


@dataclass
class CleaningReport:
    """Record of which features were dropped before modelling and why.

    Kept as an object rather than printed away, because the brief requires the
    exact feature counts at each stage to appear in the final report.
    """

    n_input: int = 0
    dropped_all_missing: List[str] = field(default_factory=list)
    dropped_constant: List[str] = field(default_factory=list)
    dropped_near_zero_variance: List[str] = field(default_factory=list)
    dropped_duplicate: List[str] = field(default_factory=list)
    n_output: int = 0

    @property
    def n_dropped(self) -> int:
        return (
            len(self.dropped_all_missing)
            + len(self.dropped_constant)
            + len(self.dropped_near_zero_variance)
            + len(self.dropped_duplicate)
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_input": self.n_input,
            "n_dropped_all_missing": len(self.dropped_all_missing),
            "n_dropped_constant": len(self.dropped_constant),
            "n_dropped_near_zero_variance": len(self.dropped_near_zero_variance),
            "n_dropped_duplicate": len(self.dropped_duplicate),
            "n_output": self.n_output,
            "dropped_constant": self.dropped_constant,
            "dropped_near_zero_variance": self.dropped_near_zero_variance,
            "dropped_duplicate": self.dropped_duplicate,
        }


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the modelling columns of a table (everything that is not an ID or label)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def build_quality_report(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Produce the feature-quality report required by the brief.

    WHAT: one row per feature with dtype, missing count/percentage, number of
    unique values and basic descriptive statistics.

    WHY: engineered features are generated programmatically, so it is entirely
    possible to create a column that is constant, or that silently became NaN
    because a denominator was zero for every cycle. Auditing the table before
    training is what catches that; the report is also the evidence that the
    dataset genuinely contains no missing values, as its documentation claims.

    Input : feature DataFrame; optionally an explicit column subset.
    Output: DataFrame with columns
            feature_name, dtype, missing_count, missing_percentage,
            unique_values, mean, std, min, max, n_infinite, is_constant.
    """
    columns = list(columns) if columns is not None else feature_columns(df)
    n_rows = len(df)
    rows: List[Dict[str, object]] = []

    for name in columns:
        series = df[name]
        values = pd.to_numeric(series, errors="coerce")
        missing = int(values.isna().sum())
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "feature_name": name,
                "dtype": str(series.dtype),
                "missing_count": missing,
                "missing_percentage": round(100.0 * missing / n_rows, 4) if n_rows else 0.0,
                "unique_values": int(values.nunique(dropna=True)),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "n_infinite": int(np.isinf(values.to_numpy(dtype=float)).sum()),
                "is_constant": bool(values.nunique(dropna=True) <= 1),
            }
        )
    return pd.DataFrame(rows)


def clean_features(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    near_zero_variance_threshold: float = 1e-10,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Drop structurally useless feature columns.

    WHAT IS REMOVED AND WHY
        * all-missing columns - carry no information at all;
        * constant columns - zero variance means zero discriminative power, and
          they break correlation computations (division by a zero std);
        * near-zero-variance columns - numerically constant; keeping them adds
          noise to scaling (dividing by an almost-zero std inflates values);
        * exact duplicates - two identical columns double-count the same
          evidence and destabilise linear models through collinearity.

    LEAKAGE NOTE: all four criteria are label-free structural checks. None of
    them uses the target, so applying them to the whole table before splitting
    does not leak label information. Learned statistics (median, mean, scale,
    thresholds involving the target) remain inside the pipeline.

    Input : engineered feature table.
    Output: (cleaned DataFrame keeping non-feature columns, CleaningReport).
    """
    columns = list(columns) if columns is not None else feature_columns(df)
    report = CleaningReport(n_input=len(columns))
    keep: List[str] = []

    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    # Infinities are treated as missing here so they cannot survive into the
    # model; the imputer inside the pipeline fills them on a per-fold basis.
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    for name in columns:
        values = numeric[name]
        if values.isna().all():
            report.dropped_all_missing.append(name)
            continue
        if values.nunique(dropna=True) <= 1:
            report.dropped_constant.append(name)
            continue
        variance = float(values.var(ddof=1))
        if not np.isfinite(variance) or variance < near_zero_variance_threshold:
            report.dropped_near_zero_variance.append(name)
            continue
        keep.append(name)

    # Duplicate detection: compare column value vectors, keeping the first
    # occurrence. Done via a hash of the rounded values so that two columns
    # that differ only by floating-point noise are not falsely merged.
    seen: Dict[bytes, str] = {}
    unique_keep: List[str] = []
    for name in keep:
        digest = np.ascontiguousarray(
            numeric[name].to_numpy(dtype=np.float64)
        ).round(12).tobytes()
        if digest in seen:
            report.dropped_duplicate.append(name)
            continue
        seen[digest] = name
        unique_keep.append(name)

    report.n_output = len(unique_keep)
    retained_meta = [c for c in df.columns if c in NON_FEATURE_COLUMNS]
    cleaned = pd.concat([df[retained_meta], numeric[unique_keep]], axis=1)

    if verbose:
        print_header("FEATURE QUALITY / CLEANING")
        print_kv("Features before cleaning", report.n_input)
        print_kv("Dropped - all values missing", len(report.dropped_all_missing))
        print_kv("Dropped - constant", len(report.dropped_constant))
        print_kv("Dropped - near-zero variance", len(report.dropped_near_zero_variance))
        print_kv("Dropped - exact duplicates", len(report.dropped_duplicate))
        print_kv("Features after cleaning", report.n_output)
    return cleaned, report


def build_preprocessor(scale: bool = True) -> Pipeline:
    """Build the learned-preprocessing pipeline used for every experiment.

    STEPS
        impute : median imputation. Median rather than mean because several
                 engineered ratios are heavy-tailed, and a single extreme cycle
                 would drag a mean-based fill far from the typical value.
        scale  : standardisation. Required by Logistic Regression and SVM,
                 whose objectives depend on distance/penalty scale; harmless for
                 the tree models, so one shared pipeline keeps the comparison
                 fair across models.

    LEAKAGE: both steps are estimators. Because they are returned inside a
    Pipeline and never fitted directly on the full table, `cross_val_score` and
    the final `fit` re-estimate them on training data only.

    Input : whether to include the scaler.
    Output: an unfitted sklearn Pipeline.
    """
    steps: List[Tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def describe_class_balance(y: np.ndarray, class_names: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    """Summarise a target's class distribution and quantify the imbalance.

    WHY: the four targets are imbalanced to very different degrees. The valve
    target has 51 % of cycles in one class, and accumulator/pump are similarly
    skewed. Reporting the imbalance ratio up front is what justifies using
    macro-F1 (which weights every class equally) as the headline metric instead
    of accuracy, and using class-weighted models rather than resampling.

    Input : integer label array; optional mapping from raw value to a name.
    Output: DataFrame with class, count, percentage and the imbalance ratio.
    """
    values, counts = np.unique(y, return_counts=True)
    frame = pd.DataFrame(
        {
            "class_value": values,
            "count": counts,
            "percentage": np.round(100.0 * counts / counts.sum(), 2),
        }
    )
    if class_names:
        frame["class_label"] = [class_names.get(int(v), str(v)) for v in values]
    frame.attrs["imbalance_ratio"] = float(counts.max() / counts.min())
    return frame


def resolve_class_weight(y: np.ndarray, threshold: float = 1.5) -> Optional[str]:
    """Decide whether a model should be class-weighted.

    RULE: if the largest class is more than `threshold` times the smallest,
    return "balanced" so that under-represented degradation states are not
    ignored by the optimiser; otherwise return None and leave the model
    unweighted.

    WHY NOT SMOTE: synthetic oversampling interpolates between neighbouring
    feature vectors. Here the classes are ordered degradation levels of a
    physical component, so interpolating between "weak leakage" and "severe
    leakage" manufactures cycles that the rig never produced and that sit
    exactly where the decision boundary is hardest. Class weighting achieves
    the same rebalancing without inventing data, so it is preferred. The
    imbalance in this dataset (worst case roughly 3:1) is also mild enough that
    resampling is not warranted.

    Input : training labels only.
    Output: "balanced" or None, to be passed to the estimator.
    """
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return None
    return "balanced" if (counts.max() / counts.min()) > threshold else None
