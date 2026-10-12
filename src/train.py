"""
train.py
========
End-to-end training pipeline.

WHAT THIS SCRIPT DOES
    1. Inspects the dataset and prints the DATASET SUMMARY.
    2. Builds the engineered feature table and the baseline representation.
    3. Audits and cleans the features, printing the FEATURE ENGINEERING SUMMARY.
    4. For each of the four components, runs three feature experiments
       (baseline / engineered / selected) x four models, using cross-validation
       on the training portion only.
    5. Selects one configuration per component by the documented rule
       (highest cross-validated macro-F1) and fits it on the full training set.
    6. Saves models, pipelines, encoders, feature lists and metadata.
    7. Runs a temporal-block cross-validation as a leakage sanity check.

WHAT IT DELIBERATELY DOES NOT DO
    It never touches the held-out test set. Test evaluation happens exactly
    once, in `evaluate.py`, on the split indices saved here.

RUN
    python src/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (  # noqa: E402
    SENSOR_NAMES,
    TARGET_SPECS,
    get_dataset_summary,
    load_profile,
    load_sensor_matrix,
    print_dataset_summary,
)
from feature_engineering import (  # noqa: E402
    build_baseline_table,
    build_feature_table,
    print_feature_summary,
)
from feature_selection import (  # noqa: E402
    CorrelationFilter,
    MutualInformationSelector,
    TreeImportanceSelector,
    trace_selected_feature_names,
)
from models import XGBOOST_AVAILABLE, make_model  # noqa: E402
from preprocessing import (  # noqa: E402
    build_quality_report,
    clean_features,
    describe_class_balance,
    feature_columns,
    resolve_class_weight,
)
from utils import (  # noqa: E402
    COMPONENTS,
    FEATURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    Timer,
    ensure_project_dirs,
    load_json,
    print_header,
    print_kv,
    save_dataframe,
    save_json,
)
import visualization as viz  # noqa: E402

# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------
TEST_SIZE = 0.20
CV_FOLDS = 5
# Temporal-block size for the leakage sanity check: consecutive cycles are
# grouped into blocks of this many cycles, and whole blocks are held out
# together so that a cycle never sits in the same fold as its direct neighbour.
BLOCK_SIZE = 25

SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
}

FEATURE_SETS: tuple[str, ...] = ("baseline", "engineered", "selected")
MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "random_forest", "gradient_boosting", "svm")


def build_experiment_pipeline(
    feature_set: str,
    model_name: str,
    class_weight: Optional[str],
    n_select: int = 40,
    with_proba: bool = False,
) -> Pipeline:
    """Assemble the full estimator for one (feature set, model) combination.

    THE LEAKAGE GUARANTEE: every learned step - imputation, scaling,
    correlation filtering, mutual-information ranking, tree importance and the
    classifier itself - is a stage of this one Pipeline object. Nothing is
    fitted outside it, so when `cross_validate` or `fit` is called on training
    data, scikit-learn re-estimates all of them from that data alone.

    EXPERIMENT DEFINITIONS
        baseline   : 17 per-sensor means, impute + scale + model.
        engineered : all engineered features, impute + scale + model.
        selected   : engineered features plus the three-stage selection chain.

    Input : experiment name, model key, class weighting, target feature count.
    Output: an unfitted Pipeline.
    """
    steps: List[Tuple[str, object]] = [
        # Median imputation: robust to the heavy tails of ratio features.
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]

    if feature_set == "selected":
        # Correlation filtering runs BEFORE scaling because Pearson r is
        # scale-invariant, so filtering first saves the scaler from fitting
        # hundreds of columns that are about to be discarded.
        steps.append(("correlation_filter", CorrelationFilter(threshold=0.95)))

    steps.append(("scale", StandardScaler()))

    if feature_set == "selected":
        steps.append(
            ("mutual_information", MutualInformationSelector(percentile=50.0, min_features=30))
        )
        steps.append(
            (
                "tree_importance",
                TreeImportanceSelector(
                    k=n_select, n_estimators=200, class_weight=class_weight
                ),
            )
        )

    steps.append(("model", make_model(model_name, class_weight, with_proba=with_proba)))
    return Pipeline(steps)


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def assemble_datasets(rebuild: bool = False, verbose: bool = True) -> Dict[str, object]:
    """Build, audit, clean and persist every feature representation.

    CACHING: feature extraction reads ~550 MB of raw text and takes about a
    minute, so the result is cached on disk. A later run - or a run of a single
    component - reloads `outputs/features/engineered_features.csv`,
    `data/processed/baseline_features.csv` and
    `data/processed/pipeline_meta.json` instead of recomputing them. Pass
    `rebuild=True` (CLI: `--rebuild`) to force a fresh extraction, which is
    what you want after editing feature_engineering.py.

    Input : whether to force a rebuild.
    Output: dict with the engineered table, the baseline table, the label
            table, the block map and the cleaning report.
    """
    meta_path = PROCESSED_DATA_DIR / "pipeline_meta.json"
    engineered_path = FEATURES_DIR / "engineered_features.csv"
    baseline_path = PROCESSED_DATA_DIR / "baseline_features.csv"
    cache_ready = all(p.exists() for p in (meta_path, engineered_path, baseline_path))

    if cache_ready and not rebuild:
        meta = load_json(meta_path)
        engineered = pd.read_csv(engineered_path)
        baseline = pd.read_csv(baseline_path)
        profile = load_profile()
        # The cached CSV carries the label columns as well; the modelling code
        # expects the feature table without them, so they are dropped here.
        label_columns = [c for c in profile.columns if c != "cycle_id"]
        engineered = engineered.drop(columns=[c for c in label_columns if c in engineered.columns])
        baseline = baseline.drop(columns=[c for c in label_columns if c in baseline.columns])
        if verbose:
            print_header("FEATURE CACHE")
            print_kv("Loaded cached engineered features", f"{engineered.shape[0]} cycles x {engineered.shape[1] - 1} features")
            print_kv("Rebuild with", "python src/train.py --rebuild")
        return {
            "summary": meta["dataset_summary"],
            "profile": profile,
            "engineered": engineered,
            "baseline": baseline,
            "block_map": meta["feature_block_names"],
            "clean_report": meta["cleaning"],
            "cached": True,
        }

    profile = load_profile()
    summary = print_dataset_summary()

    engineered, block_map = build_feature_table(verbose=verbose)
    baseline = build_baseline_table(verbose=verbose)

    # ---- Feature quality audit (before any cleaning) --------------------
    quality = build_quality_report(engineered)
    save_dataframe(quality, REPORTS_DIR / "feature_quality_report.csv")
    if verbose:
        print_header("FEATURE QUALITY REPORT")
        print_kv("Features audited", len(quality))
        print_kv("Features with missing values", int((quality["missing_count"] > 0).sum()))
        print_kv("Features with infinite values", int((quality["n_infinite"] > 0).sum()))
        print_kv("Constant features", int(quality["is_constant"].sum()))
        print_kv("Report written to", "outputs/reports/feature_quality_report.csv")

    # ---- Structural cleaning -------------------------------------------
    cleaned, clean_report = clean_features(engineered, verbose=verbose)

    print_feature_summary(
        block_map,
        total_raw_dimensions=int(summary["total_raw_dimensions"]),
        counts_after={"Features after cleaning": clean_report.n_output},
    )

    # ---- Persist the engineered dataset --------------------------------
    labels = profile.drop(columns=["cycle_id"])
    engineered_out = pd.concat([cleaned.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    save_dataframe(engineered_out, engineered_path)
    save_dataframe(
        pd.concat([baseline.reset_index(drop=True), labels.reset_index(drop=True)], axis=1),
        baseline_path,
    )
    save_json(
        {
            "dataset_summary": summary,
            "feature_block_names": block_map,
            "feature_blocks": {block: len(names) for block, names in block_map.items()},
            "cleaning": clean_report.to_dict(),
        },
        meta_path,
    )
    if verbose:
        print_kv("Engineered dataset written to", "outputs/features/engineered_features.csv")

    return {
        "summary": summary,
        "profile": profile,
        "engineered": cleaned,
        "baseline": baseline,
        "block_map": block_map,
        "clean_report": clean_report.to_dict(),
        "quality": quality,
        "cached": False,
    }


def make_eda_figures(data: Dict[str, object]) -> None:
    """Produce the exploratory figures that describe the data, not the models."""
    print_header("EXPLORATORY FIGURES")
    engineered: pd.DataFrame = data["engineered"]  # type: ignore[assignment]
    profile: pd.DataFrame = data["profile"]  # type: ignore[assignment]

    paths = [
        viz.plot_sensor_distributions(engineered, SENSOR_NAMES),
        viz.plot_class_distributions(profile, TARGET_SPECS),
    ]

    # One raw cycle across a few representative channels: the figure that shows
    # why a cycle cannot be summarised by a single number.
    example_cycle = {}
    for sensor in ("PS1", "PS3", "EPS1", "FS1", "TS1", "CE"):
        matrix = load_sensor_matrix(sensor)
        example_cycle[sensor] = matrix[0].astype(float)
        del matrix
    paths.append(viz.plot_raw_cycle_examples(example_cycle))

    columns = feature_columns(engineered)
    # Highest-variance features make the most informative heatmap subset.
    variances = engineered[columns].var().sort_values(ascending=False)
    paths.append(viz.plot_correlation_heatmap(engineered, variances.index.tolist()))

    for path in paths:
        print(f"  saved {path.relative_to(path.parents[2])}")


# --------------------------------------------------------------------------
# Experiment execution
# --------------------------------------------------------------------------
def run_component(
    component: str,
    data: Dict[str, object],
    cv_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """Run all experiments for one component and fit its final model.

    STEPS
        1. Encode the component's labels.
        2. Split 80/20, stratified on that component's target.
        3. Cross-validate every (feature set, model) pair on the training part.
        4. Select the configuration with the highest mean CV macro-F1.
        5. Refit it on the whole training part and save it.

    Input : component key, the assembled datasets, and the list that collects
            cross-validation rows for outputs/metrics/model_results.csv.
    Output: dict describing the selected configuration (also saved to disk).
    """
    spec = TARGET_SPECS[component]
    engineered: pd.DataFrame = data["engineered"]  # type: ignore[assignment]
    baseline: pd.DataFrame = data["baseline"]  # type: ignore[assignment]
    profile: pd.DataFrame = data["profile"]  # type: ignore[assignment]

    print_header(f"COMPONENT: {spec.display_name.upper()}")

    # ---- Targets --------------------------------------------------------
    y_raw = profile[spec.column].to_numpy()
    encoder = LabelEncoder().fit(y_raw)
    y = encoder.transform(y_raw)

    balance = describe_class_balance(y_raw, spec.value_labels)
    print("  Class balance:")
    for _, row in balance.iterrows():
        print(
            f"    {int(row['class_value']):>5} = {row.get('class_label', ''):<28}"
            f"{int(row['count']):>5} cycles ({row['percentage']:5.2f} %)"
        )
    imbalance_ratio = balance.attrs["imbalance_ratio"]
    class_weight = resolve_class_weight(y)
    print_kv("Imbalance ratio (max/min)", f"{imbalance_ratio:.2f}")
    print_kv(
        "Class weighting",
        "balanced (imbalance > 1.5)" if class_weight else "none (classes near-uniform)",
    )

    # ---- Split ----------------------------------------------------------
    # Stratified random split. The dataset documentation describes each row as
    # one complete, independently annotated load cycle, and a purely
    # chronological split is impossible here because the conditions were varied
    # as a designed sequence: the final 20 % of cycles contain only ONE cooler
    # class, which would leave the model nothing to be tested on. The temporal
    # correlation between neighbouring cycles is instead quantified separately
    # by the block-grouped cross-validation below.
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print_kv("Training cycles", f"{len(train_idx)} ({100 * (1 - TEST_SIZE):.0f} %)")
    print_kv("Held-out test cycles", f"{len(test_idx)} ({100 * TEST_SIZE:.0f} %) - untouched until evaluate.py")

    feature_tables = {
        "baseline": baseline,
        "engineered": engineered,
        "selected": engineered,  # selection happens inside the pipeline
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best: Optional[Dict[str, object]] = None

    print(f"\n  Cross-validated comparison ({CV_FOLDS}-fold on the training set only):")
    print(f"    {'feature set':<12}{'model':<22}{'accuracy':>10}{'macro F1':>10}{'weighted F1':>13}{'fit s':>8}")

    for feature_set in FEATURE_SETS:
        table = feature_tables[feature_set]
        columns = feature_columns(table)
        X = table[columns].to_numpy(dtype=np.float64)
        X_train, y_train = X[train_idx], y[train_idx]

        for model_name in MODEL_NAMES:
            pipeline = build_experiment_pipeline(feature_set, model_name, class_weight)
            with Timer() as timer:
                scores = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=SCORING,
                    n_jobs=1,
                    error_score="raise",
                )
            mean_fit_time = float(np.mean(scores["fit_time"]))
            row = {
                "component": component,
                "feature_set": feature_set,
                "model": model_name,
                "accuracy": float(np.mean(scores["test_accuracy"])),
                "precision_macro": float(np.mean(scores["test_precision_macro"])),
                "recall_macro": float(np.mean(scores["test_recall_macro"])),
                "f1_macro": float(np.mean(scores["test_f1_macro"])),
                "f1_weighted": float(np.mean(scores["test_f1_weighted"])),
                "training_time": round(mean_fit_time, 3),
                "f1_macro_std": float(np.std(scores["test_f1_macro"])),
                "n_input_features": len(columns),
                "total_cv_seconds": round(timer.elapsed, 1),
            }
            cv_rows.append(row)
            print(
                f"    {feature_set:<12}{model_name:<22}{row['accuracy']:>10.4f}"
                f"{row['f1_macro']:>10.4f}{row['f1_weighted']:>13.4f}{mean_fit_time:>8.2f}"
            )

            # SELECTION RULE (fixed in advance, applied identically to every
            # component): highest mean cross-validated macro-F1. Macro-F1 is
            # used rather than accuracy because it weights every degradation
            # class equally, so a model cannot look good by ignoring a rare but
            # operationally critical failure state.
            #
            # TIE-BREAK: the comparison is a strict ">", and the loops run in
            # the order baseline -> engineered -> selected and
            # logistic regression -> random forest -> gradient boosting -> SVM.
            # An exact tie therefore keeps the configuration seen first, i.e.
            # the simplest feature representation and the simplest model. That
            # is deliberate parsimony: if 17 sensor means match 553 engineered
            # features, the honest conclusion is that the extra machinery was
            # not needed for that component.
            if best is None or row["f1_macro"] > best["f1_macro"]:
                best = dict(row)

    assert best is not None
    print(
        f"\n  SELECTED: {best['model']} on the '{best['feature_set']}' feature set "
        f"(CV macro F1 = {best['f1_macro']:.4f} +/- {best['f1_macro_std']:.4f})"
    )

    # ---- Refit the selected configuration on the full training set ------
    table = feature_tables[str(best["feature_set"])]
    columns = feature_columns(table)
    X = table[columns].to_numpy(dtype=np.float64)

    final_pipeline = build_experiment_pipeline(
        str(best["feature_set"]), str(best["model"]), class_weight, with_proba=True
    )
    with Timer() as timer:
        final_pipeline.fit(X[train_idx], y[train_idx])
    print_kv("Final refit on training set", f"{timer.elapsed:.1f} s")

    selected_names = trace_selected_feature_names(final_pipeline, columns)
    print_kv("Features reaching the classifier", len(selected_names))

    # ---- Leakage sanity check: temporal-block cross-validation -----------
    # Neighbouring cycles were recorded seconds apart under the same component
    # setting, so a random split can place near-duplicate cycles on both sides.
    # Holding out whole contiguous blocks removes that advantage; the gap
    # between the two numbers is an honest estimate of how optimistic the
    # random-split score is.
    groups = np.arange(len(y)) // BLOCK_SIZE
    block_cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    block_pipeline = build_experiment_pipeline(
        str(best["feature_set"]), str(best["model"]), class_weight
    )
    block_scores = cross_validate(
        block_pipeline, X, y, cv=block_cv.split(X, y, groups),
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"}, n_jobs=1,
    )
    block_f1 = float(np.mean(block_scores["test_f1_macro"]))
    block_acc = float(np.mean(block_scores["test_accuracy"]))
    print_kv(
        f"Temporal-block CV (blocks of {BLOCK_SIZE} cycles)",
        f"macro F1 = {block_f1:.4f}, accuracy = {block_acc:.4f}",
    )
    print_kv("Optimism of random split", f"{best['f1_macro'] - block_f1:+.4f} macro F1")

    # ---- Persist everything needed to reproduce a prediction -------------
    component_dir = MODELS_DIR / component
    joblib.dump(final_pipeline, component_dir / "model.joblib")
    joblib.dump(encoder, component_dir / "label_encoder.joblib")

    metadata = {
        "component": component,
        "display_name": spec.display_name,
        "unit": spec.unit,
        "selected_model": best["model"],
        "feature_set": best["feature_set"],
        "selection_rule": ("highest mean cross-validated macro F1 on the training set; "
                           "ties broken in favour of the simpler feature set and model"),
        "cv_f1_macro": best["f1_macro"],
        "cv_f1_macro_std": best["f1_macro_std"],
        "cv_accuracy": best["accuracy"],
        "block_cv_f1_macro": block_f1,
        "block_cv_accuracy": block_acc,
        "random_split_optimism_f1_macro": best["f1_macro"] - block_f1,
        "class_weight": class_weight,
        "imbalance_ratio": imbalance_ratio,
        "input_feature_columns": columns,
        "n_input_features": len(columns),
        "selected_feature_names": selected_names,
        "n_selected_features": len(selected_names),
        "class_values": [int(v) for v in encoder.classes_],
        "class_labels": {str(int(v)): spec.value_labels.get(int(v), str(v)) for v in encoder.classes_},
        "severity": {str(int(v)): spec.severity.get(int(v), "Unknown") for v in encoder.classes_},
        "train_indices_n": int(len(train_idx)),
        "test_indices_n": int(len(test_idx)),
        "random_state": RANDOM_STATE,
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    save_json(metadata, component_dir / "metadata.json")
    np.savez_compressed(
        component_dir / "split_indices.npz", train_idx=train_idx, test_idx=test_idx
    )
    # Per-component metrics are written immediately. A long run can then be
    # executed one component at a time (`--component valve`) and resumed after
    # an interruption without losing the work already done.
    component_rows = [row for row in cv_rows if row["component"] == component]
    save_dataframe(pd.DataFrame(component_rows), METRICS_DIR / f"model_results_{component}.csv")

    return {
        "component": component,
        "best": best,
        "selected_names": selected_names,
        "metadata": metadata,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def write_selected_features_csv(component_results: List[Dict[str, object]]) -> Path:
    """Write outputs/features/selected_features.csv.

    CONTENT: one row per (component, feature) pair for the features that
    survive selection in the chosen model, so the reader can see exactly which
    engineered quantities each component's model relies on.
    """
    rows: List[Dict[str, object]] = []
    for result in component_results:
        for rank, name in enumerate(result["selected_names"], start=1):  # type: ignore[index]
            rows.append(
                {
                    "component": result["component"],
                    "feature_set": result["best"]["feature_set"],  # type: ignore[index]
                    "rank": rank,
                    "feature_name": name,
                }
            )
    frame = pd.DataFrame(rows)
    return save_dataframe(frame, FEATURES_DIR / "selected_features.csv")


def print_comparison_table(results: pd.DataFrame) -> None:
    """Print the model-comparison table required by the brief."""
    print_header("MODEL COMPARISON (cross-validated on training data)")
    header = f"{'Component':<14}{'Feature Set':<13}{'Model':<22}{'Accuracy':>10}{'Macro F1':>10}{'Weighted F1':>13}"
    print(header)
    print("-" * len(header))
    for _, row in results.iterrows():
        print(
            f"{row['component']:<14}{row['feature_set']:<13}{row['model']:<22}"
            f"{row['accuracy']:>10.4f}{row['f1_macro']:>10.4f}{row['f1_weighted']:>13.4f}"
        )


def summarise_experiment_effect(results: pd.DataFrame) -> pd.DataFrame:
    """Quantify the research question: did feature engineering help?

    Input : the cross-validation results table.
    Output: per-component best macro-F1 for each feature set plus the deltas.
    """
    best_per_set = (
        results.groupby(["component", "feature_set"])["f1_macro"].max().unstack()
    )
    ordered = best_per_set[[c for c in FEATURE_SETS if c in best_per_set.columns]]
    ordered = ordered.assign(
        gain_engineered_vs_baseline=lambda d: d["engineered"] - d["baseline"],
        gain_selected_vs_baseline=lambda d: d["selected"] - d["baseline"],
        gain_selected_vs_engineered=lambda d: d["selected"] - d["engineered"],
    )
    return ordered.reset_index()


def finalize(data: Dict[str, object]) -> None:
    """Aggregate per-component results into the project-level outputs.

    WHY A SEPARATE STAGE: components can be trained in separate invocations, so
    the tables and figures that span all four of them are assembled afterwards
    from the per-component files on disk. This also means re-running one
    component and then `--stage finalize` refreshes everything consistently.

    Input : the assembled datasets (for the block map and dataset summary).
    Output: writes model_results.csv, feature_experiment_effect.csv,
            selected_features.csv, the comparison figures and run_metadata.json.
    """
    frames = []
    for component in COMPONENTS:
        path = METRICS_DIR / f"model_results_{component}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            "No per-component metrics found. Train at least one component first, "
            "e.g. `python src/train.py --component cooler`."
        )
    results = pd.concat(frames, ignore_index=True)

    required_columns = [
        "component", "feature_set", "model", "accuracy", "precision_macro",
        "recall_macro", "f1_macro", "f1_weighted", "training_time",
    ]
    extra_columns = [c for c in results.columns if c not in required_columns]
    results = results[required_columns + extra_columns]
    save_dataframe(results, METRICS_DIR / "model_results.csv")
    print_comparison_table(results)

    effect = summarise_experiment_effect(results)
    save_dataframe(effect, METRICS_DIR / "feature_experiment_effect.csv")
    print_header("EFFECT OF FEATURE ENGINEERING (best macro F1 per feature set)")
    print(effect.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    # Rebuild the per-component summary from the saved metadata files, so this
    # stage does not depend on objects held in memory by an earlier run.
    component_results: List[Dict[str, object]] = []
    for component in COMPONENTS:
        metadata_path = MODELS_DIR / component / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        component_results.append(
            {
                "component": component,
                "selected_names": metadata["selected_feature_names"],
                "best": {
                    "model": metadata["selected_model"],
                    "feature_set": metadata["feature_set"],
                    "f1_macro": metadata["cv_f1_macro"],
                },
                "metadata": metadata,
            }
        )

    write_selected_features_csv(component_results)

    viz.plot_model_comparison(results, metric="f1_macro")
    block_map: Dict[str, List[str]] = data["block_map"]  # type: ignore[assignment]
    for result in component_results:
        viz.plot_feature_block_contribution(
            result["selected_names"],  # type: ignore[arg-type]
            block_map,
            title=f"Which engineering blocks feed the {result['component']} model?",
            filename=f"feature_blocks_{result['component']}.png",
        )

    cleaning = data["clean_report"]
    run_metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "block_size": BLOCK_SIZE,
        "feature_sets": list(FEATURE_SETS),
        "models": list(MODEL_NAMES),
        "xgboost_available": XGBOOST_AVAILABLE,
        "dataset_summary": data["summary"],
        "feature_blocks": {block: len(names) for block, names in block_map.items()},
        "feature_block_names": block_map,
        "cleaning": cleaning if isinstance(cleaning, dict) else cleaning.to_dict(),  # type: ignore[union-attr]
        "selected_per_component": {
            result["component"]: {
                "model": result["best"]["model"],  # type: ignore[index]
                "feature_set": result["best"]["feature_set"],  # type: ignore[index]
                "cv_f1_macro": result["best"]["f1_macro"],  # type: ignore[index]
                "n_selected_features": len(result["selected_names"]),  # type: ignore[arg-type]
            }
            for result in component_results
        },
    }
    save_json(run_metadata, MODELS_DIR / "run_metadata.json")


def parse_args(argv: Optional[List[str]] = None):
    """Command-line interface.

    WHY STAGES EXIST: the full run takes several minutes and reads ~550 MB of
    raw data. Splitting it into `features`, per-component `train` and
    `finalize` makes the pipeline restartable on a modest machine and lets you
    re-train a single component after changing one model's settings, without
    redoing feature extraction.

        python src/train.py                      # everything, in order
        python src/train.py --stage features     # extraction + EDA figures only
        python src/train.py --component valve    # one component (uses the cache)
        python src/train.py --stage finalize     # aggregate tables and figures
        python src/train.py --rebuild            # force feature re-extraction
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train condition-monitoring models for a hydraulic test rig."
    )
    parser.add_argument(
        "--stage", choices=["all", "features", "train", "finalize"], default="all",
        help="which part of the pipeline to run (default: all)",
    )
    parser.add_argument(
        "--component", choices=list(COMPONENTS), default=None,
        help="train a single component instead of all four",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="force feature re-extraction instead of using the cached tables",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="skip the exploratory figures (useful when only re-training models)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the requested stage(s) of the training pipeline."""
    args = parse_args(argv)
    ensure_project_dirs()

    print_header("HYDRAULIC PREDICTIVE MAINTENANCE - TRAINING PIPELINE", char="#")
    print(f"  Random state       : {RANDOM_STATE}")
    print(f"  Gradient boosting  : {'XGBoost' if XGBOOST_AVAILABLE else 'HistGradientBoosting (XGBoost unavailable)'}")
    print(f"  Stage              : {args.stage}" + (f" (component: {args.component})" if args.component else ""))

    data = assemble_datasets(rebuild=args.rebuild, verbose=True)

    if args.stage in ("all", "features"):
        if not args.no_figures:
            make_eda_figures(data)
        if args.stage == "features":
            print_header("FEATURE STAGE COMPLETE")
            print("  Next: python src/train.py --stage train")
            return

    if args.stage in ("all", "train"):
        components = [args.component] if args.component else list(COMPONENTS)
        cv_rows: List[Dict[str, object]] = []
        for component in components:
            run_component(component, data, cv_rows)
        if args.component and args.stage != "all":
            print_header("COMPONENT TRAINING COMPLETE")
            print("  Run `python src/train.py --stage finalize` once every component is trained.")
            return

    if args.stage in ("all", "finalize"):
        finalize(data)

    print_header("TRAINING COMPLETE")
    print("  Saved:")
    print("    models/<component>/model.joblib        fitted pipeline (preprocessing + selection + classifier)")
    print("    models/<component>/label_encoder.joblib")
    print("    models/<component>/metadata.json")
    print("    models/<component>/split_indices.npz   train/test cycle indices")
    print("    outputs/features/engineered_features.csv")
    print("    outputs/features/selected_features.csv")
    print("    outputs/metrics/model_results.csv")
    print("\n  Next step: python src/evaluate.py  (single evaluation on the untouched test set)")


if __name__ == "__main__":
    main()
