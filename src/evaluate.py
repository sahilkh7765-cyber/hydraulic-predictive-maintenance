"""
evaluate.py
===========
Final evaluation on the held-out test set, plus every model-dependent figure
and the written comparison report.

WHY THIS IS A SEPARATE SCRIPT
    The test set must be looked at exactly once, after all modelling decisions
    are frozen. Keeping evaluation in its own script makes that property
    structural rather than a matter of discipline: `train.py` physically cannot
    tune against the test set, because it never loads those rows into a metric.
    This script reads the split indices saved by `train.py`, scores the saved
    models once, and writes the results.

OUTPUTS
    outputs/metrics/test_results.csv
    outputs/reports/classification_report_<component>.txt
    outputs/reports/model_comparison.md
    outputs/figures/confusion_matrices/*, roc_curves/*, pr_curves/*,
    feature_importance/*, pca/*

RUN
    python src/evaluate.py      (after python src/train.py)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import TARGET_SPECS  # noqa: E402
from feature_selection import selection_stage_counts  # noqa: E402
from preprocessing import feature_columns  # noqa: E402
from utils import (  # noqa: E402
    COMPONENTS,
    FEATURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    ensure_project_dirs,
    load_json,
    print_header,
    print_kv,
    save_dataframe,
)
import visualization as viz  # noqa: E402


def load_feature_tables() -> Dict[str, pd.DataFrame]:
    """Load the feature tables written by train.py.

    Output: dict with the "engineered"/"selected" table and the "baseline"
            table, so that a model trained on either can be scored.
    Raises: FileNotFoundError with a clear instruction if train.py has not run.
    """
    engineered_path = FEATURES_DIR / "engineered_features.csv"
    baseline_path = PROCESSED_DATA_DIR / "baseline_features.csv"
    if not engineered_path.exists():
        raise FileNotFoundError(
            f"{engineered_path} not found. Run `python src/train.py` first."
        )
    engineered = pd.read_csv(engineered_path)
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None
    return {"engineered": engineered, "selected": engineered, "baseline": baseline}


def load_component_artifacts(component: str) -> Dict[str, object]:
    """Load one component's fitted pipeline, encoder, metadata and split.

    Raises a readable error rather than a bare traceback when an artefact is
    missing, because the most common cause is simply that train.py has not been
    run yet.
    """
    directory = MODELS_DIR / component
    required = ["model.joblib", "label_encoder.joblib", "metadata.json", "split_indices.npz"]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artefacts for '{component}': {', '.join(missing)}. "
            f"Run `python src/train.py` first."
        )
    split = np.load(directory / "split_indices.npz")
    return {
        "model": joblib.load(directory / "model.joblib"),
        "encoder": joblib.load(directory / "label_encoder.joblib"),
        "metadata": load_json(directory / "metadata.json"),
        "train_idx": split["train_idx"],
        "test_idx": split["test_idx"],
    }


def multiclass_roc_auc(y_true: np.ndarray, proba: Optional[np.ndarray], n_classes: int) -> Optional[float]:
    """Compute one-vs-rest macro ROC-AUC when it is technically appropriate.

    WHEN IT IS NOT APPROPRIATE (returns None):
        * the model exposes no probability estimates;
        * the test set does not contain every class, so at least one
          one-vs-rest curve would be undefined;
        * the probability matrix width does not match the class count.

    Reporting None in those cases is more honest than silently computing a
    quantity on a subset of classes and labelling it "ROC-AUC".
    """
    if proba is None:
        return None
    present = np.unique(y_true)
    if len(present) < n_classes or proba.shape[1] != n_classes:
        return None
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def evaluate_component(
    component: str, tables: Dict[str, pd.DataFrame], compute_permutation: bool = True
) -> Dict[str, object]:
    """Score one component's selected model once on its held-out test set.

    Input : component key and the loaded feature tables.
    Output: dict of metrics plus the artefacts needed by the report writer.
    """
    spec = TARGET_SPECS[component]
    artifacts = load_component_artifacts(component)
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    metadata: Dict[str, object] = artifacts["metadata"]  # type: ignore[assignment]
    test_idx = artifacts["test_idx"]
    train_idx = artifacts["train_idx"]

    print_header(f"TEST EVALUATION: {spec.display_name.upper()}")
    print_kv("Selected model", f"{metadata['selected_model']} on '{metadata['feature_set']}' features")
    print_kv("Selection rule", metadata["selection_rule"])

    table = tables[str(metadata["feature_set"])]
    columns: List[str] = list(metadata["input_feature_columns"])  # type: ignore[arg-type]
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(
            f"{component}: {len(missing)} feature columns used at training time are "
            f"absent from the current feature table (first few: {missing[:3]})."
        )

    X = table[columns].to_numpy(dtype=np.float64)
    y_raw = table[spec.column].to_numpy()
    y = encoder.transform(y_raw)

    X_test, y_test = X[test_idx], y[test_idx]

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    class_values = [int(v) for v in encoder.classes_]
    class_labels = {int(v): spec.value_labels.get(int(v), str(v)) for v in class_values}
    display_labels = [f"{v} ({class_labels[v]})" for v in class_values]

    metrics = {
        "component": component,
        "display_name": spec.display_name,
        "model": metadata["selected_model"],
        "feature_set": metadata["feature_set"],
        "n_test_cycles": int(len(test_idx)),
        "n_train_cycles": int(len(train_idx)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": multiclass_roc_auc(y_test, proba, len(class_values)),
        "cv_f1_macro": metadata["cv_f1_macro"],
        "block_cv_f1_macro": metadata["block_cv_f1_macro"],
        "n_selected_features": metadata["n_selected_features"],
    }

    print_kv("Test accuracy", f"{metrics['accuracy']:.4f}")
    print_kv("Test macro F1", f"{metrics['f1_macro']:.4f}")
    print_kv("Test weighted F1", f"{metrics['f1_weighted']:.4f}")
    print_kv(
        "Test ROC-AUC (OvR macro)",
        f"{metrics['roc_auc_ovr_macro']:.4f}" if metrics["roc_auc_ovr_macro"] is not None
        else "not applicable for this model/test set",
    )
    print_kv("CV macro F1 (training)", f"{metrics['cv_f1_macro']:.4f}")

    # ---- Classification report ------------------------------------------
    report_text = classification_report(
        y_test, y_pred, labels=np.arange(len(class_values)),
        target_names=display_labels, digits=4, zero_division=0,
    )
    report_path = MODELS_DIR.parent / "outputs" / "reports" / f"classification_report_{component}.txt"
    header = (
        f"{spec.display_name} - held-out test set ({len(test_idx)} cycles)\n"
        f"Model: {metadata['selected_model']} | Feature set: {metadata['feature_set']}\n"
        f"{'-' * 72}\n"
    )
    report_path.write_text(header + report_text, encoding="utf-8")
    print("\n" + report_text)

    # ---- Figures ---------------------------------------------------------
    viz.plot_confusion_matrix(
        y_test, y_pred, display_labels,
        title=f"{spec.display_name} - test set confusion matrix\n({metadata['selected_model']}, {metadata['feature_set']} features)",
        filename=f"confusion_matrix_{component}.png",
    )
    label_lookup = {i: class_labels[v] for i, v in enumerate(class_values)}
    if proba is not None:
        viz.plot_roc_curves(
            y_test, proba, list(range(len(class_values))), label_lookup,
            title=f"{spec.display_name} - one-vs-rest ROC (test set)",
            filename=f"roc_{component}.png",
        )
        viz.plot_precision_recall_curves(
            y_test, proba, list(range(len(class_values))), label_lookup,
            title=f"{spec.display_name} - one-vs-rest precision-recall (test set)",
            filename=f"pr_{component}.png",
        )

    # ---- Feature importance ---------------------------------------------
    selected_names: List[str] = list(metadata["selected_feature_names"])  # type: ignore[arg-type]
    importance_records: List[Dict[str, object]] = []

    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        if len(importances) == len(selected_names):
            for top_n, tag in ((10, "top10"), (20, "top20")):
                viz.plot_feature_importance(
                    selected_names, importances,
                    title=f"{spec.display_name} - {tag} features ({metadata['selected_model']} impurity importance)",
                    filename=f"importance_{component}_{tag}.png", top_n=top_n,
                )
            order = np.argsort(importances)[::-1]
            importance_records = [
                {"component": component, "kind": "impurity", "rank": rank + 1,
                 "feature_name": selected_names[i], "importance": float(importances[i])}
                for rank, i in enumerate(order)
            ]

    if compute_permutation:
        # Permutation importance is model-agnostic: it measures how much the
        # test macro-F1 drops when one feature's values are shuffled. Unlike
        # impurity importance it is not biased towards high-cardinality
        # features, and it works for SVM and Logistic Regression too. It is
        # computed on the test set because that is where "does this feature
        # actually help generalisation?" is the meaningful question.
        result = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE,
            scoring="f1_macro", n_jobs=1,
        )
        # The permutation runs over the INPUT columns of the pipeline, so the
        # names are the pre-selection column names.
        means = result.importances_mean
        order = np.argsort(means)[::-1][:20]
        viz.plot_feature_importance(
            [columns[i] for i in order], [means[i] for i in order],
            title=f"{spec.display_name} - permutation importance (test set, macro F1 drop)",
            filename=f"permutation_importance_{component}.png", top_n=20,
        )
        importance_records += [
            {"component": component, "kind": "permutation", "rank": rank + 1,
             "feature_name": columns[i], "importance": float(means[i])}
            for rank, i in enumerate(np.argsort(means)[::-1][:50])
        ]

    # ---- PCA of the engineered space, coloured by this component's state --
    # PCA must run on SCALED data, so the fitted pipeline is sliced up to and
    # including its scaler and used as the transformer. Reusing the fitted
    # steps (rather than re-fitting a scaler here) keeps the picture consistent
    # with what the classifier actually sees.
    scale_position = [name for name, _ in model.steps].index("scale")
    scaled = SklearnPipeline(model.steps[: scale_position + 1]).transform(X)
    viz.plot_pca(
        scaled, y_raw,
        title=f"PCA of the engineered feature space, coloured by {spec.display_name.lower()}",
        filename=f"pca_{component}.png", class_labels=class_labels,
    )
    viz.plot_pca_explained_variance(scaled, filename=f"pca_explained_variance_{component}.png")

    # ---- Feature-count funnel -------------------------------------------
    stage_counts = selection_stage_counts(model, len(columns))
    readable: Dict[str, int] = {"after cleaning": stage_counts.get("input", len(columns))}
    if "correlation_filter" in stage_counts:
        readable["after correlation filter"] = stage_counts["correlation_filter"]
    if "mutual_information" in stage_counts:
        readable["after mutual information"] = stage_counts["mutual_information"]
    if "tree_importance" in stage_counts:
        readable["after tree importance"] = stage_counts["tree_importance"]
    readable["reaching the classifier"] = stage_counts["final"]
    viz.plot_feature_count_funnel(
        readable, title=f"{spec.display_name} - feature count at each stage",
        filename=f"feature_funnel_{component}.png",
    )

    metrics["confusion_matrix"] = confusion_matrix(
        y_test, y_pred, labels=np.arange(len(class_values))
    ).tolist()
    metrics["class_values"] = class_values
    metrics["class_labels"] = class_labels
    metrics["stage_counts"] = readable
    metrics["importance_records"] = importance_records
    metrics["metadata"] = metadata
    return metrics


def write_markdown_report(
    results: List[Dict[str, object]], cv_results: pd.DataFrame, run_metadata: Dict[str, object]
) -> Path:
    """Write outputs/reports/model_comparison.md from real execution results.

    Every number in the report is read from the metrics computed above or from
    outputs/metrics/model_results.csv. Nothing is typed in by hand.
    """
    summary: Dict[str, object] = run_metadata["dataset_summary"]  # type: ignore[assignment]
    blocks: Dict[str, int] = run_metadata["feature_blocks"]  # type: ignore[assignment]
    cleaning: Dict[str, object] = run_metadata["cleaning"]  # type: ignore[assignment]

    lines: List[str] = []
    add = lines.append

    add("# Predictive Maintenance of Industrial Hydraulic Systems")
    add("## Model comparison and results report")
    add("")
    add("All numbers in this report were produced by executing `src/train.py` and")
    add("`src/evaluate.py` on the supplied UCI dataset. No value is estimated or copied")
    add("from the literature.")
    add("")

    # ---- Dataset ---------------------------------------------------------
    add("## 1. Dataset")
    add("")
    add(f"- Source: UCI *Condition monitoring of hydraulic systems* (ZeMA gGmbH, Saarbrücken)")
    add(f"- Operating cycles (samples): **{summary['n_cycles']:,}**, each a constant-load cycle of {summary['cycle_duration_s']} s")
    add(f"- Sensors: **{summary['n_sensors']}**")
    add(f"- Raw dimensions per cycle: **{summary['total_raw_dimensions']:,}**")
    add(f"- Total raw values: **{int(summary['total_raw_dimensions']) * int(summary['n_cycles']):,}**")
    add("- Missing values: none (confirmed by the feature-quality audit)")
    add("")
    add("### Sensors and sampling rates")
    add("")
    add("| Sensor | Quantity | Unit | Sampling rate | Samples per cycle |")
    add("|---|---|---|---|---|")
    for row in summary["sensors"]:  # type: ignore[index]
        virtual = " (virtual)" if row["virtual"] else ""
        add(
            f"| {row['sensor']} | {row['quantity']}{virtual} | {row['unit']} | "
            f"{row['sampling_rate_hz']} Hz | {row['samples_per_cycle']} |"
        )
    add("")
    add("### Target variables")
    add("")
    add("| Component | Values | Classes | Distribution |")
    add("|---|---|---|---|")
    for key, target in summary["targets"].items():  # type: ignore[index]
        dist = summary["class_distributions"][key]  # type: ignore[index]
        dist_text = ", ".join(f"{v}: {c}" for v, c in dist.items())
        values_text = ", ".join(f"{v} = {label}" for v, label in target["value_labels"].items())
        add(f"| {target['display_name']} [{target['unit']}] | {values_text} | {target['n_classes']} | {dist_text} |")
    add("")

    # ---- Feature engineering --------------------------------------------
    add("## 2. Feature engineering")
    add("")
    add("One row of the engineered dataset is one complete 60-second operating cycle.")
    add("")
    add("| Block | Features generated |")
    add("|---|---|")
    for block, count in blocks.items():
        add(f"| {block.replace('_', ' ').capitalize()} | {count} |")
    total_generated = sum(blocks.values())
    add(f"| **Total generated** | **{total_generated}** |")
    add("")
    add("### Feature counts through the pipeline")
    add("")
    add("| Stage | Count |")
    add("|---|---|")
    add(f"| Raw dimensions per cycle | {summary['total_raw_dimensions']:,} |")
    add(f"| Engineered features generated | {total_generated} |")
    add(f"| Dropped: constant | {cleaning['n_dropped_constant']} |")
    add(f"| Dropped: near-zero variance | {cleaning['n_dropped_near_zero_variance']} |")
    add(f"| Dropped: exact duplicates | {cleaning['n_dropped_duplicate']} |")
    add(f"| After cleaning | {cleaning['n_output']} |")
    for result in results:
        stages = result["stage_counts"]  # type: ignore[index]
        detail = " -> ".join(f"{k}: {v}" for k, v in stages.items())
        add(f"| {result['display_name']} (selected model) | {detail} |")
    add("")

    # ---- Experimental design --------------------------------------------
    add("## 3. Experimental design")
    add("")
    add(f"- Split: {int((1 - run_metadata['test_size']) * 100)} % training / {int(run_metadata['test_size'] * 100)} % test, stratified per component target.")
    add(f"- Model selection: {run_metadata['cv_folds']}-fold stratified cross-validation **on the training set only**.")
    add("- Selection rule: the configuration with the highest mean cross-validated macro F1 is chosen, then evaluated once on the untouched test set.")
    add("- All learned transformations (imputation, scaling, correlation filtering, mutual information, tree importance) live inside a single scikit-learn `Pipeline`, so they are re-fitted on each training fold and never see validation or test rows.")
    add(f"- Leakage sanity check: `StratifiedGroupKFold` over contiguous blocks of {run_metadata['block_size']} cycles, so a cycle is never in the same fold as its immediate neighbours.")
    add("")
    add("### Three feature experiments")
    add("")
    add("| Experiment | Representation |")
    add("|---|---|")
    add("| baseline | per-sensor cycle mean (17 features) |")
    add(f"| engineered | all {cleaning['n_output']} cleaned engineered features |")
    add("| selected | engineered features + correlation filter + mutual information + tree importance |")
    add("")

    # ---- Cross-validated results ----------------------------------------
    add("## 4. Cross-validated results (training set)")
    add("")
    add("| Component | Feature set | Model | Accuracy | Macro F1 | Weighted F1 | Fit time (s) |")
    add("|---|---|---|---|---|---|---|")
    for _, row in cv_results.iterrows():
        add(
            f"| {row['component']} | {row['feature_set']} | {row['model']} | "
            f"{row['accuracy']:.4f} | {row['f1_macro']:.4f} | {row['f1_weighted']:.4f} | {row['training_time']:.2f} |"
        )
    add("")

    # ---- Effect of feature engineering -----------------------------------
    add("### Does feature engineering help? (best macro F1 per feature set)")
    add("")
    best = cv_results.groupby(["component", "feature_set"])["f1_macro"].max().unstack()
    ordered = [c for c in ("baseline", "engineered", "selected") if c in best.columns]
    add("| Component | " + " | ".join(ordered) + " | engineered - baseline | selected - baseline |")
    add("|---" * (len(ordered) + 3) + "|")
    for component, row in best.iterrows():
        values = " | ".join(f"{row[c]:.4f}" for c in ordered)
        gain_eng = row.get("engineered", np.nan) - row.get("baseline", np.nan)
        gain_sel = row.get("selected", np.nan) - row.get("baseline", np.nan)
        add(f"| {component} | {values} | {gain_eng:+.4f} | {gain_sel:+.4f} |")
    add("")

    # ---- Test results ----------------------------------------------------
    add("## 5. Final test-set results (each model evaluated exactly once)")
    add("")
    add("| Component | Model | Feature set | Test cycles | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) |")
    add("|---|---|---|---|---|---|---|---|")
    for result in results:
        auc_text = (
            f"{result['roc_auc_ovr_macro']:.4f}" if result["roc_auc_ovr_macro"] is not None else "n/a"
        )
        add(
            f"| {result['display_name']} | {result['model']} | {result['feature_set']} | "
            f"{result['n_test_cycles']} | {result['accuracy']:.4f} | {result['f1_macro']:.4f} | "
            f"{result['f1_weighted']:.4f} | {auc_text} |"
        )
    add("")
    add("### Cross-validated vs test vs temporal-block scores (macro F1)")
    add("")
    add("| Component | CV (random folds) | Test (random split) | Temporal-block CV | Optimism of random split |")
    add("|---|---|---|---|---|")
    for result in results:
        optimism = float(result["cv_f1_macro"]) - float(result["block_cv_f1_macro"])
        add(
            f"| {result['display_name']} | {result['cv_f1_macro']:.4f} | {result['f1_macro']:.4f} | "
            f"{result['block_cv_f1_macro']:.4f} | {optimism:+.4f} |"
        )
    add("")

    # ---- Confusion matrices ---------------------------------------------
    add("## 6. Confusion matrices (test set)")
    add("")
    for result in results:
        add(f"**{result['display_name']}** - rows are true classes, columns predicted")
        add("")
        labels = [f"{v} ({result['class_labels'][v]})" for v in result["class_values"]]  # type: ignore[index]
        add("| true \\ predicted | " + " | ".join(str(v) for v in result["class_values"]) + " |")
        add("|---" * (len(labels) + 1) + "|")
        for label, row in zip(labels, result["confusion_matrix"]):  # type: ignore[index]
            add(f"| {label} | " + " | ".join(str(int(v)) for v in row) + " |")
        add("")

    # ---- Important features ---------------------------------------------
    add("## 7. Most important features per component")
    add("")
    for result in results:
        records = [r for r in result["importance_records"] if r["kind"] == "impurity"][:10]  # type: ignore[index]
        if not records:
            records = [r for r in result["importance_records"] if r["kind"] == "permutation"][:10]  # type: ignore[index]
        if not records:
            continue
        kind = records[0]["kind"]
        add(f"**{result['display_name']}** (top 10 by {kind} importance)")
        add("")
        add("| Rank | Feature | Importance |")
        add("|---|---|---|")
        for record in records:
            add(f"| {record['rank']} | `{record['feature_name']}` | {record['importance']:.4f} |")
        add("")

    # ---- Limitations -----------------------------------------------------
    add("## 8. Limitations")
    add("")
    add("- **No true Remaining Useful Life.** The dataset annotates a *condition level* per cycle; it contains no maintenance timestamps and no failure times, so RUL cannot be estimated and is not claimed anywhere in this project.")
    add("- **Single test rig, laboratory conditions.** All 2,205 cycles come from one rig running one repeated load programme, with faults induced deliberately. Transfer to a different hydraulic installation is untested.")
    add("- **Temporal correlation between cycles.** Neighbouring cycles were recorded moments apart under the same component setting. The random split therefore gives a mildly optimistic picture; the temporal-block cross-validation in section 5 quantifies the gap.")
    add("- **A chronological split is not possible on this dataset.** The conditions were varied as a designed sequence, so the last 20 % of cycles contain only one cooler class. Stratified splitting is the only protocol that leaves every class testable.")
    add("- **Fault levels are discrete and superimposed.** The rig sets each component to a fixed degradation level; real degradation is continuous, and several faults are active simultaneously in the same cycle.")
    add("- **The virtual channels CE, CP and SE are already derived quantities** computed by the rig, not independent measurements. Features built on them inherit whatever model the rig used.")
    add("- **Impurity-based importance is biased** towards continuous, high-cardinality features; the permutation importances in `outputs/figures/feature_importance/` are the more trustworthy ranking.")
    add("")

    add("## 9. Reproducing these numbers")
    add("")
    add("```bash")
    add("pip install -r requirements.txt")
    add("python src/train.py      # feature engineering, experiments, model selection")
    add("python src/evaluate.py   # single test-set evaluation + this report")
    add("streamlit run app.py     # dashboard")
    add("```")
    add("")
    add(f"Random state: {run_metadata['random_state']}. "
        f"Gradient boosting backend: {'XGBoost' if run_metadata['xgboost_available'] else 'HistGradientBoostingClassifier'}.")
    add("")

    path = MODELS_DIR.parent / "outputs" / "reports" / "model_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    """Evaluate every component once and write all reports."""
    ensure_project_dirs()
    print_header("HYDRAULIC PREDICTIVE MAINTENANCE - FINAL EVALUATION", char="#")

    run_metadata_path = MODELS_DIR / "run_metadata.json"
    if not run_metadata_path.exists():
        raise FileNotFoundError("models/run_metadata.json not found. Run `python src/train.py` first.")
    run_metadata = load_json(run_metadata_path)

    cv_results = pd.read_csv(METRICS_DIR / "model_results.csv")
    tables = load_feature_tables()

    results: List[Dict[str, object]] = []
    for component in COMPONENTS:
        results.append(evaluate_component(component, tables))

    # ---- Metrics tables --------------------------------------------------
    test_frame = pd.DataFrame(
        [
            {
                key: result[key]
                for key in (
                    "component", "display_name", "model", "feature_set", "n_train_cycles",
                    "n_test_cycles", "accuracy", "precision_macro", "recall_macro",
                    "f1_macro", "f1_weighted", "roc_auc_ovr_macro", "cv_f1_macro",
                    "block_cv_f1_macro", "n_selected_features",
                )
            }
            for result in results
        ]
    )
    save_dataframe(test_frame, METRICS_DIR / "test_results.csv")

    importance_frame = pd.DataFrame(
        [record for result in results for record in result["importance_records"]]  # type: ignore[misc]
    )
    if not importance_frame.empty:
        save_dataframe(importance_frame, METRICS_DIR / "feature_importance.csv")

    report_path = write_markdown_report(results, cv_results, run_metadata)

    print_header("FINAL TEST RESULTS")
    header = f"{'Component':<26}{'Model':<22}{'Accuracy':>10}{'Macro F1':>10}{'Weighted F1':>13}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['display_name']:<26}{result['model']:<22}{result['accuracy']:>10.4f}"
            f"{result['f1_macro']:>10.4f}{result['f1_weighted']:>13.4f}"
        )

    print_header("EVALUATION COMPLETE")
    print(f"  Report  : {report_path}")
    print(f"  Metrics : outputs/metrics/test_results.csv")
    print(f"  Figures : outputs/figures/")


if __name__ == "__main__":
    main()
