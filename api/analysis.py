"""
analysis.py
===========
Builds every API response from the artefacts held in `AppState`.

The evaluation, error-analysis, what-if and robustness computations follow
MLapp.py exactly (same bins, thresholds, noise levels and seed), so the web app
reports the same numbers the Streamlit studio did. Prediction goes through
`src/prediction.py` - nothing in this file re-implements preprocessing or
feature engineering.
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

import prediction
from data_loader import SENSOR_SPECS, TARGET_SPECS
from utils import COMPONENTS

from . import schemas as S
from .core import AppState, ComponentArtefacts
from .errors import ApiError

SIGNAL_BUCKETS = 300  # 100 Hz channels: 6000 samples -> 300 min/max buckets (600 points)
NOISE_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
NOISE_REPEATS = 3
NOISE_SEED = 42
SENSOR_GROUPS = (
    ("pressure", "Pressure"),
    ("flow", "Flow"),
    ("temperature", "Temperature"),
    ("power", "Motor power"),
    ("vibration", "Vibration"),
    ("efficiency", "Virtual (computed)"),
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _f(value: object) -> Optional[float]:
    """Float or None for NaN/inf, so the JSON never carries NaN."""
    if value is None:
        return None
    x = float(value)  # type: ignore[arg-type]
    return x if math.isfinite(x) else None


def _class_info(component: str, values: List[int]) -> List[S.ClassInfo]:
    spec = TARGET_SPECS[component]
    return [
        S.ClassInfo(value=v, label=spec.value_labels.get(v, "unknown"), severity=spec.severity.get(v, "Unknown"))  # type: ignore[arg-type]
        for v in values
    ]


def _probabilities(art: ComponentArtefacts, proba: np.ndarray) -> List[S.ClassProbability]:
    return [
        S.ClassProbability(value=v, label=art.label(v), severity=art.severity(v), p=float(p))  # type: ignore[arg-type]
        for v, p in zip(art.classes, proba)
    ]


def _margin_entropy(proba: np.ndarray) -> "tuple[float, float]":
    srt = np.sort(proba)
    margin = float(srt[-1] - srt[-2]) if len(srt) > 1 else 1.0
    nz = proba[proba > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(len(proba))) if len(proba) > 1 else 0.0
    return margin, max(0.0, entropy)


def component_infos() -> List[S.ComponentInfo]:
    out = []
    for c in COMPONENTS:
        spec = TARGET_SPECS[c]
        out.append(
            S.ComponentInfo(
                key=c,  # type: ignore[arg-type]
                display_name=spec.display_name,
                unit=spec.unit,
                classes=_class_info(c, sorted(spec.value_labels)),
            )
        )
    return out


# --------------------------------------------------------------------------
# Overview, sensors, labels, cycles
# --------------------------------------------------------------------------
def overview(state: AppState) -> S.Overview:
    state.require_metrics()
    meta = state.run_meta
    if meta is None:
        raise ApiError(503, "metrics_unavailable", "models/run_metadata.json is missing. Run `python src/train.py`.")
    summary: Dict[str, object] = meta["dataset_summary"]  # type: ignore[assignment]
    sensors: List[Dict[str, object]] = summary["sensors"]  # type: ignore[assignment]
    blocks: Dict[str, int] = meta["feature_blocks"]  # type: ignore[assignment]
    cleaning: Dict[str, object] = meta["cleaning"]  # type: ignore[assignment]
    selected: Dict[str, Dict[str, object]] = meta["selected_per_component"]  # type: ignore[assignment]
    test = state.metrics["test_results"]
    cv = state.metrics["model_results"]
    n_cycles = int(summary["n_cycles"])  # type: ignore[arg-type]
    raw_dims = int(summary["total_raw_dimensions"])  # type: ignore[arg-type]
    per_model = [int(v["n_selected_features"]) for v in selected.values()]  # type: ignore[arg-type]

    headline = []
    for row in test.itertuples():
        headline.append(
            S.Headline(
                component=row.component,
                display_name=row.display_name,
                model=row.model,
                feature_set=row.feature_set,
                n_features=int(row.n_selected_features),
                test_accuracy=float(row.accuracy),
                test_f1_macro=float(row.f1_macro),
                test_roc_auc=_f(row.roc_auc_ovr_macro),
                cv_f1_macro=float(row.cv_f1_macro),
                block_cv_f1_macro=float(row.block_cv_f1_macro),
                optimism=float(row.cv_f1_macro) - float(row.block_cv_f1_macro),
            )
        )
    order = {c: i for i, c in enumerate(COMPONENTS)}
    headline.sort(key=lambda h: order[h.component])

    return S.Overview(
        components=component_infos(),
        dataset=S.DatasetFacts(
            n_cycles=n_cycles,
            n_sensors=int(summary["n_sensors"]),  # type: ignore[arg-type]
            n_virtual_sensors=sum(1 for s in sensors if s.get("virtual")),
            cycle_duration_s=int(summary["cycle_duration_s"]),  # type: ignore[arg-type]
            raw_values_per_cycle=raw_dims,
            total_raw_values=raw_dims * n_cycles,
            sampling_rates_hz=sorted({int(s["sampling_rate_hz"]) for s in sensors}, reverse=True),  # type: ignore[arg-type]
        ),
        run=S.RunFacts(
            random_state=int(meta["random_state"]),  # type: ignore[arg-type]
            test_size=float(meta["test_size"]),  # type: ignore[arg-type]
            cv_folds=int(meta["cv_folds"]),  # type: ignore[arg-type]
            block_size=int(meta["block_size"]),  # type: ignore[arg-type]
            xgboost_available=bool(meta["xgboost_available"]),
            feature_sets=list(meta["feature_sets"]),  # type: ignore[arg-type]
            models=list(meta["models"]),  # type: ignore[arg-type]
            n_configurations=len(cv),
            n_train_cycles=int(test["n_train_cycles"].iloc[0]),
            n_test_cycles=int(test["n_test_cycles"].iloc[0]),
        ),
        feature_counts=S.FeatureCounts(
            raw=raw_dims,
            engineered=int(sum(blocks.values())),
            cleaned=int(cleaning["n_output"]),  # type: ignore[arg-type]
            selected_min=min(per_model),
            selected_max=max(per_model),
        ),
        headline=headline,
    )


def sensors() -> S.SensorsResponse:
    items = [
        S.Sensor(
            name=s.name,
            quantity=s.quantity,
            unit=s.unit,
            sampling_rate_hz=s.sampling_rate_hz,
            n_samples=s.n_samples,
            group=s.group,
            virtual=s.virtual,
        )
        for s in SENSOR_SPECS
    ]
    groups = [
        S.SensorGroup(key=key, label=label, sensors=[s.name for s in SENSOR_SPECS if s.group == key])
        for key, label in SENSOR_GROUPS
    ]
    return S.SensorsResponse(sensors=items, groups=groups)


def labels(state: AppState) -> S.LabelsResponse:
    state.require_ready()
    frame = state.labels_frame()
    out = []
    for c in COMPONENTS:
        spec = TARGET_SPECS[c]
        counts = frame[spec.column].value_counts().sort_index()
        total = int(counts.sum())
        out.append(
            S.ComponentLabels(
                component=c,  # type: ignore[arg-type]
                display_name=spec.display_name,
                unit=spec.unit,
                counts=[
                    S.LabelCount(
                        value=int(v),
                        label=spec.value_labels.get(int(v), "unknown"),
                        severity=spec.severity.get(int(v), "Unknown"),  # type: ignore[arg-type]
                        count=int(n),
                        share=int(n) / total,
                    )
                    for v, n in counts.items()
                ],
                imbalance_ratio=float(counts.max() / counts.min()),
            )
        )
    stable = frame["stable_flag"].value_counts()
    return S.LabelsResponse(
        n_cycles=len(frame),
        components=out,
        stable_flag={"stable": int(stable.get(0, 0)), "not_stable": int(stable.get(1, 0))},
    )


def _true_labels(state: AppState, cycle_id: int) -> Dict[str, S.TrueLabel]:
    row = state.labels_frame().iloc[cycle_id]
    out = {}
    for c in COMPONENTS:
        spec = TARGET_SPECS[c]
        v = int(row[spec.column])
        out[c] = S.TrueLabel(value=v, label=spec.value_labels.get(v, "unknown"), severity=spec.severity.get(v, "Unknown"))  # type: ignore[arg-type]
    return out


def _downsample(values: np.ndarray, rate: int) -> "tuple[List[float], List[float], bool]":
    """Min/max envelope decimation: keeps spikes that plain striding would drop."""
    n = len(values)
    if n <= 2 * SIGNAL_BUCKETS:
        t = (np.arange(n) / rate).round(3).tolist()
        return t, [float(f"{x:.6g}") for x in values], False
    edges = np.linspace(0, n, SIGNAL_BUCKETS + 1).astype(int)
    idx: List[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = values[lo:hi]
        a, b = lo + int(np.argmin(seg)), lo + int(np.argmax(seg))
        idx.extend(sorted({a, b}))
    arr = np.asarray(idx)
    return (arr / rate).round(3).tolist(), [float(f"{x:.6g}") for x in values[arr]], True


def cycle(state: AppState, cycle_id: int) -> S.CycleResponse:
    state.require_raw()
    state.check_cycle(cycle_id)
    raw = state.raw_cycle(cycle_id)
    signals = []
    for spec in SENSOR_SPECS:
        values = raw[spec.name]
        t, v, down = _downsample(values, spec.sampling_rate_hz)
        signals.append(
            S.Signal(
                sensor=spec.name,
                quantity=spec.quantity,
                unit=spec.unit,
                group=spec.group,
                sampling_rate_hz=spec.sampling_rate_hz,
                n_samples=len(values),
                n_points=len(v),
                downsampled=down,
                t=t,
                v=v,
                mean=float(values.mean()),
                min=float(values.min()),
                max=float(values.max()),
            )
        )
    frame = state.labels_frame()
    return S.CycleResponse(
        cycle_id=cycle_id,
        n_cycles=state.n_cycles,
        labels=_true_labels(state, cycle_id),
        stable_flag=int(frame.iloc[cycle_id]["stable_flag"]),
        split={c: ("test" if cycle_id in art.test_set else "train") for c, art in state.components.items()},
        signals=signals,
    )


def cycle_csv(state: AppState, cycle_id: int) -> bytes:
    """A valid upload file for one dataset cycle, written by `export_cycle_csv`."""
    state.require_raw()
    state.check_cycle(cycle_id)
    with tempfile.TemporaryDirectory() as tmp:
        path = prediction.export_cycle_csv(state.raw_cycle(cycle_id), Path(tmp) / "cycle.csv")
        return path.read_bytes()


# --------------------------------------------------------------------------
# Metrics and features
# --------------------------------------------------------------------------
def metrics(state: AppState) -> S.MetricsResponse:
    state.require_metrics()
    cv = state.metrics["model_results"]
    test = state.metrics["test_results"]
    effect = state.metrics["feature_experiment_effect"]
    # Features that reach the classifier. For "selected", n_input_features is
    # the pipeline's input (553); the selection happens inside the pipeline, so
    # the reaching count is read from a saved model that uses that set.
    sizes = {str(k): int(v) for k, v in cv.groupby("feature_set")["n_input_features"].max().items()}
    for art in state.components.values():
        if art.metadata.get("feature_set") == "selected":
            sizes["selected"] = int(art.metadata["n_selected_features"])  # type: ignore[arg-type]
    return S.MetricsResponse(
        cv=[S.CvRow(**{k: r[k] for k in S.CvRow.model_fields}) for r in cv.to_dict("records")],
        test=[
            S.TestRow(**{k: (_f(r[k]) if k == "roc_auc_ovr_macro" else r[k]) for k in S.TestRow.model_fields})
            for r in test.to_dict("records")
        ],
        experiment_effect=[S.ExperimentRow(**{k: r[k] for k in S.ExperimentRow.model_fields}) for r in effect.to_dict("records")],
        feature_set_sizes=sizes,
    )


def importance(state: AppState, component: str, top: int) -> S.ImportanceResponse:
    state.require_metrics()
    imp = state.metrics["feature_importance"]
    d = imp[imp.component == component]

    def items(kind: str) -> List[S.ImportanceItem]:
        rows = d[d.kind == kind].sort_values("importance", ascending=False).head(top)
        return [
            S.ImportanceItem(rank=i, feature=r.feature_name, importance=float(r.importance), block=state.block_of.get(r.feature_name))
            for i, r in enumerate(rows.itertuples(), start=1)
        ]

    return S.ImportanceResponse(component=component, impurity=items("impurity"), permutation=items("permutation"))  # type: ignore[arg-type]


def features(state: AppState) -> S.FeaturesResponse:
    state.require_ready()
    meta = state.run_meta
    if meta is None:
        raise ApiError(503, "metrics_unavailable", "models/run_metadata.json is missing.")
    names: Dict[str, List[str]] = meta["feature_block_names"]  # type: ignore[assignment]
    cleaning: Dict[str, object] = meta["cleaning"]  # type: ignore[assignment]
    selected: Dict[str, Dict[str, object]] = meta["selected_per_component"]  # type: ignore[assignment]
    summary: Dict[str, object] = meta["dataset_summary"]  # type: ignore[assignment]
    selection = []
    for c in COMPONENTS:
        art = state.components[c]
        sel = selected[c]
        feats: List[str] = list(art.metadata["selected_feature_names"])  # type: ignore[arg-type]
        selection.append(
            S.ComponentSelection(
                component=c,  # type: ignore[arg-type]
                display_name=TARGET_SPECS[c].display_name,
                model=str(sel["model"]),
                feature_set=str(sel["feature_set"]),
                cv_f1_macro=float(sel["cv_f1_macro"]),  # type: ignore[arg-type]
                n_selected=int(sel["n_selected_features"]),  # type: ignore[arg-type]
                features=[S.SelectedFeature(name=f, block=state.block_of.get(f)) for f in feats],
            )
        )
    return S.FeaturesResponse(
        raw_dimensions=int(summary["total_raw_dimensions"]),  # type: ignore[arg-type]
        blocks=[S.FeatureBlock(key=k, count=len(v), examples=v[:8]) for k, v in names.items()],
        cleaning=S.Cleaning(
            n_input=int(cleaning["n_input"]),  # type: ignore[arg-type]
            n_output=int(cleaning["n_output"]),  # type: ignore[arg-type]
            dropped_constant=list(cleaning["dropped_constant"]),  # type: ignore[arg-type]
            dropped_near_zero_variance=list(cleaning["dropped_near_zero_variance"]),  # type: ignore[arg-type]
            dropped_duplicate=list(cleaning["dropped_duplicate"]),  # type: ignore[arg-type]
        ),
        selection=selection,
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _neighbours(state: AppState, art: ComponentArtefacts, x: np.ndarray, exclude: Optional[int], pred: int) -> List[S.Neighbour]:
    if art.train_embedding is None or state.table is None:
        return []
    q = art.model[:-1].transform(x)  # type: ignore[index]
    d = np.sqrt(((art.train_embedding - q) ** 2).sum(axis=1))
    if exclude is not None:
        d = np.where(art.train_idx == exclude, np.inf, d)  # a cycle is never its own neighbour
    nn = np.argsort(d)[:5]
    out = []
    for i in nn:
        cyc = int(art.train_idx[i])
        true_v = int(state.table.loc[cyc, art.spec.column])
        out.append(S.Neighbour(cycle=cyc, distance=float(d[i]), true_value=true_v, agrees=true_v == pred))
    return out


def _reliability(art: ComponentArtefacts, conf: Optional[float]) -> Optional[S.Reliability]:
    if conf is None:
        return None
    for b in art.reliability:
        if b["lo"] <= conf < (b["hi"] if b["hi"] < 1.0 else 1.0001):
            return S.Reliability(lo=b["lo"], hi=b["hi"], n=int(b["n"]), accuracy=b["accuracy"])
    return None


def predict(state: AppState, cycle: Dict[str, np.ndarray], source: S.PredictionSource) -> S.PredictionResponse:
    """Score one raw cycle with the four saved pipelines via src/prediction.py."""
    try:
        feats = prediction.features_from_cycle(cycle)
        result = prediction.predict_system_condition(features=feats)
    except ValueError as error:
        raise ApiError(400, "invalid_cycle", str(error)) from error

    truth = _true_labels(state, source.cycle_id) if source.cycle_id is not None else None
    comps: Dict[str, Dict[str, object]] = result["components"]  # type: ignore[assignment]
    out = []
    for c in COMPONENTS:
        p = comps[c]
        art = state.components[c]
        x = feats[art.input_columns].to_numpy(dtype=np.float64)
        probs: Optional[Dict[int, float]] = p["probabilities"]  # type: ignore[assignment]
        proba = np.array([probs[v] for v in art.classes]) if probs else None
        margin, entropy = _margin_entropy(proba) if proba is not None else (None, None)
        pred = int(p["predicted_value"])  # type: ignore[arg-type]
        explanation = prediction.explain_prediction(c, feats, top_n=10)
        tv = truth[c].value if truth else None
        out.append(
            S.ComponentPrediction(
                component=c,  # type: ignore[arg-type]
                display_name=str(p["display_name"]),
                unit=str(p["unit"]),
                predicted_value=pred,
                predicted_label=str(p["predicted_label"]),
                severity=str(p["severity"]),  # type: ignore[arg-type]
                confidence=_f(p["confidence"]),
                margin=margin,
                entropy=entropy,
                probabilities=_probabilities(art, proba) if proba is not None else [],
                model=str(p["model"]),
                feature_set=str(p["feature_set"]),
                n_features=int(p["n_selected_features"]),  # type: ignore[arg-type]
                explanation=[
                    S.ExplanationItem(
                        rank=int(r.rank),
                        feature=str(r.feature_name),
                        value=_f(r.value),
                        importance=_f(r.importance),
                        block=state.block_of.get(str(r.feature_name)),
                    )
                    for r in explanation.itertuples()
                ],
                neighbours=_neighbours(state, art, x, source.cycle_id, pred),
                reliability=_reliability(art, _f(p["confidence"])),
                true_value=tv,
                true_label=truth[c].label if truth else None,
                correct=(tv == pred) if tv is not None else None,
                in_test_set=(source.cycle_id in art.test_set) if source.cycle_id is not None else None,
            )
        )
    return S.PredictionResponse(
        source=source,
        system_status=str(result["system_status"]),  # type: ignore[arg-type]
        driving_component=str(result["driving_component"]),
        mean_confidence=_f(result["mean_confidence"]),
        n_engineered_features=int(feats.shape[1]),
        components=out,
    )


def predict_dataset_cycle(state: AppState, cycle_id: int) -> S.PredictionResponse:
    state.require_raw()
    state.check_cycle(cycle_id)
    try:
        raw = state.raw_cycle(cycle_id)
    except (IndexError, ValueError, FileNotFoundError) as error:
        raise ApiError(404, "cycle_not_found", str(error)) from error
    return predict(state, raw, S.PredictionSource(kind="dataset", cycle_id=cycle_id, filename=None))


def predict_upload(state: AppState, content: bytes, filename: Optional[str]) -> S.PredictionResponse:
    state.require_ready()
    try:
        raw = prediction.load_cycle_csv(io.BytesIO(content))
    except ValueError as error:
        raise ApiError(400, "invalid_csv", str(error)) from error
    return predict(state, raw, S.PredictionSource(kind="upload", cycle_id=None, filename=filename))


# --------------------------------------------------------------------------
# Models and evaluation
# --------------------------------------------------------------------------
PARAM_KEYS = (
    "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_samples_leaf",
    "max_features", "C", "gamma", "kernel", "class_weight", "balance_classes", "tree_method", "random_state",
)


def pipeline(state: AppState, component: str) -> S.PipelineResponse:
    art = state.component(component)
    table = state.require_table()
    pipe = art.model
    meta = art.metadata
    est = pipe.named_steps["model"]  # type: ignore[union-attr]

    steps = []
    funnel = [S.FunnelStage(stage="Input columns", features=len(art.input_columns))]
    for name, step in pipe.steps:  # type: ignore[union-attr]
        n_in = getattr(step, "n_features_in_", None)
        sel = getattr(step, "selected_index_", None)
        n_out = len(sel) if sel is not None else (n_in if name != "model" else None)
        steps.append(S.PipelineStep(name=name, type=type(step).__name__, n_in=n_in, n_out=n_out))
        if sel is not None:
            funnel.append(S.FunnelStage(stage=name.replace("_", " "), features=len(sel)))
    funnel.append(S.FunnelStage(stage="Reach classifier", features=int(meta["n_selected_features"])))  # type: ignore[arg-type]

    params = est.get_params()
    items = [S.Param(key=k, value=str(params[k])) for k in PARAM_KEYS if k in params and params[k] is not None]
    if hasattr(est, "estimators_"):
        depths = [t.get_depth() for t in est.estimators_]
        leaves = [t.get_n_leaves() for t in est.estimators_]
        items += [
            S.Param(key="mean tree depth (fitted)", value=f"{np.mean(depths):.1f}"),
            S.Param(key="mean leaves per tree (fitted)", value=f"{np.mean(leaves):.0f}"),
        ]

    y_train = table.loc[art.train_idx, art.spec.column].to_numpy()
    vals, counts = np.unique(y_train, return_counts=True)
    balanced = meta.get("class_weight") == "balanced"
    weights = len(y_train) / (len(vals) * counts) if balanced else np.ones_like(counts, dtype=float)

    return S.PipelineResponse(
        component=component,  # type: ignore[arg-type]
        model=str(meta["selected_model"]),
        feature_set=str(meta["feature_set"]),
        selection_rule=str(meta["selection_rule"]),
        cv_f1_macro=float(meta["cv_f1_macro"]),  # type: ignore[arg-type]
        cv_f1_macro_std=float(meta["cv_f1_macro_std"]),  # type: ignore[arg-type]
        block_cv_f1_macro=float(meta["block_cv_f1_macro"]),  # type: ignore[arg-type]
        steps=steps,
        estimator=type(est).__name__,
        params=items,
        funnel=funnel,
        balanced=balanced,
        imbalance_ratio=float(meta["imbalance_ratio"]),  # type: ignore[arg-type]
        class_weights=[
            S.ClassWeight(value=int(v), label=art.label(int(v)), count=int(n), weight=float(w))
            for v, n, w in zip(vals, counts, weights)
        ],
        selected_features=[
            S.SelectedFeature(name=f, block=state.block_of.get(f)) for f in meta["selected_feature_names"]  # type: ignore[union-attr]
        ],
    )


def _thin(xs: np.ndarray, ys: np.ndarray, limit: int = 400) -> List[S.Point]:
    if len(xs) > limit:
        keep = np.unique(np.linspace(0, len(xs) - 1, limit).astype(int))
        xs, ys = xs[keep], ys[keep]
    return [S.Point(x=float(a), y=float(b)) for a, b in zip(xs, ys)]


def evaluation(state: AppState, component: str) -> S.EvaluationResponse:
    art = state.component(component)
    table = state.require_table()
    assert art.test_y is not None and art.test_proba is not None
    y, proba, pred = art.test_y, art.test_proba, art.test_pred
    k = len(art.classes)
    conf = proba.max(axis=1)
    correct = pred == y

    acc = float(accuracy_score(y, pred))
    f1m = float(f1_score(y, pred, average="macro"))
    auc = float(roc_auc_score(y, proba, multi_class="ovr", average="macro")) if k > 2 else float(roc_auc_score(y, proba[:, 1]))

    bins = np.linspace(0, 1, 11)
    ece = 0.0
    calibration = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += float(m.mean() * abs(correct[m].mean() - conf[m].mean()))
            calibration.append(S.CalibrationBin(mean_confidence=float(conf[m].mean()), accuracy=float(correct[m].mean()), count=int(m.sum())))

    hist_edges = np.round(np.arange(0, 1.0001, 0.05), 2)
    hist = []
    for i, (lo, hi) in enumerate(zip(hist_edges[:-1], hist_edges[1:])):
        last = i == len(hist_edges) - 2
        m = (conf >= lo) & ((conf <= hi) if last else (conf < hi))
        hist.append(S.ConfidenceBin(lo=float(lo), hi=float(hi), correct=int((m & correct).sum()), wrong=int((m & ~correct).sum())))

    cm = confusion_matrix(y, pred, labels=list(range(k)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(k)), zero_division=0)
    report = [
        S.ReportRow(value=v, label=art.label(v), precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i]))
        for i, v in enumerate(art.classes)
    ]

    roc, pr = [], []
    for i, v in enumerate(art.classes):
        yi = (y == i).astype(int)
        if yi.min() == yi.max():
            continue
        fpr, tpr, _ = roc_curve(yi, proba[:, i])
        prec, rec, _ = precision_recall_curve(yi, proba[:, i])
        roc.append(S.Curve(value=v, label=art.label(v), score=float(roc_auc_score(yi, proba[:, i])), points=_thin(fpr, tpr)))
        pr.append(S.Curve(value=v, label=art.label(v), score=float(average_precision_score(yi, proba[:, i])), points=_thin(rec, prec)))

    wrong = np.where(~correct)[0]
    misclassified = [
        S.Misclassified(
            cycle=int(art.test_idx[i]),
            true_value=art.classes[int(y[i])],
            predicted_value=art.classes[int(pred[i])],
            confidence=float(conf[i]),
            p_true=float(proba[i, y[i]]),
            stable_flag=int(table.loc[int(art.test_idx[i]), "stable_flag"]),
        )
        for i in wrong
    ]

    report_path = state.paths.reports / f"classification_report_{component}.txt"
    saved_text = report_path.read_text(encoding="utf-8") if report_path.exists() else None

    parity = S.Parity(saved_accuracy=None, saved_f1_macro=None, matches=None)
    test = state.metrics.get("test_results")
    if test is not None and (test.component == component).any():
        row = test[test.component == component].iloc[0]
        sa, sf = float(row.accuracy), float(row.f1_macro)
        parity = S.Parity(saved_accuracy=sa, saved_f1_macro=sf, matches=abs(sa - acc) < 1e-9 and abs(sf - f1m) < 1e-9)

    return S.EvaluationResponse(
        component=component,  # type: ignore[arg-type]
        n_test=len(y),
        accuracy=acc,
        f1_macro=f1m,
        roc_auc=auc,
        ece=ece,
        n_errors=int((~correct).sum()),
        classes=_class_info(component, art.classes),
        confusion=cm.astype(int).tolist(),
        report=report,
        roc=roc,
        pr=pr,
        calibration=calibration,
        confidence_hist=hist,
        misclassified=misclassified,
        saved_report_text=saved_text,
        parity=parity,
    )


def generalization(state: AppState) -> S.GeneralizationResponse:
    state.require_metrics()
    test = state.metrics["test_results"]
    order = {c: i for i, c in enumerate(COMPONENTS)}
    rows = sorted(
        (
            S.GeneralizationRow(
                component=r.component,
                display_name=r.display_name,
                block_cv_f1_macro=float(r.block_cv_f1_macro),
                cv_f1_macro=float(r.cv_f1_macro),
                test_f1_macro=float(r.f1_macro),
                optimism=float(r.cv_f1_macro) - float(r.block_cv_f1_macro),
            )
            for r in test.itertuples()
        ),
        key=lambda g: order[g.component],
    )
    block = int(state.run_meta["block_size"]) if state.run_meta else 0  # type: ignore[arg-type]
    return S.GeneralizationResponse(block_size=block, rows=rows)


def timeline(state: AppState, component: str, start: int, end: int) -> S.TimelineResponse:
    art = state.component(component)
    frame = state.labels_frame()
    n = len(frame)
    if not (0 <= start <= end < n):
        raise ApiError(400, "invalid_window", f"Window must satisfy 0 <= start <= end <= {n - 1}.")
    values = frame[art.spec.column].to_numpy()
    points = [
        S.TimelinePoint(cycle=i, value=int(values[i]), split="test" if i in art.test_set else "train")
        for i in range(start, end + 1)
    ]
    return S.TimelineResponse(
        component=component,  # type: ignore[arg-type]
        start=start,
        end=end,
        n_cycles=n,
        classes=_class_info(component, art.classes),
        points=points,
    )


# --------------------------------------------------------------------------
# Errors, what-if, robustness
# --------------------------------------------------------------------------
def error_cases(state: AppState, component: str) -> S.ErrorCasesResponse:
    art = state.component(component)
    state.require_table()
    assert art.test_y is not None and art.test_proba is not None
    y, proba, pred = art.test_y, art.test_proba, art.test_pred
    wrong = np.where(pred != y)[0]
    if len(wrong):
        pick, mode = wrong, "errors"
    else:
        pick, mode = np.argsort(proba[np.arange(len(y)), y])[:10], "least_confident"
    cases = []
    for i in pick:
        m, _ = _margin_entropy(proba[i])
        cases.append(
            S.ErrorCase(
                cycle=int(art.test_idx[i]),
                true_value=art.classes[int(y[i])],
                predicted_value=art.classes[int(pred[i])],
                confidence=float(proba[i].max()),
                p_true=float(proba[i, y[i]]),
                margin=m,
            )
        )
    return S.ErrorCasesResponse(
        component=component, mode=mode, n_test=len(y), n_errors=len(wrong), cases=cases  # type: ignore[arg-type]
    )


def _top_features(state: AppState, art: ComponentArtefacts, n: int = 10) -> List[str]:
    imp = state.metrics.get("feature_importance")
    if imp is None:
        return []
    d = imp[imp.component == art.key]
    kind = "impurity" if (d.kind == "impurity").any() else "permutation"
    names = d[d.kind == kind].sort_values("importance", ascending=False).feature_name.tolist()
    table_cols = set(state.table.columns) if state.table is not None else set()
    return [f for f in names if f in table_cols][:n]


def error_profile(state: AppState, component: str, cycle_id: int) -> S.ErrorProfileResponse:
    art = state.component(component)
    table = state.require_table()
    state.check_cycle(cycle_id)
    top = _top_features(state, art)
    train = table.loc[art.train_idx]
    mu, sd = train[top].mean(), train[top].std().replace(0, 1)

    x = table.loc[[cycle_id], art.input_columns].to_numpy(dtype=np.float64)
    proba = art.model.predict_proba(x)[0]  # type: ignore[union-attr]
    true_v = int(table.loc[cycle_id, art.spec.column])
    pred_v = art.classes[int(np.argmax(proba))]
    margin, _ = _margin_entropy(proba)
    row = table.loc[cycle_id, top]
    rows = []
    for f in top:
        z_true = (train.loc[train[art.spec.column] == true_v, f].mean() - mu[f]) / sd[f]
        z_pred = (train.loc[train[art.spec.column] == pred_v, f].mean() - mu[f]) / sd[f] if pred_v != true_v else None
        rows.append(
            S.ZRow(
                feature=f,
                z_cycle=float((row[f] - mu[f]) / sd[f]),
                z_true_mean=float(z_true),
                z_pred_mean=_f(z_pred) if z_pred is not None else None,
            )
        )
    return S.ErrorProfileResponse(
        component=component,  # type: ignore[arg-type]
        cycle=cycle_id,
        true_value=true_v,
        predicted_value=pred_v,
        margin=margin,
        stable_flag=int(table.loc[cycle_id, "stable_flag"]),
        in_test_set=cycle_id in art.test_set,
        probabilities=_probabilities(art, proba),
        features=rows,
    )


def whatif_options(state: AppState, component: str, cycle_id: Optional[int]) -> S.WhatIfOptions:
    art = state.component(component)
    table = state.require_table()
    cyc = int(art.test_idx[0]) if cycle_id is None else cycle_id
    state.check_cycle(cyc)
    top = [f for f in _top_features(state, art) if f in art.input_columns]
    train = table.loc[art.train_idx]
    return S.WhatIfOptions(
        component=component,  # type: ignore[arg-type]
        cycle=cyc,
        true_value=int(table.loc[cyc, art.spec.column]),
        features=[
            S.WhatIfFeature(
                name=f,
                current=float(table.loc[cyc, f]),
                p01=float(train[f].quantile(0.01)),
                p99=float(train[f].quantile(0.99)),
            )
            for f in top
        ],
    )


def whatif(state: AppState, component: str, req: S.WhatIfRequest) -> S.WhatIfResponse:
    art = state.component(component)
    table = state.require_table()
    state.check_cycle(req.cycle)
    if req.feature not in art.input_columns:
        raise ApiError(400, "unknown_feature", f"'{req.feature}' is not an input of the {component} model.")
    if not math.isfinite(req.value):
        raise ApiError(400, "invalid_value", "The what-if value must be a finite number.")
    x0 = table.loc[[req.cycle], art.input_columns].to_numpy(dtype=np.float64)
    x1 = x0.copy()
    x1[0, art.input_columns.index(req.feature)] = req.value
    p0 = art.model.predict_proba(x0)[0]  # type: ignore[union-attr]
    p1 = art.model.predict_proba(x1)[0]  # type: ignore[union-attr]
    a, b = art.classes[int(np.argmax(p0))], art.classes[int(np.argmax(p1))]
    return S.WhatIfResponse(
        component=component,  # type: ignore[arg-type]
        cycle=req.cycle,
        feature=req.feature,
        original_value=float(x0[0, art.input_columns.index(req.feature)]),
        value=req.value,
        true_value=int(table.loc[req.cycle, art.spec.column]),
        original=_probabilities(art, p0),
        modified=_probabilities(art, p1),
        original_prediction=a,
        modified_prediction=b,
        changed=a != b,
    )


def noise(state: AppState, component: str) -> S.NoiseResponse:
    """Gaussian noise stress test - identical procedure to MLapp.noise_curve."""
    art = state.component(component)
    state.require_table()

    def compute() -> S.NoiseResponse:
        X, y = state.xy(art, art.test_idx)
        Xtr, _ = state.xy(art, art.train_idx)
        sd = np.nanstd(Xtr, axis=0)
        rng = np.random.default_rng(NOISE_SEED)
        points = []
        for s in NOISE_LEVELS:
            scores = []
            for _ in range(NOISE_REPEATS if s > 0 else 1):
                Xn = X + rng.normal(0, 1, X.shape) * sd * s
                scores.append(f1_score(y, art.model.predict(Xn), average="macro"))  # type: ignore[union-attr]
            points.append(S.NoisePoint(level=s, f1_macro=float(np.mean(scores)), min=float(np.min(scores)), max=float(np.max(scores))))
        return S.NoiseResponse(component=component, repeats=NOISE_REPEATS, seed=NOISE_SEED, points=points)  # type: ignore[arg-type]

    return state.cached(f"noise:{component}", compute)  # type: ignore[no-any-return]


def selective(state: AppState, component: str) -> S.SelectiveResponse:
    art = state.component(component)
    state.require_table()
    assert art.test_y is not None and art.test_proba is not None
    conf = art.test_proba.max(axis=1)
    correct = art.test_pred == art.test_y
    grid = []
    for t in np.round(np.arange(0.30, 1.0001, 0.01), 2):
        answered = conf >= t
        grid.append(
            S.SelectivePoint(
                threshold=float(t),
                coverage=float(answered.mean()),
                answered=int(answered.sum()),
                accuracy=float(correct[answered].mean()) if answered.any() else None,
                errors_answered=int((~correct & answered).sum()),
            )
        )
    return S.SelectiveResponse(
        component=component, n_test=len(conf), n_errors=int((~correct).sum()), grid=grid  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Static figures (only the ones the frontend does not redraw)
# --------------------------------------------------------------------------
def figure_whitelist(state: AppState) -> Dict[str, "tuple[str, Path]"]:
    fig = state.paths.figures
    out: Dict[str, "tuple[str, Path]"] = {
        "correlation_heatmap": ("Correlation heatmap of engineered features", fig / "correlation_heatmap.png"),
    }
    for c in COMPONENTS:
        name = TARGET_SPECS[c].display_name
        out[f"pca_{c}"] = (f"PCA projection - {name}", fig / "pca" / f"pca_{c}.png")
        out[f"pca_explained_variance_{c}"] = (f"PCA explained variance - {name}", fig / "pca" / f"pca_explained_variance_{c}.png")
    return out


def figures(state: AppState) -> S.FiguresResponse:
    return S.FiguresResponse(
        figures=[
            S.Figure(key=k, title=title, url=f"/api/figures/{k}")
            for k, (title, path) in figure_whitelist(state).items()
            if path.exists()
        ]
    )
