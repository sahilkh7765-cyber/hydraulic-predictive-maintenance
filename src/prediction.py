"""
prediction.py
=============
Inference layer: turn one raw operating cycle into four component-condition
predictions using the exact pipelines fitted in `train.py`.

WHY A DEDICATED MODULE
    The dashboard must never re-derive preprocessing or feature selection. If
    the app scaled its input with a freshly fitted scaler, the numbers reaching
    the classifier would be on a different scale from the ones it was trained
    on and the predictions would be quietly wrong. Every function here loads the
    saved `Pipeline`, which already contains the fitted imputer, scaler and
    selectors, and feeds it a feature vector produced by the *same*
    feature-engineering code used during training.

INPUT ACCEPTED
    * a cycle index into the original dataset, or
    * a dict of {sensor: 1-D array of that cycle's samples}, or
    * a raw-sensor CSV with one column per sensor (see `load_cycle_csv`).

OUTPUT
    Structured dictionaries containing the predicted condition value, its
    documented label, a severity word, and class probabilities where the model
    supports them.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (  # noqa: E402
    RAW_DATA_DIR,
    SENSOR_INDEX,
    SENSOR_NAMES,
    TARGET_SPECS,
    load_sensor_matrix,
)
from feature_engineering import extract_features_from_cycle  # noqa: E402
from utils import COMPONENTS, MODELS_DIR, load_json  # noqa: E402

CycleData = Dict[str, np.ndarray]


class ModelNotTrainedError(RuntimeError):
    """Raised when a saved model is requested before training has been run.

    A dedicated exception lets the Streamlit app show a one-line instruction
    instead of an unexplained traceback.
    """


@lru_cache(maxsize=8)
def load_component_bundle(component: str) -> Dict[str, object]:
    """Load and cache one component's pipeline, encoder and metadata.

    WHY CACHED: Streamlit re-runs the whole script on every widget interaction.
    Without caching, four joblib files would be deserialised on every click.

    Input : component key ("cooler", "valve", "pump", "accumulator").
    Output: dict with "model", "encoder", "metadata".
    Raises: ModelNotTrainedError when the artefacts are missing.
    """
    directory = MODELS_DIR / component
    model_path = directory / "model.joblib"
    encoder_path = directory / "label_encoder.joblib"
    metadata_path = directory / "metadata.json"

    if not (model_path.exists() and encoder_path.exists() and metadata_path.exists()):
        raise ModelNotTrainedError(
            f"No trained model for '{component}'. Run `python src/train.py` first."
        )
    return {
        "model": joblib.load(model_path),
        "encoder": joblib.load(encoder_path),
        "metadata": load_json(metadata_path),
    }


def models_are_available() -> bool:
    """Return True when all four components have a saved model on disk."""
    return all((MODELS_DIR / c / "model.joblib").exists() for c in COMPONENTS)


# --------------------------------------------------------------------------
# Building a cycle from different sources
# --------------------------------------------------------------------------
def load_cycle_from_dataset(
    cycle_id: int, raw_dir: Path = RAW_DATA_DIR, sensors: Iterable[str] = SENSOR_NAMES
) -> CycleData:
    """Read one cycle's raw samples for every sensor from the dataset files.

    Input : 0-based cycle index, the raw data directory.
    Output: dict sensor -> 1-D float array of that sensor's samples.
    Raises: IndexError when the cycle index is outside the dataset.
    """
    cycle: CycleData = {}
    for sensor in sensors:
        matrix = load_sensor_matrix(sensor, raw_dir)
        if not 0 <= cycle_id < matrix.shape[0]:
            raise IndexError(
                f"cycle_id {cycle_id} is outside the dataset (0 .. {matrix.shape[0] - 1})."
            )
        cycle[sensor] = matrix[cycle_id].astype(np.float64)
        del matrix
    return cycle


def load_cycle_row(
    cycle_id: int, raw_dir: Path = RAW_DATA_DIR, sensors: Iterable[str] = SENSOR_NAMES
) -> CycleData:
    """Read ONE cycle without loading whole sensor matrices into memory.

    WHY THIS EXISTS ALONGSIDE `load_cycle_from_dataset`: the dashboard scores a
    single cycle chosen by the user. Reading all seven 100 Hz matrices to
    extract one row each would move ~370 MB and take tens of seconds on every
    click. Because each row of the file *is* one cycle, `skiprows` + `nrows`
    lets the C parser jump straight to the requested line.

    Input : 0-based cycle index, raw data directory.
    Output: dict sensor -> 1-D float array.
    Raises: IndexError if the cycle index does not exist in a sensor file.
    """
    cycle: CycleData = {}
    for sensor in sensors:
        path = raw_dir / f"{sensor}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Sensor file not found: {path}")
        frame = pd.read_csv(
            path, sep="\t", header=None, skiprows=int(cycle_id), nrows=1, dtype=np.float64
        )
        if frame.empty:
            raise IndexError(f"cycle_id {cycle_id} does not exist in {sensor}.txt.")
        values = frame.to_numpy(dtype=np.float64).ravel()
        expected = SENSOR_INDEX[sensor].n_samples
        if values.size != expected:
            raise ValueError(
                f"{sensor}: row {cycle_id} has {values.size} samples, expected {expected}."
            )
        cycle[sensor] = values
    return cycle


def load_cycle_csv(source: Union[str, Path, "pd.io.parsers.TextFileReader", object]) -> CycleData:
    """Parse a raw-sensor CSV describing ONE operating cycle.

    EXPECTED FORMAT
        One column per sensor, named exactly as in the dataset (PS1 ... SE).
        Each column holds that sensor's samples for the cycle, in order, from
        the first row downwards. Because the sampling rates differ, the columns
        have different lengths: the shorter ones are simply padded with empty
        cells at the bottom. `export_cycle_csv` writes exactly this layout, so
        the easiest way to build a valid file is to export a known cycle and
        replace the numbers.

    VALIDATION PERFORMED (each failure raises ValueError with a readable message)
        * file is empty or unparseable
        * required sensor columns are missing
        * a column has the wrong number of samples for its sampling rate
        * a column contains no numeric values

    Input : a path or file-like object.
    Output: dict sensor -> 1-D float array.
    """
    try:
        frame = pd.read_csv(source)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The uploaded file is empty.") from error
    except Exception as error:  # noqa: BLE001 - surfaced to the user as text
        raise ValueError(f"The file could not be read as CSV: {error}") from error

    if frame.empty:
        raise ValueError("The uploaded CSV contains no rows.")

    # Column names are matched case-insensitively and whitespace-tolerantly,
    # because hand-edited exports often pick up stray spaces.
    normalised = {str(col).strip().upper(): col for col in frame.columns}
    missing = [s for s in SENSOR_NAMES if s.upper() not in normalised]
    if missing:
        raise ValueError(
            "The CSV is missing these required sensor columns: "
            + ", ".join(missing)
            + ". Expected one column per sensor: "
            + ", ".join(SENSOR_NAMES)
            + "."
        )

    cycle: CycleData = {}
    problems: List[str] = []
    for sensor in SENSOR_NAMES:
        column = frame[normalised[sensor.upper()]]
        values = pd.to_numeric(column, errors="coerce").dropna().to_numpy(dtype=np.float64)
        expected = SENSOR_INDEX[sensor].n_samples
        if values.size == 0:
            problems.append(f"{sensor}: no numeric values found")
        elif values.size != expected:
            problems.append(
                f"{sensor}: {values.size} samples, expected {expected} "
                f"({SENSOR_INDEX[sensor].sampling_rate_hz} Hz x 60 s)"
            )
        else:
            cycle[sensor] = values

    if problems:
        raise ValueError(
            "The CSV does not describe a valid 60-second cycle:\n  - " + "\n  - ".join(problems)
        )
    return cycle


def export_cycle_csv(cycle: CycleData, path: Union[str, Path]) -> Path:
    """Write a cycle to the CSV layout that `load_cycle_csv` accepts.

    Useful both as a template for users and as a round-trip test of the parser.
    """
    frame = pd.DataFrame({sensor: pd.Series(values) for sensor, values in cycle.items()})
    path = Path(path)
    frame.to_csv(path, index=False)
    return path


def features_from_cycle(cycle: CycleData) -> pd.DataFrame:
    """Run the training-time feature engineering on a single cycle.

    Input : dict sensor -> samples.
    Output: one-row DataFrame of engineered features.
    """
    missing = [s for s in SENSOR_NAMES if s not in cycle]
    if missing:
        raise ValueError(
            "Cannot extract features - missing sensors: " + ", ".join(missing)
        )
    return extract_features_from_cycle(cycle)


def _align_to_model(features: pd.DataFrame, metadata: Dict[str, object]) -> np.ndarray:
    """Re-order and subset a feature row to exactly the model's input columns.

    WHY THIS MATTERS: a scikit-learn Pipeline consumes a positional array. If
    the columns arrive in a different order from training, every fitted scale
    factor and every selection mask is applied to the wrong feature and the
    prediction is meaningless - without raising any error. This function makes
    the mismatch impossible by rebuilding the vector from the stored column
    list.

    Input : engineered features (one row) and the component's metadata.
    Output: 2-D array of shape (1, n_input_features).
    Raises: ValueError listing any feature the model needs but cannot find.
    """
    columns: List[str] = list(metadata["input_feature_columns"])  # type: ignore[arg-type]
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} feature(s) required by the model could not be computed "
            f"from this input (for example: {', '.join(missing[:5])})."
        )
    return features[columns].to_numpy(dtype=np.float64)


def predict_component(
    component: str, features: Optional[pd.DataFrame] = None, cycle: Optional[CycleData] = None
) -> Dict[str, object]:
    """Predict one component's condition.

    Input : either an already-engineered feature row or a raw cycle.
    Output: dict with
            component, display_name, predicted_value, predicted_label,
            severity, confidence, probabilities (value -> probability),
            model, feature_set, n_selected_features.
    Raises: ModelNotTrainedError, ValueError on malformed input.
    """
    if features is None:
        if cycle is None:
            raise ValueError("Provide either `features` or `cycle`.")
        features = features_from_cycle(cycle)

    bundle = load_component_bundle(component)
    model = bundle["model"]
    encoder = bundle["encoder"]
    metadata: Dict[str, object] = bundle["metadata"]  # type: ignore[assignment]
    spec = TARGET_SPECS[component]

    X = _align_to_model(features, metadata)
    encoded = int(model.predict(X)[0])
    predicted_value = int(encoder.inverse_transform([encoded])[0])

    probabilities: Optional[Dict[int, float]] = None
    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probabilities = {
            int(encoder.inverse_transform([index])[0]): float(value)
            for index, value in enumerate(proba)
        }
        confidence = float(np.max(proba))

    return {
        "component": component,
        "display_name": spec.display_name,
        "unit": spec.unit,
        "predicted_value": predicted_value,
        "predicted_label": spec.value_labels.get(predicted_value, "unknown"),
        "severity": spec.severity.get(predicted_value, "Unknown"),
        "confidence": confidence,
        "probabilities": probabilities,
        "class_labels": {int(v): spec.value_labels.get(int(v), str(v)) for v in encoder.classes_},
        "model": metadata["selected_model"],
        "feature_set": metadata["feature_set"],
        "n_selected_features": metadata["n_selected_features"],
        "selected_feature_names": metadata["selected_feature_names"],
    }


def predict_cooler(features: Optional[pd.DataFrame] = None, cycle: Optional[CycleData] = None) -> Dict[str, object]:
    """Predict cooler condition (3 %, 20 %, 100 % efficiency)."""
    return predict_component("cooler", features=features, cycle=cycle)


def predict_valve(features: Optional[pd.DataFrame] = None, cycle: Optional[CycleData] = None) -> Dict[str, object]:
    """Predict valve switching condition (73, 80, 90, 100 %)."""
    return predict_component("valve", features=features, cycle=cycle)


def predict_pump(features: Optional[pd.DataFrame] = None, cycle: Optional[CycleData] = None) -> Dict[str, object]:
    """Predict internal pump leakage level (0, 1, 2)."""
    return predict_component("pump", features=features, cycle=cycle)


def predict_accumulator(features: Optional[pd.DataFrame] = None, cycle: Optional[CycleData] = None) -> Dict[str, object]:
    """Predict hydraulic accumulator gas pre-charge pressure (90-130 bar)."""
    return predict_component("accumulator", features=features, cycle=cycle)


def predict_system_condition(
    features: Optional[pd.DataFrame] = None, cycle: Optional[CycleData] = None
) -> Dict[str, object]:
    """Predict all four component conditions from one cycle.

    EFFICIENCY NOTE: the engineered features are computed once and reused for
    all four models, because every component model reads from the same feature
    table. Extracting features four times would repeat the FFT work needlessly.

    Input : engineered features or a raw cycle.
    Output: dict with per-component predictions, an overall system status and
            the features that were used (so the caller can display them).
    """
    if features is None:
        if cycle is None:
            raise ValueError("Provide either `features` or `cycle`.")
        features = features_from_cycle(cycle)

    predictions = {
        component: predict_component(component, features=features) for component in COMPONENTS
    }

    # Overall status = worst individual severity. This is a deliberately simple
    # aggregation: a system is only as healthy as its weakest component, and any
    # weighted "health score" would need maintenance-cost data the dataset does
    # not contain.
    order = {"Normal": 0, "Degraded": 1, "Critical": 2, "Unknown": 1}
    worst = max(predictions.values(), key=lambda p: order.get(str(p["severity"]), 1))
    confidences = [p["confidence"] for p in predictions.values() if p["confidence"] is not None]

    return {
        "components": predictions,
        "system_status": worst["severity"],
        "driving_component": worst["display_name"],
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
        "features": features,
    }


def explain_prediction(component: str, features: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the model's most important features together with this cycle's values.

    WHY: a condition-monitoring system that says "valve: close to total failure"
    and nothing else is not actionable. Showing which engineered quantities the
    model relies on, and what they measured for this particular cycle, lets an
    engineer sanity-check the alarm against the physics.

    Input : component key, the engineered feature row, number of features.
    Output: DataFrame with rank, feature_name, value and (when the underlying
            model exposes one) importance.
    """
    bundle = load_component_bundle(component)
    metadata: Dict[str, object] = bundle["metadata"]  # type: ignore[assignment]
    model = bundle["model"]
    names: List[str] = list(metadata["selected_feature_names"])  # type: ignore[arg-type]

    estimator = model.named_steps["model"]
    importances: Optional[np.ndarray] = None
    if hasattr(estimator, "feature_importances_"):
        candidate = np.asarray(estimator.feature_importances_, dtype=float)
        if len(candidate) == len(names):
            importances = candidate
    elif hasattr(estimator, "coef_"):
        # For linear models the magnitude of the (scaled) coefficient is the
        # closest analogue to an importance; averaged over classes for a
        # multiclass model.
        candidate = np.abs(np.asarray(estimator.coef_, dtype=float)).mean(axis=0)
        if len(candidate) == len(names):
            importances = candidate

    order = np.argsort(importances)[::-1] if importances is not None else np.arange(len(names))
    rows: List[Dict[str, object]] = []
    for rank, index in enumerate(order[:top_n], start=1):
        name = names[index]
        rows.append(
            {
                "rank": rank,
                "feature_name": name,
                "value": float(features[name].iloc[0]) if name in features.columns else np.nan,
                "importance": float(importances[index]) if importances is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test: score cycle 0 with whatever models are currently saved.
    if not models_are_available():
        print("No trained models found. Run `python src/train.py` first.")
    else:
        cycle_data = load_cycle_from_dataset(0)
        result = predict_system_condition(cycle=cycle_data)
        print(f"System status: {result['system_status']} (driven by {result['driving_component']})")
        for key, prediction in result["components"].items():  # type: ignore[union-attr]
            confidence = prediction["confidence"]
            confidence_text = f"{confidence:.3f}" if confidence is not None else "n/a"
            print(
                f"  {prediction['display_name']:<26} "
                f"{prediction['predicted_value']:>4} {prediction['unit']:<6} "
                f"{prediction['predicted_label']:<28} confidence {confidence_text}"
            )
