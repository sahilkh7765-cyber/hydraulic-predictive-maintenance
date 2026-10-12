"""
app.py
======
Streamlit predictive-maintenance dashboard for the hydraulic test rig.

WHAT IT DOES
    Six sections, as specified by the project brief:
      1. Project overview
      2. Dataset explorer
      3. Feature engineering
      4. Condition prediction (pick a cycle, or upload a raw-sensor CSV)
      5. Maintenance status for cooler / valve / pump / accumulator
      6. Explainability

WHAT IT DELIBERATELY DOES NOT DO
    It never trains anything. Every prediction comes from the pipelines saved by
    `src/train.py`, loaded once and cached. That is not just a speed decision:
    re-fitting a scaler on whatever data the app happens to hold would put the
    classifier's inputs on a different scale from training and silently corrupt
    every prediction.

RUN
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# The project keeps its modules in src/; add it to the path before importing.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import (  # noqa: E402
    SENSOR_INDEX,
    SENSOR_NAMES,
    TARGET_SPECS,
    get_dataset_summary,
    load_profile,
)
from prediction import (  # noqa: E402
    ModelNotTrainedError,
    explain_prediction,
    features_from_cycle,
    load_component_bundle,
    load_cycle_csv,
    load_cycle_row,
    models_are_available,
    predict_system_condition,
)
from utils import COMPONENTS, FEATURES_DIR, METRICS_DIR, MODELS_DIR, load_json  # noqa: E402

st.set_page_config(
    page_title="Hydraulic Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
)

# Colour coding for the maintenance status cards. The three words map directly
# onto the dataset's own documented condition levels - no invented thresholds.
SEVERITY_COLOURS = {
    "Normal": "#1b7f3b",
    "Degraded": "#c77700",
    "Critical": "#b3261e",
    "Unknown": "#5f6368",
}


# ==========================================================================
# Cached loaders
# ==========================================================================
@st.cache_data(show_spinner=False)
def cached_dataset_summary() -> Optional[Dict[str, object]]:
    """Dataset structure, read from the raw files once per session."""
    try:
        return get_dataset_summary()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def cached_profile() -> Optional[pd.DataFrame]:
    """Cycle-wise condition labels."""
    try:
        return load_profile()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def cached_run_metadata() -> Optional[Dict[str, object]]:
    """Training-run metadata written by train.py."""
    path = MODELS_DIR / "run_metadata.json"
    return load_json(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def cached_metrics() -> Dict[str, Optional[pd.DataFrame]]:
    """Cross-validation and test metrics, if they have been produced."""
    def read(path: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(path) if path.exists() else None

    return {
        "cv": read(METRICS_DIR / "model_results.csv"),
        "test": read(METRICS_DIR / "test_results.csv"),
        "selected": read(FEATURES_DIR / "selected_features.csv"),
    }


@st.cache_data(show_spinner="Reading cycle from the dataset...")
def cached_cycle(cycle_id: int) -> Dict[str, np.ndarray]:
    """One cycle's raw samples, read row-wise so the app stays responsive."""
    return load_cycle_row(int(cycle_id))


def models_ready() -> bool:
    """True when all four saved pipelines are present."""
    return models_are_available()


def require_models() -> bool:
    """Show a clear instruction instead of a traceback when models are missing."""
    if models_ready():
        return True
    st.error(
        "No trained models found in `models/`.\n\n"
        "Run the training pipeline first:\n\n"
        "```bash\npython src/train.py\npython src/evaluate.py\n```"
    )
    return False


def require_dataset() -> bool:
    """Check that the raw dataset has been extracted into data/raw/."""
    if cached_dataset_summary() is not None:
        return True
    st.error(
        "The raw dataset was not found in `data/raw/`.\n\n"
        "Extract the UCI ZIP there so that `PS1.txt ... profile.txt` sit directly "
        "inside `data/raw/`."
    )
    return False


# ==========================================================================
# Section 1 - Project overview
# ==========================================================================
def section_overview() -> None:
    st.header("1. Project overview")

    st.markdown(
        """
**Predictive maintenance** means judging the health of a machine from the signals
it produces while it runs, so that a component is serviced *before* it fails
rather than after. For a hydraulic power unit, an unplanned failure stops the
whole production line it feeds, so early warning is worth a great deal.

**The hydraulic system here** is a laboratory test rig with a primary working
circuit and a secondary cooling-filtration circuit, connected through the oil
tank. It repeats a constant 60-second load cycle over and over while pressures,
flows, temperatures, motor power and vibration are recorded. Four components are
deliberately degraded to known levels: the **cooler**, a **valve**, the **pump**
(internal leakage) and the **hydraulic accumulator**.

**Why feature engineering is the focus.** One cycle contains 43,680 raw sensor
samples, but there are only 2,205 cycles. Feeding raw samples to a classifier
means far more columns than rows, and the three different sampling rates mean
column *k* of a pressure channel and column *k* of a temperature channel are not
even the same instant in time. Instead, each cycle is summarised by statistics
that describe the *shape* of every signal, plus cross-channel quantities that
encode hydraulic physics - a pressure differential, a flow imbalance, an
electro-hydraulic efficiency.

**What is deliberately not claimed.** This dataset annotates a *condition level*
per cycle. It contains no maintenance timestamps and no failure times, so
Remaining Useful Life cannot be computed and is not reported anywhere.
        """
    )

    metadata = cached_run_metadata()
    if metadata:
        blocks: Dict[str, int] = metadata["feature_blocks"]  # type: ignore[assignment]
        columns = st.columns(4)
        columns[0].metric("Operating cycles", f"{metadata['dataset_summary']['n_cycles']:,}")  # type: ignore[index]
        columns[1].metric("Raw values per cycle", f"{metadata['dataset_summary']['total_raw_dimensions']:,}")  # type: ignore[index]
        columns[2].metric("Engineered features", sum(blocks.values()))
        columns[3].metric("Components predicted", len(COMPONENTS))


# ==========================================================================
# Section 2 - Dataset explorer
# ==========================================================================
def section_dataset() -> None:
    st.header("2. Dataset explorer")
    if not require_dataset():
        return

    summary = cached_dataset_summary()
    profile = cached_profile()
    assert summary is not None and profile is not None

    columns = st.columns(4)
    columns[0].metric("Cycles", f"{summary['n_cycles']:,}")
    columns[1].metric("Sensors", summary["n_sensors"])
    columns[2].metric("Cycle duration", f"{summary['cycle_duration_s']} s")
    columns[3].metric("Raw dimensions/cycle", f"{summary['total_raw_dimensions']:,}")

    st.subheader("Sensors and sampling frequencies")
    sensor_frame = pd.DataFrame(summary["sensors"])  # type: ignore[arg-type]
    sensor_frame = sensor_frame.rename(
        columns={
            "sensor": "Sensor", "quantity": "Physical quantity", "unit": "Unit",
            "sampling_rate_hz": "Sampling rate [Hz]", "samples_per_cycle": "Samples per cycle",
            "group": "Group", "virtual": "Virtual channel",
        }
    )
    st.dataframe(sensor_frame, use_container_width=True, hide_index=True)
    st.caption(
        "Virtual channels (CE, CP, SE) are quantities the rig computes rather than "
        "measures directly, so features built on them inherit the rig's own model."
    )

    st.subheader("Condition distributions")
    tabs = st.tabs([TARGET_SPECS[key].display_name for key in COMPONENTS])
    for tab, key in zip(tabs, COMPONENTS):
        spec = TARGET_SPECS[key]
        with tab:
            counts = profile[spec.column].value_counts().sort_index()
            frame = pd.DataFrame(
                {
                    f"Value [{spec.unit}]": counts.index,
                    "Documented meaning": [
                        spec.value_labels.get(int(v), "unknown") for v in counts.index
                    ],
                    "Cycles": counts.to_numpy(),
                    "Share [%]": (100.0 * counts / counts.sum()).round(2).to_numpy(),
                }
            )
            left, right = st.columns([1.3, 1])
            left.dataframe(frame, use_container_width=True, hide_index=True)
            right.bar_chart(frame.set_index("Documented meaning")["Cycles"])
            ratio = counts.max() / counts.min()
            right.caption(
                f"Imbalance ratio (largest / smallest class): **{ratio:.2f}**. "
                "This is why macro F1, which weights every class equally, is the "
                "headline metric rather than accuracy."
            )

    with st.expander("Stable flag (metadata, not a prediction target)"):
        stable = profile["stable_flag"].value_counts().sort_index()
        st.write(
            f"- `0` conditions were stable: **{int(stable.get(0, 0))}** cycles\n"
            f"- `1` static conditions might not have been reached: **{int(stable.get(1, 0))}** cycles"
        )
        st.caption(
            "The flag describes the rig's state, not a component's health, so it is "
            "kept as metadata rather than modelled."
        )


# ==========================================================================
# Section 3 - Feature engineering
# ==========================================================================
def section_features() -> None:
    st.header("3. Feature engineering")
    metadata = cached_run_metadata()
    if metadata is None:
        st.warning("Run `python src/train.py` to populate this section.")
        return

    summary: Dict[str, object] = metadata["dataset_summary"]  # type: ignore[assignment]
    blocks: Dict[str, int] = metadata["feature_blocks"]  # type: ignore[assignment]
    cleaning: Dict[str, object] = metadata["cleaning"]  # type: ignore[assignment]

    columns = st.columns(4)
    columns[0].metric("Raw dimensions per cycle", f"{summary['total_raw_dimensions']:,}")
    columns[1].metric("Engineered features", sum(blocks.values()))
    columns[2].metric("After cleaning", cleaning["n_output"])
    columns[3].metric(
        "Compression",
        f"{int(summary['total_raw_dimensions']) / max(sum(blocks.values()), 1):.0f}x",
    )

    st.subheader("Feature categories")
    descriptions = {
        "statistical": "Distribution shape within the cycle: mean, median, std, variance, min, max, range, RMS, MAV, skewness, kurtosis, quartiles, IQR, coefficient of variation.",
        "time_domain": "How the signal evolves: peak-to-peak, peak amplitude, energy, first-difference statistics, linear slope, zero crossings, peak count.",
        "rolling": "Local behaviour in a 1-second window, sized from each sensor's own sampling rate. Only for channels at 10 Hz and above.",
        "frequency": "A deliberately small FFT summary: dominant frequency, spectral energy, spectral centroid, log-spaced band energy ratios. Only for channels at 10 Hz and above.",
        "domain": "Hydraulic relationships between channels: pressure differentials and ratios, flow imbalance, hydraulic power, electro-hydraulic efficiency, specific power, temperature spreads, cooling conductance, vibration per unit power.",
    }
    frame = pd.DataFrame(
        {
            "Block": [b.replace("_", " ").capitalize() for b in blocks],
            "Features generated": list(blocks.values()),
            "What it captures": [descriptions.get(b, "") for b in blocks],
        }
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)

    st.subheader("Features removed during cleaning")
    st.write(
        f"- constant: **{cleaning['n_dropped_constant']}**\n"
        f"- near-zero variance: **{cleaning['n_dropped_near_zero_variance']}**\n"
        f"- exact duplicates: **{cleaning['n_dropped_duplicate']}**\n"
        f"- remaining: **{cleaning['n_output']}**"
    )

    st.subheader("Selected features per component")
    selected: Dict[str, object] = metadata["selected_per_component"]  # type: ignore[assignment]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Component": TARGET_SPECS[key].display_name,
                    "Model": value["model"],
                    "Feature set": value["feature_set"],
                    "Features used": value["n_selected_features"],
                    "CV macro F1": round(float(value["cv_f1_macro"]), 4),
                }
                for key, value in selected.items()
            ]
        ),
        use_container_width=True, hide_index=True,
    )

    metrics = cached_metrics()
    if metrics["selected"] is not None:
        with st.expander("Which engineered features survived selection?"):
            component = st.selectbox(
                "Component", COMPONENTS,
                format_func=lambda key: TARGET_SPECS[key].display_name,
                key="feature_component",
            )
            subset = metrics["selected"]
            subset = subset[subset["component"] == component]
            st.dataframe(subset, use_container_width=True, hide_index=True)


# ==========================================================================
# Section 4 + 5 - Prediction and maintenance status
# ==========================================================================
def render_status_card(prediction: Dict[str, object]) -> None:
    """Render one component's predicted condition as a coloured card."""
    severity = str(prediction["severity"])
    colour = SEVERITY_COLOURS.get(severity, SEVERITY_COLOURS["Unknown"])
    confidence = prediction["confidence"]
    confidence_text = f"{float(confidence) * 100:.1f} % confidence" if confidence is not None else "confidence not available"
    st.markdown(
        f"""
<div style="border-left: 6px solid {colour}; background: rgba(0,0,0,0.03);
            padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.7rem;">
  <div style="font-size: 0.85rem; opacity: 0.75;">{prediction['display_name']}</div>
  <div style="font-size: 1.35rem; font-weight: 600; color: {colour};">{severity}</div>
  <div style="font-size: 0.95rem;">{prediction['predicted_value']} {prediction['unit']} &mdash;
      {prediction['predicted_label']}</div>
  <div style="font-size: 0.8rem; opacity: 0.7;">{confidence_text}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def section_prediction() -> None:
    st.header("4. Condition prediction")
    if not require_models():
        return

    source = st.radio(
        "Input source",
        ["Select a cycle from the dataset", "Upload a raw-sensor CSV"],
        horizontal=True,
    )

    cycle: Optional[Dict[str, np.ndarray]] = None
    true_labels: Optional[pd.Series] = None

    if source == "Select a cycle from the dataset":
        if not require_dataset():
            return
        profile = cached_profile()
        assert profile is not None
        cycle_id = st.number_input(
            "Cycle number (0-based)", min_value=0, max_value=len(profile) - 1,
            value=0, step=1,
        )
        if st.button("Predict this cycle", type="primary"):
            try:
                cycle = cached_cycle(int(cycle_id))
                true_labels = profile.iloc[int(cycle_id)]
            except (FileNotFoundError, IndexError, ValueError) as error:
                st.error(f"Could not read cycle {cycle_id}: {error}")
                return
    else:
        st.caption(
            "One column per sensor, named exactly as in the dataset "
            f"({', '.join(SENSOR_NAMES)}). Each column holds that sensor's samples "
            "for a single 60-second cycle: 6000 rows for 100 Hz channels, 600 for "
            "10 Hz, 60 for 1 Hz. Shorter columns are simply padded with empty cells."
        )
        with st.expander("Required sample count per sensor"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Sensor": name,
                            "Sampling rate [Hz]": SENSOR_INDEX[name].sampling_rate_hz,
                            "Samples required": SENSOR_INDEX[name].n_samples,
                        }
                        for name in SENSOR_NAMES
                    ]
                ),
                use_container_width=True, hide_index=True,
            )
        uploaded = st.file_uploader("Raw-sensor CSV for one cycle", type=["csv"])
        if uploaded is not None:
            try:
                cycle = load_cycle_csv(uploaded)
            except ValueError as error:
                st.error(str(error))
                return
            except Exception as error:  # noqa: BLE001 - shown as readable text
                st.error(f"The file could not be processed: {error}")
                return

    if cycle is None:
        st.info("Choose a cycle or upload a CSV to run a prediction.")
        return

    # ---- Feature extraction + prediction --------------------------------
    try:
        with st.spinner("Extracting features and scoring the four models..."):
            features = features_from_cycle(cycle)
            result = predict_system_condition(features=features)
    except ModelNotTrainedError as error:
        st.error(str(error))
        return
    except ValueError as error:
        st.error(f"Prediction failed: {error}")
        return

    st.success(
        f"Features extracted ({features.shape[1]} engineered values) and scored "
        "with the saved pipelines."
    )

    # ---- Section 5: maintenance status ----------------------------------
    st.header("5. Maintenance status")
    status = str(result["system_status"])
    st.markdown(
        f"### Overall system status: "
        f"<span style='color:{SEVERITY_COLOURS.get(status, '#5f6368')}'>{status}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Driven by: {result['driving_component']}. The overall status is simply the "
        "worst individual component status - a system is only as healthy as its "
        "weakest part, and the dataset contains no maintenance-cost information "
        "that would justify a weighted health score."
    )

    predictions: Dict[str, Dict[str, object]] = result["components"]  # type: ignore[assignment]
    columns = st.columns(4)
    for column, key in zip(columns, COMPONENTS):
        with column:
            render_status_card(predictions[key])

    # ---- Probabilities ---------------------------------------------------
    with st.expander("Class probabilities"):
        for key in COMPONENTS:
            prediction = predictions[key]
            probabilities = prediction["probabilities"]
            st.markdown(f"**{prediction['display_name']}**")
            if not probabilities:
                st.caption("This model does not expose calibrated probabilities.")
                continue
            labels: Dict[int, str] = prediction["class_labels"]  # type: ignore[assignment]
            frame = pd.DataFrame(
                {
                    "Condition": [f"{v} - {labels.get(int(v), '')}" for v in probabilities],
                    "Probability": [round(float(p), 4) for p in probabilities.values()],
                }
            ).sort_values("Probability", ascending=False)
            st.dataframe(frame, use_container_width=True, hide_index=True)

    # ---- Ground truth, when the cycle came from the dataset --------------
    if true_labels is not None:
        with st.expander("Compare with the dataset's recorded condition"):
            rows = []
            for key in COMPONENTS:
                spec = TARGET_SPECS[key]
                actual = int(true_labels[spec.column])
                predicted = int(predictions[key]["predicted_value"])
                rows.append(
                    {
                        "Component": spec.display_name,
                        "Predicted": f"{predicted} ({spec.value_labels.get(predicted, '')})",
                        "Recorded": f"{actual} ({spec.value_labels.get(actual, '')})",
                        "Match": "yes" if predicted == actual else "no",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "Note: most dataset cycles were part of model training, so agreement "
                "here is not an unbiased accuracy estimate. The honest numbers are in "
                "`outputs/metrics/test_results.csv`."
            )

    # ---- Section 6: explainability --------------------------------------
    section_explainability(features, predictions, cycle)


def section_explainability(
    features: pd.DataFrame,
    predictions: Dict[str, Dict[str, object]],
    cycle: Dict[str, np.ndarray],
) -> None:
    st.header("6. Explainability")
    st.caption(
        "An alarm that cannot be checked against the physics is not actionable. "
        "These are the engineered quantities each model relies on, together with "
        "what they measured for this cycle."
    )

    component = st.selectbox(
        "Component", COMPONENTS,
        format_func=lambda key: TARGET_SPECS[key].display_name,
        key="explain_component",
    )
    try:
        explanation = explain_prediction(component, features, top_n=12)
        st.dataframe(explanation, use_container_width=True, hide_index=True)
    except (ModelNotTrainedError, ValueError) as error:
        st.warning(f"Explanation unavailable: {error}")

    bundle = load_component_bundle(component)
    metadata: Dict[str, object] = bundle["metadata"]  # type: ignore[assignment]
    st.caption(
        f"Model: {metadata['selected_model']} on the '{metadata['feature_set']}' "
        f"feature set, {metadata['n_selected_features']} features reaching the "
        f"classifier. Selection rule: {metadata['selection_rule']}."
    )

    st.subheader("Raw sensor traces for this cycle")
    chosen = st.multiselect(
        "Sensors to plot", list(SENSOR_NAMES), default=["PS1", "EPS1", "FS1", "TS1"],
    )
    for sensor in chosen:
        values = cycle.get(sensor)
        if values is None:
            continue
        spec = SENSOR_INDEX[sensor]
        frame = pd.DataFrame(
            {f"{sensor} [{spec.unit}]": values},
            index=np.linspace(0, 60, len(values)),
        )
        frame.index.name = "time within cycle [s]"
        st.line_chart(frame, height=180)


# ==========================================================================
# Results section (model performance)
# ==========================================================================
def section_results() -> None:
    st.header("Model performance")
    metrics = cached_metrics()
    if metrics["cv"] is None:
        st.warning("Run `python src/train.py` to populate this section.")
        return

    st.subheader("Effect of feature engineering (cross-validated macro F1)")
    best = metrics["cv"].groupby(["component", "feature_set"])["f1_macro"].max().unstack()
    ordered = [c for c in ("baseline", "engineered", "selected") if c in best.columns]
    st.dataframe(best[ordered].round(4), use_container_width=True)
    st.bar_chart(best[ordered])

    st.subheader("Full cross-validation grid")
    st.dataframe(
        metrics["cv"].round(4), use_container_width=True, hide_index=True
    )

    if metrics["test"] is not None:
        st.subheader("Held-out test set (each model evaluated exactly once)")
        st.dataframe(metrics["test"].round(4), use_container_width=True, hide_index=True)
        st.caption(
            "`block_cv_f1_macro` is a leakage check: whole contiguous blocks of "
            "cycles are held out, so a cycle is never scored while its immediate "
            "neighbours sit in the training set. The gap to `cv_f1_macro` is an "
            "honest estimate of how optimistic the random split is."
        )
    else:
        st.info("Run `python src/evaluate.py` to add test-set results.")


# ==========================================================================
# Layout
# ==========================================================================
def main() -> None:
    st.title("🛠️ Predictive Maintenance of Industrial Hydraulic Systems")
    st.caption(
        "Condition monitoring of a hydraulic test rig through feature engineering - "
        "UCI *Condition monitoring of hydraulic systems* dataset (ZeMA gGmbH)."
    )

    with st.sidebar:
        st.header("Status")
        st.write("Dataset:", "found" if cached_dataset_summary() else "missing")
        st.write("Trained models:", "found" if models_ready() else "missing")
        metadata = cached_run_metadata()
        if metadata:
            st.write(
                "Gradient boosting backend:",
                "XGBoost" if metadata.get("xgboost_available") else "HistGradientBoosting",
            )
            st.write("Random state:", metadata.get("random_state"))
        st.divider()
        st.caption(
            "The dashboard loads saved pipelines only. It never retrains, so the "
            "preprocessing and feature selection are exactly those used at "
            "training time."
        )

    tabs = st.tabs(
        ["Overview", "Dataset", "Feature engineering", "Predict & status", "Results"]
    )
    with tabs[0]:
        section_overview()
    with tabs[1]:
        section_dataset()
    with tabs[2]:
        section_features()
    with tabs[3]:
        section_prediction()
    with tabs[4]:
        section_results()


if __name__ == "__main__":
    main()
