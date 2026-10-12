"""
MLapp.py
========
Model evaluation studio for the hydraulic predictive-maintenance project.

WHERE app.py TELLS THE FEATURE-ENGINEERING STORY, THIS APP TELLS THE MODEL STORY
    1. Overview          - the four classification tasks and the final models
    2. Predict           - score any cycle or CSV, with evidence
    3. Model arena       - all 48 cross-validated configurations, compared
    4. Pipeline          - what is actually inside each saved .joblib pipeline
    5. Evaluation lab    - live test-set metrics, ROC / PR, calibration, errors
    6. Generalization    - random vs blocked validation, and why they differ
    7. Error analysis    - why the misclassified cycles were misclassified
    8. Explainability    - global importance and a local what-if probe
    9. Robustness        - noise stress test and selective prediction

INFERENCE ONLY
    Nothing is trained here. Every number on the evaluation tabs is computed
    live by the saved pipelines on the saved held-out test indices, using the
    engineered feature table written by the training run. Those numbers match
    outputs/metrics/test_results.csv exactly.

RUN
    streamlit run MLapp.py
"""

from __future__ import annotations

import html as html_lib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import TARGET_SPECS  # noqa: E402
from utils import COMPONENTS, FEATURES_DIR, METRICS_DIR, MODELS_DIR, load_json  # noqa: E402

st.set_page_config(
    page_title="Hydraulic ML Studio",
    page_icon="🧠",
    layout="wide",
)

# ==========================================================================
# Constants
# ==========================================================================
COMPONENT_ICONS = {"cooler": "❄️", "valve": "🔧", "pump": "⚙️", "accumulator": "🛢️"}
MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting (XGBoost)",
    "svm": "SVM (RBF)",
}
MODEL_SHORT = {"logistic_regression": "LR", "random_forest": "RF", "gradient_boosting": "XGB", "svm": "SVM"}
FEATURE_SET_NAMES = {"baseline": "Baseline (17)", "engineered": "Engineered (553)", "selected": "Selected (40)"}
FEATURE_SET_ORDER = ["baseline", "engineered", "selected"]

# Categorical palette in fixed order (validated for a dark surface).
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
MODEL_COLOURS = {
    "logistic_regression": SERIES[0],
    "random_forest": SERIES[1],
    "gradient_boosting": SERIES[2],
    "svm": SERIES[3],
}
SEVERITY_COLOURS = {"Normal": "#22C55E", "Degraded": "#F59E0B", "Critical": "#EF4444", "Unknown": "#94A3B8"}
BLOCK_COLOURS = {
    "statistical": "#38BDF8",
    "time_domain": "#818CF8",
    "rolling": "#22C55E",
    "frequency": "#F59E0B",
    "domain": "#F472B6",
}
INK = "#E6EDF7"
MUTED = "#94A3B8"
GRID = "rgba(148,163,184,0.16)"
SEQ = ["#16233a", "#1f4f8f", "#3987e5", "#8cc3ff"]  # dark -> bright, for a dark surface


def _streamlit_version() -> Tuple[int, int]:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return major, minor
    except ValueError:
        return 0, 0


FULL_WIDTH: Dict[str, object] = (
    {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}
)

# ==========================================================================
# Styling
# ==========================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
:root {
  --accent: #38BDF8; --accent-2: #818CF8;
  --card: rgba(148,163,184,0.08); --card-hover: rgba(148,163,184,0.13);
  --line: rgba(148,163,184,0.20); --muted: rgba(148,163,184,0.95); --track: rgba(148,163,184,0.16);
}
.stApp { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1400px; }

@keyframes fadeUp  { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes growX   { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes shimmer { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes pulseNode { 0%,100% { box-shadow: 0 0 0 0 rgba(56,189,248,.0); } 50% { box-shadow: 0 0 0 6px rgba(56,189,248,.12); } }
@keyframes dash { to { background-position: 24px 0; } }

.hero {
  position: relative; overflow: hidden; padding: 1.7rem 2rem 1.5rem; border-radius: 20px; margin-bottom: 1.2rem;
  border: 1px solid var(--line);
  background: linear-gradient(120deg, rgba(129,140,248,.18), rgba(56,189,248,.12), rgba(244,114,182,.10), rgba(129,140,248,.18));
  background-size: 300% 300%; animation: shimmer 16s ease infinite, fadeUp .6s ease both;
}
.hero-eyebrow { font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent-2); font-weight: 700; }
.hero-title { font-size: 2.0rem; font-weight: 800; line-height: 1.15; margin: .35rem 0 .45rem; }
.hero-title span { background: linear-gradient(90deg, var(--accent-2), var(--accent)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-sub { font-size: .96rem; opacity: .85; max-width: 62rem; }

.sec { display: flex; align-items: center; gap: .9rem; margin: 1.2rem 0 .9rem; animation: fadeUp .5s ease both; }
.sec-num { flex: none; width: 2.4rem; height: 2.4rem; border-radius: 12px; display: grid; place-items: center;
  font-weight: 800; font-size: .95rem; color: #0B1220; background: linear-gradient(135deg, var(--accent-2), var(--accent)); }
.sec-title { font-size: 1.45rem; font-weight: 750; line-height: 1.2; }
.sec-sub { font-size: .88rem; color: var(--muted); }

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: .8rem; margin: .4rem 0 1rem; }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: .95rem 1.1rem;
  animation: fadeUp .55s ease both; transition: transform .2s ease, background .2s ease, border-color .2s ease; }
.stat:hover { transform: translateY(-3px); background: var(--card-hover); border-color: rgba(129,140,248,.45); }
.stat-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
.stat-value { font-size: 1.65rem; font-weight: 800; margin-top: .2rem; font-variant-numeric: tabular-nums; }
.stat-hint { font-size: .8rem; color: var(--muted); margin-top: .1rem; }
.stat-value.good { color: #22C55E; } .stat-value.warn { color: #F59E0B; } .stat-value.bad { color: #EF4444; }

.panel { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.15rem; animation: fadeUp .55s ease both; margin-bottom: .8rem; }
.panel h4 { margin: 0 0 .5rem; font-size: .98rem; font-weight: 700; }
.panel-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: .9rem; margin-bottom: .9rem; }

.flow { display: flex; align-items: stretch; gap: 0; flex-wrap: nowrap; margin: .4rem 0 1.1rem; overflow-x: auto; }
.node { flex: 1 1 0; min-width: 96px; background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: .75rem .85rem;
  animation: fadeUp .5s ease both, pulseNode 3.2s ease-in-out infinite; text-align: center; }
.node b { display: block; font-size: .92rem; } .node span { font-size: .75rem; color: var(--muted); }
.wire { flex: 0 0 18px; align-self: center; height: 3px; border-radius: 2px;
  background: linear-gradient(90deg, var(--accent) 50%, transparent 50%); background-size: 12px 3px; animation: dash .8s linear infinite; }

.model-card { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.15rem;
  border-top: 3px solid var(--accent-2); animation: fadeUp .55s ease both; transition: transform .2s ease; }
.model-card:hover { transform: translateY(-4px); }
.mc-top { font-size: .82rem; color: var(--muted); font-weight: 600; }
.mc-model { font-size: 1.15rem; font-weight: 800; margin: .3rem 0 .15rem; }
.mc-meta { font-size: .8rem; color: var(--muted); }
.mc-row { display: flex; justify-content: space-between; font-size: .85rem; margin-top: .45rem; font-variant-numeric: tabular-nums; }

.bar-row { display: grid; grid-template-columns: minmax(120px, 1.2fr) 3fr minmax(58px, auto); gap: .7rem; align-items: center; margin: .36rem 0; font-size: .86rem; }
.bar-row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-row .num { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.track { height: 9px; border-radius: 99px; background: var(--track); overflow: hidden; }
.fill { height: 100%; border-radius: 99px; transform-origin: left center; animation: growX .9s cubic-bezier(.2,.8,.2,1) both; }

.kv { width: 100%; border-collapse: collapse; font-size: .86rem; }
.kv td { padding: .38rem .2rem; border-bottom: 1px solid var(--line); }
.kv td:first-child { color: var(--muted); width: 48%; }
.kv td:last-child { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .8rem; text-align: right; }

.chipcloud { display: flex; flex-wrap: wrap; gap: .35rem; }
.fchip { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .74rem; padding: .18rem .5rem; border-radius: 8px;
  background: var(--card); border: 1px solid var(--line); border-left: 3px solid var(--c); animation: fadeUp .4s ease both; }
.legend { display: flex; gap: 1rem; flex-wrap: wrap; font-size: .78rem; color: var(--muted); margin: .2rem 0 .6rem; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: .35rem; vertical-align: -1px; }
.chip { display: inline-block; padding: .12rem .6rem; border-radius: 99px; font-size: .76rem; font-weight: 700; }
.chip.ok { background: rgba(34,197,94,.18); color: #22C55E; } .chip.bad { background: rgba(239,68,68,.18); color: #EF4444; }
.chip.neutral { background: var(--track); color: inherit; font-weight: 600; }
.note { border-left: 3px solid var(--accent-2); background: rgba(129,140,248,.08); padding: .7rem .9rem; border-radius: 0 12px 12px 0; font-size: .88rem; margin: .4rem 0 1rem; }

.side-row { display: flex; justify-content: space-between; align-items: center; padding: .42rem 0; border-bottom: 1px solid var(--line); font-size: .86rem; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .4rem; }

.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: .5rem .9rem; transition: background .2s ease; font-weight: 600; }
.stTabs [data-baseweb="tab"]:hover { background: var(--card); }
.stTabs [data-baseweb="tab-panel"] { animation: fadeUp .45s ease both; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }
[data-testid="stExpander"] details { border-radius: 14px; border: 1px solid var(--line); }
[data-testid="stVegaLiteChart"] { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: .6rem .4rem .2rem; animation: fadeUp .5s ease both; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""


def render_html(markup: str) -> None:
    """Render raw HTML; lines are joined so Markdown never treats them as code."""
    st.markdown(" ".join(l.strip() for l in markup.splitlines() if l.strip()), unsafe_allow_html=True)


def esc(value: object) -> str:
    return html_lib.escape(str(value))


def section_header(number: str, title: str, subtitle: str = "") -> None:
    render_html(
        f'<div class="sec"><div class="sec-num">{esc(number)}</div><div>'
        f'<div class="sec-title">{esc(title)}</div><div class="sec-sub">{esc(subtitle)}</div></div></div>'
    )


def stat_cards(items: Iterable[Tuple[str, str, str, str]], min_width: int = 180) -> None:
    """(label, value, hint, tone) where tone is '', 'good', 'warn' or 'bad'."""
    cells = "".join(
        f'<div class="stat" style="animation-delay:{i * 0.06:.2f}s"><div class="stat-label">{esc(l)}</div>'
        f'<div class="stat-value {tone}">{esc(v)}</div><div class="stat-hint">{esc(h)}</div></div>'
        for i, (l, v, h, tone) in enumerate(items)
    )
    render_html(f'<div class="stat-grid" style="grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr))">{cells}</div>')


def note(text: str) -> None:
    render_html(f'<div class="note">{text}</div>')


def tone_for(f1: float) -> str:
    return "good" if f1 >= 0.98 else ("warn" if f1 >= 0.93 else "bad")


def style_chart(chart: alt.TopLevelMixin, height: int = 300) -> alt.TopLevelMixin:
    """Apply one consistent, recessive chart style."""
    return (
        chart.properties(height=height)
        .configure(background="transparent", font="Inter")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID, tickColor=GRID,
            labelFontSize=11, titleFontSize=11, titleFontWeight=500, labelOverlap=True, tickCount=8,
        )
        .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=11, titleFontSize=11, orient="top")
        .configure_header(labelColor=INK, titleColor=MUTED, labelFontSize=12, labelFontWeight=600)
        .configure_title(color=INK, fontSize=13, anchor="start")
    )


def show_chart(chart: alt.TopLevelMixin, height: int = 300) -> None:
    st.altair_chart(style_chart(chart, height), theme=None, **FULL_WIDTH)


# ==========================================================================
# Data loading (all cached; nothing is trained)
# ==========================================================================
@st.cache_data(show_spinner="Loading the engineered feature table...")
def load_features() -> Optional[pd.DataFrame]:
    path = FEATURES_DIR / "engineered_features.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_metrics() -> Dict[str, Optional[pd.DataFrame]]:
    def read(path: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(path) if path.exists() else None

    return {
        "cv": read(METRICS_DIR / "model_results.csv"),
        "test": read(METRICS_DIR / "test_results.csv"),
        "importance": read(METRICS_DIR / "feature_importance.csv"),
    }


@st.cache_data(show_spinner=False)
def load_run_metadata() -> Optional[Dict[str, object]]:
    path = MODELS_DIR / "run_metadata.json"
    return load_json(path) if path.exists() else None


@st.cache_resource(show_spinner="Loading the four saved pipelines...")
def load_bundles() -> Dict[str, Dict[str, object]]:
    """Every component's fitted pipeline, label encoder, metadata and split."""
    bundles: Dict[str, Dict[str, object]] = {}
    for c in COMPONENTS:
        d = MODELS_DIR / c
        with np.load(d / "split_indices.npz") as split:
            train_idx, test_idx = split["train_idx"].copy(), split["test_idx"].copy()
        bundles[c] = {
            "model": joblib.load(d / "model.joblib"),
            "encoder": joblib.load(d / "label_encoder.joblib"),
            "meta": load_json(d / "metadata.json"),
            "train_idx": train_idx,
            "test_idx": test_idx,
        }
    return bundles


def artefacts_ready() -> bool:
    return all((MODELS_DIR / c / "model.joblib").exists() for c in COMPONENTS) and (
        FEATURES_DIR / "engineered_features.csv"
    ).exists()


def xy(component: str, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Model-ready X and encoded y for a set of cycle indices."""
    b = load_bundles()[component]
    table = load_features()
    cols = list(b["meta"]["input_feature_columns"])  # type: ignore[index]
    X = table.loc[idx, cols].to_numpy(dtype=np.float64)
    y = b["encoder"].transform(table.loc[idx, TARGET_SPECS[component].column].to_numpy())  # type: ignore[union-attr]
    return X, y


@st.cache_data(show_spinner="Scoring the held-out test set...")
def evaluate(component: str) -> Dict[str, object]:
    """Live test-set evaluation of the saved pipeline."""
    b = load_bundles()[component]
    test_idx: np.ndarray = b["test_idx"]  # type: ignore[assignment]
    X, y = xy(component, test_idx)
    model = b["model"]
    proba = model.predict_proba(X)  # type: ignore[union-attr]
    pred = proba.argmax(axis=1)
    classes = [int(v) for v in b["encoder"].classes_]  # type: ignore[union-attr]
    return {"y": y, "pred": pred, "proba": proba, "classes": classes, "idx": test_idx}


def class_label(component: str, value: int) -> str:
    spec = TARGET_SPECS[component]
    return f"{value} {spec.unit} · {spec.value_labels.get(int(value), '')}"


# ==========================================================================
# Hero + sidebar
# ==========================================================================
def render_hero() -> None:
    render_html(
        """
        <div class="hero">
          <div class="hero-eyebrow">Machine learning studio · model evaluation console</div>
          <div class="hero-title">Hydraulic Condition Models: <span>How They Learn, Score and Fail</span></div>
          <div class="hero-sub">Four supervised multiclass classifiers, one per component, inspected end to end:
          the 48-configuration model search, the fitted pipelines, live held-out evaluation, generalisation
          under blocked validation, error analysis, explainability and robustness.</div>
        </div>
        """
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🧠 ML Studio")
        component = st.radio(
            "Focus component",
            list(COMPONENTS),
            format_func=lambda c: f"{COMPONENT_ICONS[c]}  {TARGET_SPECS[c].display_name}",
            key="focus_component",
        )
        st.caption("Drives the Predict evidence panel and the Pipeline, Evaluation lab, Error analysis, Explainability and Robustness tabs.")
        st.divider()
        ok = artefacts_ready()
        meta = load_run_metadata()
        rows = [
            ("Saved pipelines", "4 / 4" if ok else "missing", "#22C55E" if ok else "#EF4444"),
            ("Feature table", "found" if (FEATURES_DIR / "engineered_features.csv").exists() else "missing", "#22C55E"),
        ]
        if meta:
            rows += [
                ("Boosting backend", "XGBoost" if meta.get("xgboost_available") else "HistGB", "#94A3B8"),
                ("CV folds", str(meta.get("cv_folds")), "#94A3B8"),
                ("Test size", f"{float(meta.get('test_size', 0)) * 100:.0f} %", "#94A3B8"),
                ("Random state", str(meta.get("random_state")), "#94A3B8"),
            ]
        render_html(
            "".join(
                f'<div class="side-row"><span>{esc(k)}</span><span><span class="dot" style="background:{c}"></span>{esc(v)}</span></div>'
                for k, v, c in rows
            )
        )
        st.divider()
        st.caption(
            "Inference only. Every evaluation number is computed live by the saved pipelines "
            "on the saved held-out test indices - nothing is retrained in this app."
        )
    return str(component)


# ==========================================================================
# 1. Overview
# ==========================================================================
def tab_overview() -> None:
    section_header("01", "Overview", "Four independent multiclass problems, one pipeline each")
    metrics = load_metrics()
    test = metrics["test"]
    cv = metrics["cv"]
    n_cfg = 0 if cv is None else len(cv)
    stat_cards(
        [
            ("Learning tasks", "4", "multiclass, one per component", ""),
            ("Configurations searched", str(n_cfg), "4 models × 3 feature sets × 4 targets", ""),
            ("Train / test cycles", "1,764 / 441", "stratified 80 / 20 split per target", ""),
            ("Model families", "4", "LR · RF · XGBoost · SVM", ""),
        ]
    )

    st.subheader("Inference pipeline")
    steps = [
        ("Raw cycle", "17 sensors · 43,680 values"),
        ("Feature extraction", "626 engineered features"),
        ("Cleaning", "553 features"),
        ("Impute + scale", "median · z-score"),
        ("Selection", "corr → MI → trees"),
        ("Classifier", "LR / RF / XGB / SVM"),
        ("Condition", "class + probability"),
    ]
    parts: List[str] = []
    for i, (a, b) in enumerate(steps):
        if i:
            parts.append('<div class="wire"></div>')
        parts.append(f'<div class="node" style="animation-delay:{i * 0.08:.2f}s,{i * 0.4:.1f}s"><b>{esc(a)}</b><span>{esc(b)}</span></div>')
    render_html(f'<div class="flow">{"".join(parts)}</div>')
    note(
        "Every learned step - imputation, scaling, correlation filter, mutual information, tree importance "
        "and the classifier - lives inside one scikit-learn <b>Pipeline</b>, so cross-validation re-fits all of "
        "them on each training fold and no validation or test cycle ever influences training."
    )

    st.subheader("Final model per component")
    if test is not None:
        cards: List[str] = []
        for i, c in enumerate(COMPONENTS):
            row = test[test.component == c].iloc[0]
            gap = float(row.cv_f1_macro) - float(row.block_cv_f1_macro)
            cards.append(
                f'<div class="model-card" style="animation-delay:{i * 0.08:.2f}s">'
                f'<div class="mc-top">{COMPONENT_ICONS[c]} {esc(TARGET_SPECS[c].display_name)}</div>'
                f'<div class="mc-model">{esc(MODEL_NAMES.get(row.model, row.model))}</div>'
                f'<div class="mc-meta">{esc(FEATURE_SET_NAMES.get(row.feature_set, row.feature_set))} · '
                f'{int(row.n_selected_features)} inputs reach the classifier</div>'
                f'<div class="mc-row"><span>Test macro-F1</span><b>{float(row.f1_macro):.3f}</b></div>'
                f'<div class="mc-row"><span>Blocked CV macro-F1</span><b>{float(row.block_cv_f1_macro):.3f}</b></div>'
                f'<div class="mc-row"><span>Test ROC-AUC (OvR)</span><b>{float(row.roc_auc_ovr_macro):.4f}</b></div>'
                f'<div class="mc-row"><span>Random-split optimism</span><b>{gap:+.3f}</b></div></div>'
            )
        render_html(f'<div class="stat-grid" style="grid-template-columns:repeat(auto-fit,minmax(250px,1fr))">{"".join(cards)}</div>')

    st.subheader("Targets and class balance")
    table = load_features()
    rows = []
    for c in COMPONENTS:
        spec = TARGET_SPECS[c]
        counts = table[spec.column].value_counts().sort_index()
        rows.append(
            {
                "Component": f"{COMPONENT_ICONS[c]} {spec.display_name}",
                "Classes": len(counts),
                "Class counts": " / ".join(f"{int(k)}: {int(v)}" for k, v in counts.items()),
                "Imbalance (max/min)": round(float(counts.max() / counts.min()), 2),
                "Headline metric": "macro-F1",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, **FULL_WIDTH)


# ==========================================================================
# 2. Predict
# ==========================================================================
def _pick_random_cycle(n_cycles: int) -> None:
    """Button callback: choose and score a random dataset cycle."""
    choice = int(np.random.default_rng().integers(n_cycles))
    st.session_state["ml_cycle_input"] = choice
    st.session_state["ml_active_cycle"] = choice


def _predict_selected_cycle() -> None:
    st.session_state["ml_active_cycle"] = int(st.session_state["ml_cycle_input"])


@st.cache_data(show_spinner=False)
def reliability_bins(component: str) -> pd.DataFrame:
    """Test-set accuracy per confidence bin, used to qualify a new prediction."""
    ev = evaluate(component)
    conf = ev["proba"].max(axis=1)  # type: ignore[union-attr]
    correct = ev["pred"] == ev["y"]  # type: ignore[operator]
    edges = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "acc": float(correct[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Indexing the training cycles in model space...")
def training_embedding(component: str) -> np.ndarray:
    """Training cycles transformed by every pipeline step before the classifier."""
    b = load_bundles()[component]
    X, _ = xy(component, b["train_idx"])  # type: ignore[arg-type]
    return b["model"][:-1].transform(X)  # type: ignore[index]


def score_row(component: str, row: pd.DataFrame) -> Dict[str, object]:
    """Score one engineered feature row with a component's saved pipeline."""
    b = load_bundles()[component]
    cols: List[str] = list(b["meta"]["input_feature_columns"])  # type: ignore[index]
    missing = [c for c in cols if c not in row.columns]
    if missing:
        raise ValueError(f"{len(missing)} model input(s) missing, e.g. {', '.join(missing[:3])}")
    x = row[cols].to_numpy(dtype=np.float64)
    p = b["model"].predict_proba(x)[0]  # type: ignore[union-attr]
    classes = [int(v) for v in b["encoder"].classes_]  # type: ignore[union-attr]
    k = int(np.argmax(p))
    srt = np.sort(p)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p)))
    return {"x": x, "proba": p, "classes": classes, "pred": classes[k], "conf": float(p[k]),
            "margin": float(srt[-1] - srt[-2]), "entropy": entropy}


def tab_predict(component: str) -> None:
    section_header("02", "Predict", "Score one operating cycle with all four saved pipelines")
    table = load_features()
    source = st.radio("Input source", ["Dataset cycle", "Upload a raw-sensor CSV"], horizontal=True, key="ml_source")

    row: Optional[pd.DataFrame] = None
    cycle_id: Optional[int] = None
    if source == "Dataset cycle":
        n = len(table)
        if "ml_cycle_input" not in st.session_state:
            st.session_state["ml_cycle_input"] = 0
        kw = {"vertical_alignment": "bottom"} if _streamlit_version() >= (1, 36) else {}
        c1, c2, c3 = st.columns([2, 1, 1], **kw)
        with c1:
            st.number_input(f"Cycle number (0 - {n - 1})", min_value=0, max_value=n - 1, step=1, key="ml_cycle_input")
        with c2:
            st.button("⚡ Predict this cycle", type="primary", on_click=_predict_selected_cycle, **FULL_WIDTH)
        with c3:
            st.button("🎲 Random cycle", on_click=_pick_random_cycle, args=(n,), **FULL_WIDTH)
        active = st.session_state.get("ml_active_cycle")
        if active is not None:
            cycle_id = int(active)
            row = table.loc[[cycle_id]]
            if cycle_id != int(st.session_state["ml_cycle_input"]):
                st.caption(f"Showing cycle **{cycle_id}**. Press *Predict this cycle* to score cycle {st.session_state['ml_cycle_input']}.")
    else:
        st.caption("One column per sensor (PS1 ... SE) holding that sensor's samples for one 60-second cycle: 6000 rows for 100 Hz, 600 for 10 Hz, 60 for 1 Hz; shorter columns padded with empty cells.")
        uploaded = st.file_uploader("Raw-sensor CSV for one cycle", type=["csv"], key="ml_upload")
        if uploaded is not None:
            try:
                from prediction import features_from_cycle, load_cycle_csv  # local import keeps start-up light
                with st.spinner("Extracting the 626 engineered features..."):
                    row = features_from_cycle(load_cycle_csv(uploaded))
            except ValueError as error:
                st.error(str(error))
                return

    if row is None:
        render_html(
            '<div class="panel" style="text-align:center;padding:2rem 1rem"><div style="font-size:2rem">🔮</div>'
            '<div style="font-weight:700;margin:.3rem 0">No cycle scored yet</div>'
            '<div class="sec-sub">Pick a cycle and press <b>Predict</b>, try a <b>random cycle</b>, or upload a CSV.</div></div>'
        )
        return

    try:
        results = {c: score_row(c, row) for c in COMPONENTS}
    except ValueError as error:
        st.error(f"Prediction failed: {error}")
        return

    # ---- overall status ---------------------------------------------------
    order = {"Normal": 0, "Degraded": 1, "Critical": 2, "Unknown": 1}
    sev = {c: TARGET_SPECS[c].severity.get(int(r["pred"]), "Unknown") for c, r in results.items()}
    worst = max(COMPONENTS, key=lambda c: order[sev[c]])
    colour = SEVERITY_COLOURS[sev[worst]]
    mean_conf = np.mean([r["conf"] for r in results.values()])
    label = f"cycle {cycle_id}" if cycle_id is not None else "uploaded cycle"
    render_html(
        f'<div class="panel" style="display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;border-color:{colour}66;'
        f'background:linear-gradient(100deg,{colour}22,transparent 70%)">'
        f'<span class="dot" style="background:{colour};width:16px;height:16px;box-shadow:0 0 0 6px {colour}33"></span>'
        f'<div><div class="stat-label">Overall system status · {esc(label)}</div>'
        f'<div style="font-size:1.8rem;font-weight:800;color:{colour};line-height:1.1">{esc(sev[worst])}</div>'
        f'<div class="sec-sub">worst component: {esc(TARGET_SPECS[worst].display_name)}</div></div>'
        f'<div style="margin-left:auto;text-align:right"><div class="stat-label">Mean confidence</div>'
        f'<div style="font-size:1.4rem;font-weight:800">{mean_conf * 100:.1f} %</div></div></div>'
    )

    # ---- per-component prediction cards ----------------------------------
    test_sets = {c: set(int(i) for i in load_bundles()[c]["test_idx"]) for c in COMPONENTS}  # type: ignore[union-attr]
    cards: List[str] = []
    for i, c in enumerate(COMPONENTS):
        r = results[c]
        spec = TARGET_SPECS[c]
        rb = reliability_bins(c)
        hit = rb[(rb.lo <= r["conf"]) & (r["conf"] <= rb.hi)]
        rel = hit.iloc[-1] if len(hit) else None
        rel_txt = (f"test cycles at this confidence: {rel.acc * 100:.0f} % correct (n={int(rel.n)})"
                   if rel is not None and rel.n > 0 else "no test cycles at this confidence")
        truth = ""
        if cycle_id is not None:
            actual = int(table.loc[cycle_id, spec.column])
            ok = actual == int(r["pred"])
            where = "test" if cycle_id in test_sets[c] else "train"
            truth = (f'<div style="margin-top:.5rem"><span class="chip {"ok" if ok else "bad"}">{"✓ matches" if ok else "✗ recorded"} '
                     f'{actual} {esc(spec.unit)}</span> <span class="chip neutral">{where} cycle</span></div>')
        sc = SEVERITY_COLOURS[sev[c]]
        cards.append(
            f'<div class="model-card" style="border-top-color:{sc};animation-delay:{i * 0.08:.2f}s">'
            f'<div class="mc-top">{COMPONENT_ICONS[c]} {esc(spec.display_name)}</div>'
            f'<div class="mc-model" style="color:{sc}">{esc(sev[c])} · {int(r["pred"])} {esc(spec.unit)}</div>'
            f'<div class="mc-meta">{esc(spec.value_labels.get(int(r["pred"]), ""))}</div>'
            f'<div class="mc-row"><span>Confidence</span><b>{r["conf"]:.3f}</b></div>'
            f'<div class="mc-row"><span>Margin (top-1 − top-2)</span><b>{r["margin"]:.3f}</b></div>'
            f'<div class="mc-row"><span>Normalised entropy</span><b>{r["entropy"]:.3f}</b></div>'
            f'<div class="mc-meta" style="margin-top:.45rem">{esc(rel_txt)}</div>{truth}</div>'
        )
    render_html(f'<div class="stat-grid" style="grid-template-columns:repeat(auto-fit,minmax(250px,1fr))">{"".join(cards)}</div>')
    st.caption("Margin and entropy measure how decisive each prediction is (entropy 0 = certain, 1 = uniform guess). The reliability line looks up how often the model was right on held-out test cycles with a similar confidence.")

    # ---- probability distributions -----------------------------------------
    st.subheader("Class probability distributions")
    prob_rows = []
    for c in COMPONENTS:
        r = results[c]
        for cls, p in zip(r["classes"], r["proba"]):  # type: ignore[arg-type]
            prob_rows.append({"Component": TARGET_SPECS[c].display_name,
                              "Class": f"{cls} {TARGET_SPECS[c].unit} · {TARGET_SPECS[c].value_labels.get(cls, '')}",
                              "Severity": TARGET_SPECS[c].severity.get(cls, "Unknown"), "Probability": float(p),
                              "Predicted": cls == r["pred"]})
    pdf = pd.DataFrame(prob_rows)
    bars = alt.Chart(pdf).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Class:N", title=None, sort=None, axis=alt.Axis(labelLimit=230)),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="probability"),
        color=alt.Color("Severity:N", scale=alt.Scale(domain=list(SEVERITY_COLOURS)[:3], range=[SEVERITY_COLOURS[k] for k in list(SEVERITY_COLOURS)[:3]])),
        opacity=alt.condition(alt.datum.Predicted, alt.value(1.0), alt.value(0.45)),
        tooltip=["Component", "Class", alt.Tooltip("Probability:Q", format=".4f")],
    ).properties(width=330, height=130).facet(
        facet=alt.Facet("Component:N", sort=[TARGET_SPECS[c].display_name for c in COMPONENTS], title=None), columns=2
    ).resolve_scale(y="independent")
    st.altair_chart(
        bars.configure(background="transparent", font="Inter").configure_view(strokeWidth=0)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID, tickColor=GRID)
        .configure_legend(labelColor=INK, titleColor=MUTED, orient="top")
        .configure_header(labelColor=INK, labelFontSize=12, labelFontWeight=600),
        theme=None, **FULL_WIDTH,
    )

    # ---- nearest training cycles in model space ----------------------------
    spec = TARGET_SPECS[component]
    st.subheader(f"Evidence: nearest training cycles - {spec.display_name}")
    b = load_bundles()[component]
    emb = training_embedding(component)
    q = b["model"][:-1].transform(results[component]["x"])  # type: ignore[index]
    d = np.sqrt(((emb - q) ** 2).sum(axis=1))
    train_idx: np.ndarray = b["train_idx"]  # type: ignore[assignment]
    if cycle_id is not None:
        d = np.where(train_idx == cycle_id, np.inf, d)  # never show the cycle as its own neighbour
    nn = np.argsort(d)[:5]
    nn_df = pd.DataFrame(
        {"Training cycle": train_idx[nn], "Distance": d[nn].round(3),
         "Recorded condition": [class_label(component, int(table.loc[i, spec.column])) for i in train_idx[nn]]}
    )
    agree = int((table.loc[train_idx[nn], spec.column].to_numpy() == int(results[component]["pred"])).sum())
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.dataframe(nn_df, hide_index=True, **FULL_WIDTH)
    with c2:
        stat_cards([("Neighbours agreeing", f"{agree} / 5", "share the predicted condition", "good" if agree >= 4 else ("warn" if agree >= 2 else "bad"))], min_width=150)
    st.caption("Distances are measured after the pipeline's own imputation, scaling and feature selection - the space the classifier actually sees. Change the focus component in the sidebar to inspect another model. Consecutive cycles are near-duplicates, so a dataset cycle's neighbours are usually its recording neighbours.")


# ==========================================================================
# 2. Model arena
# ==========================================================================
def tab_arena() -> None:
    section_header("03", "Model arena", "All cross-validated configurations on the training set")
    cv = load_metrics()["cv"]
    if cv is None:
        st.warning("outputs/metrics/model_results.csv not found - run `python src/train.py`.")
        return
    df = cv.copy()
    df["Model"] = df.model.map(MODEL_SHORT)
    df["Feature set"] = df.feature_set.map(FEATURE_SET_NAMES)
    df["Component"] = df.component.map(lambda c: TARGET_SPECS[c].display_name)
    df["F1"] = df.f1_macro.round(4)

    c1, c2 = st.columns([2, 1])
    with c1:
        chosen = st.multiselect(
            "Components", list(COMPONENTS), default=list(COMPONENTS),
            format_func=lambda c: TARGET_SPECS[c].display_name, key="arena_components",
        )
    with c2:
        floor = st.slider("Colour scale starts at", 0.50, 0.95, 0.50, 0.05, key="arena_floor")
    df = df[df.component.isin(chosen)] if chosen else df

    st.subheader("Cross-validated macro-F1 heatmap")
    base = alt.Chart(df).encode(
        x=alt.X("Model:N", sort=["LR", "RF", "XGB", "SVM"], title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Feature set:N", sort=[FEATURE_SET_NAMES[s] for s in FEATURE_SET_ORDER], title=None),
    )
    rect = base.mark_rect(cornerRadius=4, stroke="#0B1220", strokeWidth=2).encode(
        color=alt.Color("f1_macro:Q", scale=alt.Scale(range=SEQ, domain=[floor, 1.0], clamp=True), title="macro-F1"),
        tooltip=["Component", "Model", "Feature set", alt.Tooltip("f1_macro:Q", format=".4f", title="macro-F1"),
                 alt.Tooltip("f1_macro_std:Q", format=".4f", title="fold std"), alt.Tooltip("training_time:Q", format=".2f", title="fit time [s]")],
    )
    text = base.mark_text(fontSize=11, fontWeight=600).encode(
        text=alt.Text("f1_macro:Q", format=".3f"),
        color=alt.condition(alt.datum.f1_macro > floor + 0.85 * (1 - floor), alt.value("#0B1220"), alt.value(INK)),
    )
    heat = alt.layer(rect, text, data=df).properties(width=165, height=150).facet(
        column=alt.Column("Component:N", title=None, sort=[TARGET_SPECS[c].display_name for c in COMPONENTS])
    )
    st.altair_chart(
        heat.configure(background="transparent", font="Inter").configure_view(strokeWidth=0)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor=GRID, tickColor=GRID)
        .configure_legend(labelColor=INK, titleColor=MUTED, orient="right")
        .configure_header(labelColor=INK, labelFontSize=12, labelFontWeight=600),
        theme=None, **FULL_WIDTH,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Fold-to-fold stability")
        comp = st.selectbox("Component", list(COMPONENTS), format_func=lambda c: TARGET_SPECS[c].display_name, key="arena_std_comp")
        d = cv[cv.component == comp].copy()
        d["Model"] = d.model.map(MODEL_SHORT)
        d["Feature set"] = d.feature_set.map(FEATURE_SET_NAMES)
        d["lo"] = d.f1_macro - d.f1_macro_std
        d["hi"] = (d.f1_macro + d.f1_macro_std).clip(upper=1.0)
        xenc = alt.X("Feature set:N", sort=[FEATURE_SET_NAMES[s] for s in FEATURE_SET_ORDER], title=None, axis=alt.Axis(labelAngle=0))
        colour = alt.Color("Model:N", scale=alt.Scale(domain=["LR", "RF", "XGB", "SVM"], range=SERIES[:4]))
        pts = alt.Chart(d).mark_point(filled=True, size=90).encode(
            x=xenc, xOffset="Model:N", y=alt.Y("f1_macro:Q", scale=alt.Scale(zero=False), title="macro-F1 (mean ± std)"),
            color=colour, shape="Model:N",
            tooltip=["Model", "Feature set", alt.Tooltip("f1_macro:Q", format=".4f"), alt.Tooltip("f1_macro_std:Q", format=".4f", title="std")],
        )
        bars = alt.Chart(d).mark_rule(strokeWidth=2).encode(x=xenc, xOffset="Model:N", y="lo:Q", y2="hi:Q", color=colour)
        show_chart(bars + pts, 290)
    with right:
        st.subheader("Accuracy vs training cost")
        d2 = cv.copy()
        d2["Model"] = d2.model.map(MODEL_SHORT)
        d2["Feature set"] = d2.feature_set.map(FEATURE_SET_NAMES)
        d2["Component"] = d2.component.map(lambda c: TARGET_SPECS[c].display_name)
        sc = alt.Chart(d2).mark_point(filled=True, size=80, opacity=0.85).encode(
            x=alt.X("training_time:Q", scale=alt.Scale(type="log"), title="mean fit time per fold [s, log]"),
            y=alt.Y("f1_macro:Q", scale=alt.Scale(domain=[0.5, 1.0]), title="CV macro-F1"),
            color=alt.Color("Model:N", scale=alt.Scale(domain=["LR", "RF", "XGB", "SVM"], range=SERIES[:4])),
            shape=alt.Shape("Feature set:N", sort=[FEATURE_SET_NAMES[s] for s in FEATURE_SET_ORDER]),
            tooltip=["Component", "Model", "Feature set", alt.Tooltip("f1_macro:Q", format=".4f"), alt.Tooltip("training_time:Q", format=".2f")],
        )
        show_chart(sc, 290)

    st.subheader("How each winner was chosen")
    note(
        "Selection rule: the highest mean cross-validated macro-F1 on the <b>training set only</b>; ties are broken in "
        "favour of the simpler feature set and model. The test set is touched exactly once, after selection."
    )
    rows = []
    for c in COMPONENTS:
        d = cv[cv.component == c].sort_values("f1_macro", ascending=False).head(3)
        for rank, r in enumerate(d.itertuples(), start=1):
            rows.append(
                {"Component": TARGET_SPECS[c].display_name, "Rank": rank, "Model": MODEL_NAMES[r.model],
                 "Feature set": FEATURE_SET_NAMES[r.feature_set], "CV macro-F1": round(r.f1_macro, 4),
                 "± std": round(r.f1_macro_std, 4), "Fit time [s]": round(r.training_time, 2)}
            )
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=36 * (len(rows) + 1) + 4, **FULL_WIDTH)
    st.caption("The top three configurations of every target sit within about one fold standard deviation of each other, so their ranking is not statistically meaningful.")


# ==========================================================================
# 3. Pipeline inspector
# ==========================================================================
def _kv_table(items: List[Tuple[str, object]]) -> str:
    return '<table class="kv">' + "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in items) + "</table>"


def tab_pipeline(component: str) -> None:
    spec = TARGET_SPECS[component]
    section_header("04", "Pipeline inspector", f"What is inside models/{component}/model.joblib")
    b = load_bundles()[component]
    pipe = b["model"]
    meta: Dict[str, object] = b["meta"]  # type: ignore[assignment]
    est = pipe.named_steps["model"]  # type: ignore[union-attr]

    # --- step flow read from the fitted object --------------------------
    parts: List[str] = []
    for i, (name, step) in enumerate(pipe.steps):  # type: ignore[union-attr]
        n_in = getattr(step, "n_features_in_", None)
        sel = getattr(step, "selected_index_", None)
        detail = f"{n_in} → {len(sel)}" if sel is not None and n_in is not None else (f"{n_in} in" if n_in is not None else "")
        if i:
            parts.append('<div class="wire"></div>')
        parts.append(f'<div class="node" style="animation-delay:{i * 0.08:.2f}s,{i * 0.4:.1f}s"><b>{esc(name)}</b><span>{esc(type(step).__name__)}<br>{esc(detail)}</span></div>')
    render_html(f'<div class="flow">{"".join(parts)}</div>')

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Classifier")
        params = est.get_params()
        keys = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_samples_leaf",
                "max_features", "C", "gamma", "kernel", "class_weight", "balance_classes", "tree_method", "random_state"]
        items: List[Tuple[str, object]] = [("Estimator", type(est).__name__)]
        items += [(k, params[k]) for k in keys if k in params and params[k] is not None]
        if hasattr(est, "estimators_"):
            depths = [t.get_depth() for t in est.estimators_]
            leaves = [t.get_n_leaves() for t in est.estimators_]
            items += [("Mean tree depth (fitted)", f"{np.mean(depths):.1f}"), ("Mean leaves per tree (fitted)", f"{np.mean(leaves):.0f}")]
        items += [("Inputs reaching classifier", getattr(est, "n_features_in_", "?"))]
        render_html(f'<div class="panel">{_kv_table(items)}</div>')

        st.subheader("Class imbalance handling")
        train_idx: np.ndarray = b["train_idx"]  # type: ignore[assignment]
        y_train = load_features().loc[train_idx, spec.column].to_numpy()
        vals, counts = np.unique(y_train, return_counts=True)
        balanced = meta.get("class_weight") == "balanced"
        weights = len(y_train) / (len(vals) * counts) if balanced else np.ones_like(counts, dtype=float)
        wmax = weights.max()
        rows = [
            (class_label(component, int(v)), w / wmax, f"n={int(n)} · w={w:.2f}", SERIES[i % len(SERIES)])
            for i, (v, n, w) in enumerate(zip(vals, counts, weights))
        ]
        render_html(
            '<div class="panel"><h4>Training-fold class weights</h4>'
            + "".join(
                f'<div class="bar-row"><div class="name">{esc(n)}</div><div class="track"><div class="fill" style="width:{f * 100:.1f}%;background:{c}"></div></div><div class="num">{esc(t)}</div></div>'
                for n, f, t, c in rows
            )
            + f'<div class="sec-sub" style="margin-top:.5rem">Imbalance ratio {float(meta["imbalance_ratio"]):.2f} → '
            + ("inverse-frequency weights w = N / (K · N<sub>c</sub>)." if balanced else "near-uniform classes, no weighting.")
            + "</div></div>"
        )

    with right:
        st.subheader("Feature funnel inside the pipeline")
        stages = [("Input columns", len(meta["input_feature_columns"]))]  # type: ignore[arg-type]
        for name, step in pipe.steps:  # type: ignore[union-attr]
            if hasattr(step, "selected_index_"):
                stages.append((name.replace("_", " "), len(step.selected_index_)))
        stages.append(("Reach classifier", int(meta["n_selected_features"])))
        fdf = pd.DataFrame(stages, columns=["Stage", "Features"])
        fdf["order"] = range(len(fdf))
        funnel = alt.Chart(fdf).mark_bar(cornerRadiusEnd=4, color=SERIES[0]).encode(
            y=alt.Y("Stage:N", sort=alt.SortField("order"), title=None),
            x=alt.X("Features:Q", title="features"),
            tooltip=["Stage", "Features"],
        ) + alt.Chart(fdf).mark_text(align="left", dx=5, color=INK, fontSize=11).encode(
            y=alt.Y("Stage:N", sort=alt.SortField("order")), x="Features:Q", text="Features:Q"
        )
        show_chart(funnel, 60 + 38 * len(fdf))

        st.subheader("Features reaching the classifier")
        run_meta = load_run_metadata() or {}
        block_of = {f: blk for blk, fs in run_meta.get("feature_block_names", {}).items() for f in fs}  # type: ignore[union-attr]
        names: List[str] = list(meta["selected_feature_names"])  # type: ignore[arg-type]
        legend = "".join(f'<span><i style="background:{c}"></i>{esc(k.replace("_", " "))}</span>' for k, c in BLOCK_COLOURS.items())
        shown = names[:120]
        chips = "".join(
            f'<span class="fchip" style="--c:{BLOCK_COLOURS.get(block_of.get(n, ""), "#94A3B8")};animation-delay:{min(i, 40) * 0.015:.2f}s">{esc(n)}</span>'
            for i, n in enumerate(shown)
        )
        more = f'<div class="sec-sub" style="margin-top:.5rem">+ {len(names) - len(shown)} more</div>' if len(names) > len(shown) else ""
        render_html(f'<div class="panel"><div class="legend">{legend}</div><div class="chipcloud">{chips}</div>{more}</div>')


# ==========================================================================
# 4. Evaluation lab
# ==========================================================================
def tab_evaluation(component: str) -> None:
    spec = TARGET_SPECS[component]
    section_header("05", "Evaluation lab", f"{spec.display_name} - live scoring of the {len(load_bundles()[component]['test_idx'])} held-out test cycles")
    ev = evaluate(component)
    y, pred, proba, classes = ev["y"], ev["pred"], ev["proba"], ev["classes"]  # type: ignore[misc]
    k = len(classes)
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    auc = roc_auc_score(y, proba, multi_class="ovr", average="macro") if k > 2 else roc_auc_score(y, proba[:, 1])
    conf = proba.max(axis=1)
    correct = pred == y
    # Expected calibration error on the top-label confidence.
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    stat_cards(
        [
            ("Accuracy", f"{acc:.4f}", f"{int(correct.sum())} / {len(y)} correct", tone_for(acc)),
            ("Macro-F1", f"{f1m:.4f}", "every class weighted equally", tone_for(f1m)),
            ("ROC-AUC (OvR macro)", f"{auc:.4f}", "ranking quality of probabilities", tone_for(auc)),
            ("Errors", str(int((~correct).sum())), "misclassified test cycles", "good" if correct.all() else "bad"),
            ("Calibration error (ECE)", f"{ece:.3f}", "0 = confidence matches accuracy", "good" if ece < 0.05 else "warn"),
        ],
        min_width=160,
    )

    labels = [class_label(component, c) for c in classes]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, pred, labels=list(range(k)))
        cmdf = pd.DataFrame(
            [{"True": labels[i], "Predicted": labels[j], "Cycles": int(cm[i, j]),
              "Recall share": cm[i, j] / max(cm[i].sum(), 1)} for i in range(k) for j in range(k)]
        )
        base = alt.Chart(cmdf).encode(
            x=alt.X("Predicted:N", sort=labels, title="predicted", axis=alt.Axis(labelAngle=-25, labelLimit=160)),
            y=alt.Y("True:N", sort=labels, title="true", axis=alt.Axis(labelLimit=160)),
        )
        heat = base.mark_rect(cornerRadius=4, stroke="#0B1220", strokeWidth=2).encode(
            color=alt.Color("Recall share:Q", scale=alt.Scale(range=SEQ, domain=[0, 1]), legend=None),
            tooltip=["True", "Predicted", "Cycles", alt.Tooltip("Recall share:Q", format=".1%")],
        )
        txt = base.mark_text(fontSize=13, fontWeight=700).encode(
            text="Cycles:Q", color=alt.condition(alt.datum["Recall share"] > 0.85, alt.value("#0B1220"), alt.value(INK))
        )
        show_chart(heat + txt, 320)
    with right:
        st.subheader("Per-class report")
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(k)), zero_division=0)
        rep = pd.DataFrame({"Class": labels, "Precision": p.round(4), "Recall": r.round(4), "F1": f.round(4), "Support": s})
        st.dataframe(rep, hide_index=True, **FULL_WIDTH)
        severity = [spec.severity.get(c, "Unknown") for c in classes]
        crit = [i for i, sv in enumerate(severity) if sv == "Critical"]
        if crit:
            i = crit[0]
            note(
                f"<b>Maintenance view:</b> recall on the critical class <b>{esc(labels[i])}</b> is "
                f"<b>{r[i]:.3f}</b> - the fraction of genuinely critical cycles the model catches. "
                "Missing a critical state is the costly error; flagging a healthy one as critical only costs an inspection."
            )

    st.subheader("ROC and precision-recall curves (one-vs-rest)")
    roc_rows, pr_rows, summary = [], [], []
    for i in range(k):
        yi = (y == i).astype(int)
        if yi.min() == yi.max():
            continue
        fpr, tpr, _ = roc_curve(yi, proba[:, i])
        prec, rec, _ = precision_recall_curve(yi, proba[:, i])
        roc_rows += [{"Class": labels[i], "FPR": a, "TPR": b} for a, b in zip(fpr, tpr)]
        pr_rows += [{"Class": labels[i], "Recall": a, "Precision": b} for a, b in zip(rec, prec)]
        summary.append({"Class": labels[i], "AUC": round(roc_auc_score(yi, proba[:, i]), 4), "Average precision": round(average_precision_score(yi, proba[:, i]), 4)})
    colour = alt.Color("Class:N", sort=labels, scale=alt.Scale(domain=labels, range=SERIES[:k]), legend=alt.Legend(columns=2, labelLimit=220))
    c1, c2 = st.columns(2)
    with c1:
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=MUTED, opacity=0.6).encode(x="x:Q", y="y:Q")
        roc = alt.Chart(pd.DataFrame(roc_rows)).mark_line(strokeWidth=2, interpolate="step-after").encode(
            x=alt.X("FPR:Q", title="false positive rate"), y=alt.Y("TPR:Q", title="true positive rate"), color=colour,
            tooltip=["Class", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
        )
        show_chart(diag + roc, 300)
    with c2:
        pr = alt.Chart(pd.DataFrame(pr_rows)).mark_line(strokeWidth=2, interpolate="step-after").encode(
            x=alt.X("Recall:Q", title="recall"), y=alt.Y("Precision:Q", title="precision", scale=alt.Scale(domain=[0, 1.02])), color=colour,
            tooltip=["Class", alt.Tooltip("Recall:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")],
        )
        show_chart(pr, 300)
    st.dataframe(pd.DataFrame(summary), hide_index=True, **FULL_WIDTH)

    st.subheader("Probability calibration")
    c3, c4 = st.columns(2)
    with c3:
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                rows.append({"Mean confidence": conf[m].mean(), "Observed accuracy": correct[m].mean(), "Cycles": int(m.sum())})
        cal = pd.DataFrame(rows)
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=MUTED, opacity=0.6).encode(
            x=alt.X("x:Q", title="mean predicted confidence"), y=alt.Y("y:Q", title="observed accuracy"))
        line = alt.Chart(cal).mark_line(color=SERIES[0], strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
            x="Mean confidence:Q", y="Observed accuracy:Q",
            tooltip=[alt.Tooltip("Mean confidence:Q", format=".3f"), alt.Tooltip("Observed accuracy:Q", format=".3f"), "Cycles"],
        )
        show_chart(diag + line, 280)
        st.caption("Reliability diagram of the top-class probability. Points on the dashed line mean '80 % confident' really is right 80 % of the time.")
    with c4:
        hdf = pd.DataFrame({"Confidence": conf, "Outcome": np.where(correct, "correct", "wrong")})
        hist = alt.Chart(hdf).mark_bar(binSpacing=2, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Confidence:Q", bin=alt.Bin(extent=[0, 1], step=0.05), title="top-class probability"),
            y=alt.Y("count():Q", title="test cycles", scale=alt.Scale(type="symlog")),
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=["correct", "wrong"], range=["#22C55E", "#EF4444"])),
            tooltip=["Outcome", "count()"],
        )
        show_chart(hist, 280)
        st.caption("Confidence distribution (symlog scale). Wrong predictions should sit at low confidence.")

    st.subheader("Misclassified test cycles")
    wrong = np.where(~correct)[0]
    if len(wrong) == 0:
        note("No test cycle is misclassified for this component. See the Generalization tab for why a perfect random-split score is still an upper bound.")
    else:
        idx: np.ndarray = ev["idx"]  # type: ignore[assignment]
        table = load_features()
        st.dataframe(
            pd.DataFrame(
                {
                    "Cycle": idx[wrong],
                    "True": [labels[i] for i in y[wrong]],
                    "Predicted": [labels[i] for i in pred[wrong]],
                    "Confidence": conf[wrong].round(3),
                    "P(true class)": proba[wrong, y[wrong]].round(3),
                    "Stable flag": table.loc[idx[wrong], "stable_flag"].to_numpy(),
                }
            ),
            hide_index=True, **FULL_WIDTH,
        )


# ==========================================================================
# 5. Generalization
# ==========================================================================
def tab_generalization(component: str) -> None:
    section_header("06", "Generalization", "Random-split scores versus temporally blocked validation")
    test = load_metrics()["test"]
    if test is None:
        st.warning("Run `python src/evaluate.py` first.")
        return
    rows = []
    for r in test.itertuples():
        for col, name in [("block_cv_f1_macro", "Blocked CV (realistic)"), ("cv_f1_macro", "Random CV"), ("f1_macro", "Test (random split)")]:
            rows.append({"Component": TARGET_SPECS[r.component].display_name, "Estimate": name, "macro-F1": float(getattr(r, col))})
    gdf = pd.DataFrame(rows)
    order = ["Blocked CV (realistic)", "Random CV", "Test (random split)"]
    bars = alt.Chart(gdf).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Component:N", sort=[TARGET_SPECS[c].display_name for c in COMPONENTS], title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Estimate:N", sort=order),
        y=alt.Y("macro-F1:Q", scale=alt.Scale(domain=[0.88, 1.0]), title="macro-F1 (axis starts at 0.88)", axis=alt.Axis(format=".2f")),
        y2=alt.datum(0.88),
        color=alt.Color("Estimate:N", sort=order, scale=alt.Scale(domain=order, range=["#22C55E", SERIES[0], "#818CF8"])),
        tooltip=["Component", "Estimate", alt.Tooltip("macro-F1:Q", format=".4f")],
    )
    labels = alt.Chart(gdf[gdf.Estimate == order[0]]).mark_text(dy=-6, fontSize=10, color=INK).encode(
        x=alt.X("Component:N", sort=[TARGET_SPECS[c].display_name for c in COMPONENTS]),
        xOffset=alt.XOffset("Estimate:N", sort=order), y="macro-F1:Q", text=alt.Text("macro-F1:Q", format=".3f"),
    )
    show_chart(bars + labels, 320)

    stat_cards(
        [
            (TARGET_SPECS[r.component].display_name, f"{float(r.cv_f1_macro) - float(r.block_cv_f1_macro):+.3f}",
             "random CV minus blocked CV", "good" if float(r.cv_f1_macro) - float(r.block_cv_f1_macro) < 0.02 else "warn")
            for r in test.itertuples()
        ]
    )
    note(
        "<b>Why the gap exists.</b> Consecutive cycles are recorded seconds apart under identical settings, so they are "
        "near-duplicates. A random split puts neighbours on both sides of the train/test boundary and the model is "
        "partly tested on cycles it has effectively seen. Blocked CV holds out whole runs of 25 consecutive cycles, "
        "which removes that shortcut. Report the blocked score as the realistic one and the random-split test score "
        "as an upper bound."
    )

    st.subheader(f"Why random splits leak - {TARGET_SPECS[component].display_name}")
    table = load_features()
    spec = TARGET_SPECS[component]
    b = load_bundles()[component]
    test_set = set(int(i) for i in b["test_idx"])  # type: ignore[union-attr]
    window = st.slider("Cycle window", 0, len(table) - 1, (0, 300), key="leak_window")
    seg = table.loc[window[0]:window[1], ["cycle_id", spec.column]].copy()
    seg["Condition"] = seg[spec.column].map(lambda v: class_label(component, int(v)))
    seg["Split"] = seg.cycle_id.map(lambda i: "test" if int(i) in test_set else "train")
    classes = [class_label(component, int(c)) for c in sorted(table[spec.column].unique())]
    tl = alt.Chart(seg).mark_tick(thickness=2, size=26).encode(
        x=alt.X("cycle_id:Q", title="cycle index (recording order)"),
        y=alt.Y("Split:N", sort=["train", "test"], title=None),
        color=alt.Color("Condition:N", sort=classes, scale=alt.Scale(domain=classes, range=SERIES[: len(classes)]), legend=alt.Legend(columns=2, labelLimit=240)),
        tooltip=["cycle_id", "Condition", "Split"],
    )
    show_chart(tl, 150)
    st.caption("Test cycles (bottom row) are scattered among training cycles recorded moments before and after, under the same condition.")


# ==========================================================================
# 6. Error analysis
# ==========================================================================
def tab_errors(component: str) -> None:
    spec = TARGET_SPECS[component]
    section_header("07", "Error analysis", f"{spec.display_name} - why the hard cycles are hard")
    ev = evaluate(component)
    y, pred, proba, classes = ev["y"], ev["pred"], ev["proba"], ev["classes"]  # type: ignore[misc]
    idx: np.ndarray = ev["idx"]  # type: ignore[assignment]
    labels = [class_label(component, c) for c in classes]
    wrong = np.where(pred != y)[0]
    if len(wrong):
        pick_from = wrong
        note(f"<b>{len(wrong)}</b> misclassified test cycle(s). Pick one to compare it with the class profiles the model learned.")
    else:
        pick_from = np.argsort(proba[np.arange(len(y)), y])[:10]
        note("No test errors for this component, so the ten <b>least confident correct</b> predictions are shown instead - the cycles closest to a decision boundary.")

    options = [int(i) for i in pick_from]
    pos = st.selectbox(
        "Cycle", options,
        format_func=lambda i: f"cycle {idx[i]} · true {labels[y[i]]} → predicted {labels[pred[i]]} ({proba[i].max():.2f})",
        key=f"err_pick_{component}",
    )
    b = load_bundles()[component]
    imp = load_metrics()["importance"]
    top = imp[(imp.component == component) & (imp.kind == "impurity")].sort_values("importance", ascending=False).feature_name.head(10).tolist()
    table = load_features()
    train_idx: np.ndarray = b["train_idx"]  # type: ignore[assignment]
    train = table.loc[train_idx]
    mu, sd = train[top].mean(), train[top].std().replace(0, 1)
    true_v, pred_v = classes[y[pos]], classes[pred[pos]]
    cyc = table.loc[idx[pos], top]
    rows = []
    for f in top:
        rows.append({"Feature": f, "Series": "This cycle", "z": float((cyc[f] - mu[f]) / sd[f])})
        rows.append({"Feature": f, "Series": f"True class mean ({labels[y[pos]]})", "z": float((train.loc[train[spec.column] == true_v, f].mean() - mu[f]) / sd[f])})
        if pred_v != true_v:
            rows.append({"Feature": f, "Series": f"Predicted class mean ({labels[pred[pos]]})", "z": float((train.loc[train[spec.column] == pred_v, f].mean() - mu[f]) / sd[f])})
    zdf = pd.DataFrame(rows)
    series = list(dict.fromkeys(zdf.Series))
    rng = [INK, "#22C55E", "#EF4444"][: len(series)]
    rule = alt.Chart(zdf).mark_rule(color=GRID, strokeWidth=6).encode(y=alt.Y("Feature:N", sort=top, title=None), x=alt.X("min(z):Q", title="z-score vs training set"), x2="max(z):Q")
    dots = alt.Chart(zdf).mark_point(filled=True, size=120, opacity=0.95).encode(
        y=alt.Y("Feature:N", sort=top), x="z:Q",
        color=alt.Color("Series:N", sort=series, scale=alt.Scale(domain=series, range=rng), legend=alt.Legend(labelLimit=320)),
        shape=alt.Shape("Series:N", sort=series, scale=alt.Scale(domain=series, range=["diamond", "circle", "square"][: len(series)])),
        tooltip=["Feature", "Series", alt.Tooltip("z:Q", format=".2f")],
    )
    show_chart(rule + dots, 40 + 32 * len(top))
    st.caption("For the model's ten most important features: where this cycle sits relative to the average cycle of its true class and of the class it was confused with. A cycle that sits closer to the wrong class profile on the important features explains the error.")

    c1, c2 = st.columns([1, 1])
    with c1:
        pdf = pd.DataFrame({"Class": labels, "Probability": proba[pos]})
        chart = alt.Chart(pdf).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Class:N", sort=labels, title=None, axis=alt.Axis(labelLimit=240)), x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.Class == labels[pred[pos]], alt.value(SERIES[0]), alt.value("rgba(148,163,184,0.45)")),
            tooltip=["Class", alt.Tooltip("Probability:Q", format=".3f")],
        )
        show_chart(chart, 60 + 34 * len(labels))
    with c2:
        stable = int(table.loc[idx[pos], "stable_flag"])
        render_html(
            '<div class="panel"><h4>Cycle facts</h4>'
            + _kv_table([
                ("Cycle index", int(idx[pos])),
                ("True condition", labels[y[pos]]),
                ("Predicted", labels[pred[pos]]),
                ("Margin (top-1 − top-2)", f"{np.sort(proba[pos])[-1] - np.sort(proba[pos])[-2]:.3f}"),
                ("Stable flag", "1 · static conditions may not be reached" if stable else "0 · stable"),
            ])
            + "</div>"
        )


# ==========================================================================
# 7. Explainability
# ==========================================================================
def tab_explain(component: str) -> None:
    spec = TARGET_SPECS[component]
    section_header("08", "Explainability", f"{spec.display_name} - global importance and a local what-if probe")
    imp = load_metrics()["importance"]
    d = imp[imp.component == component]
    top_n = st.slider("Features shown", 5, 25, 12, key=f"imp_n_{component}")
    c1, c2 = st.columns(2)
    for col, kind, title, colour in [(c1, "impurity", "Impurity importance (model internals)", SERIES[0]), (c2, "permutation", "Permutation importance (test macro-F1 drop)", SERIES[1])]:
        with col:
            st.markdown(f"**{title}**")
            dk = d[d.kind == kind].sort_values("importance", ascending=False).head(top_n)
            ch = alt.Chart(dk).mark_bar(cornerRadiusEnd=4, color=colour).encode(
                y=alt.Y("feature_name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("importance:Q", title="importance"),
                tooltip=["feature_name", alt.Tooltip("importance:Q", format=".4f")],
            )
            show_chart(ch, 40 + 22 * len(dk))
    note(
        "Impurity importance says what the trees <i>split on</i>; permutation importance says what the model "
        "<i>cannot do without</i> on unseen data. When many features are correlated (for example the 553-input "
        "accumulator forest), permutation importance spreads out and every single feature looks unimportant - "
        "because a correlated neighbour can take over when one is shuffled."
    )

    st.subheader("Local what-if probe")
    b = load_bundles()[component]
    table = load_features()
    meta: Dict[str, object] = b["meta"]  # type: ignore[assignment]
    cols: List[str] = list(meta["input_feature_columns"])  # type: ignore[arg-type]
    top = d[d.kind == "impurity"].sort_values("importance", ascending=False).feature_name.head(10).tolist()
    top = [f for f in top if f in cols]
    labels = [class_label(component, int(c)) for c in b["encoder"].classes_]  # type: ignore[union-attr]
    c1, c2 = st.columns([1, 1])
    with c1:
        cycle = st.number_input("Cycle", 0, len(table) - 1, int(b["test_idx"][0]), key=f"wi_cycle_{component}")  # type: ignore[index]
    with c2:
        feat = st.selectbox("Feature to vary", top, key=f"wi_feat_{component}")
    train_vals = table.loc[b["train_idx"], feat]  # type: ignore[index]
    lo, hi = float(train_vals.quantile(0.01)), float(train_vals.quantile(0.99))
    current = float(table.loc[cycle, feat])
    val = st.slider(
        f"{feat} (training 1st-99th percentile)", min(lo, current), max(hi, current), current,
        key=f"wi_val_{component}_{feat}_{cycle}",
    )
    x0 = table.loc[[cycle], cols].to_numpy(dtype=np.float64)
    x1 = x0.copy()
    x1[0, cols.index(feat)] = val
    p0 = b["model"].predict_proba(x0)[0]  # type: ignore[union-attr]
    p1 = b["model"].predict_proba(x1)[0]  # type: ignore[union-attr]
    wdf = pd.DataFrame(
        [{"Class": labels[i], "Scenario": "original", "Probability": p0[i]} for i in range(len(labels))]
        + [{"Class": labels[i], "Scenario": "what-if", "Probability": p1[i]} for i in range(len(labels))]
    )
    ch = alt.Chart(wdf).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Class:N", sort=labels, title=None, axis=alt.Axis(labelLimit=240)),
        yOffset=alt.YOffset("Scenario:N", sort=["original", "what-if"]),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Scenario:N", sort=["original", "what-if"], scale=alt.Scale(domain=["original", "what-if"], range=["rgba(148,163,184,0.55)", SERIES[0]])),
        tooltip=["Class", "Scenario", alt.Tooltip("Probability:Q", format=".3f")],
    )
    show_chart(ch, 70 + 48 * len(labels))
    true_v = int(table.loc[cycle, spec.column])
    changed = int(np.argmax(p0)) != int(np.argmax(p1))
    render_html(
        '<div class="panel">'
        + _kv_table([
            ("Recorded condition", class_label(component, true_v)),
            ("Original prediction", f"{labels[int(np.argmax(p0))]} ({p0.max():.3f})"),
            ("What-if prediction", f"{labels[int(np.argmax(p1))]} ({p1.max():.3f})"),
            (f"{feat}", f"{current:.4g} → {val:.4g}"),
        ])
        + f'<div style="margin-top:.6rem"><span class="chip {"bad" if changed else "ok"}">{"decision changed" if changed else "decision unchanged"}</span></div></div>'
    )
    st.caption("Only one input is changed; the rest of the cycle is kept as recorded. Because real features are correlated, large single-feature moves can create cycles that never occur physically - treat this as a sensitivity probe, not a simulation.")


# ==========================================================================
# 8. Robustness and confidence
# ==========================================================================
@st.cache_data(show_spinner="Running the noise stress test...")
def noise_curve(component: str, levels: Tuple[float, ...], repeats: int = 3) -> pd.DataFrame:
    b = load_bundles()[component]
    X, y = xy(component, b["test_idx"])  # type: ignore[arg-type]
    Xtr, _ = xy(component, b["train_idx"])  # type: ignore[arg-type]
    sd = np.nanstd(Xtr, axis=0)
    rng = np.random.default_rng(42)
    rows = []
    for s in levels:
        scores = []
        for _ in range(repeats if s > 0 else 1):
            Xn = X + rng.normal(0, 1, X.shape) * sd * s
            scores.append(f1_score(y, b["model"].predict(Xn), average="macro"))  # type: ignore[union-attr]
        rows.append({"Noise (σ of each feature)": s, "macro-F1": float(np.mean(scores)), "min": float(np.min(scores)), "max": float(np.max(scores))})
    return pd.DataFrame(rows)


def tab_robustness(component: str) -> None:
    spec = TARGET_SPECS[component]
    section_header("09", "Robustness & confidence", f"{spec.display_name} - stress test and selective prediction")

    st.subheader("Noise stress test")
    st.caption("Gaussian noise is added to every model input, scaled to that feature's training standard deviation, and the held-out test set is re-scored (mean of 3 draws).")
    levels = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    all_comp = st.toggle("Compare all four components", value=False, key="noise_all")
    comps = list(COMPONENTS) if all_comp else [component]
    frames = []
    for c in comps:
        df = noise_curve(c, levels)
        df["Component"] = TARGET_SPECS[c].display_name
        frames.append(df)
    ndf = pd.concat(frames)
    names = [TARGET_SPECS[c].display_name for c in comps]
    colour = alt.Color("Component:N", sort=names, scale=alt.Scale(domain=names, range=SERIES[: len(names)]))
    band = alt.Chart(ndf).mark_area(opacity=0.18).encode(x="Noise (σ of each feature):Q", y="min:Q", y2="max:Q", color=colour)
    line = alt.Chart(ndf).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
        x=alt.X("Noise (σ of each feature):Q", title="noise level (× feature std)"),
        y=alt.Y("macro-F1:Q", scale=alt.Scale(zero=False), title="test macro-F1"),
        color=colour, tooltip=["Component", "Noise (σ of each feature):Q", alt.Tooltip("macro-F1:Q", format=".4f")],
    )
    show_chart(band + line, 320)
    cur = ndf[ndf.Component == spec.display_name].set_index("Noise (σ of each feature)")["macro-F1"]
    stat_cards(
        [
            ("Clean", f"{cur.loc[0.0]:.3f}", "no noise", tone_for(cur.loc[0.0])),
            ("Noise 0.1 × std", f"{cur.loc[0.1]:.3f}", "small sensor jitter", tone_for(cur.loc[0.1])),
            ("Noise 0.3 × std", f"{cur.loc[0.3]:.3f}", "moderate", tone_for(cur.loc[0.3])),
            ("Noise 1.0 × std", f"{cur.loc[1.0]:.3f}", "severe", tone_for(cur.loc[1.0])),
        ]
    )

    st.subheader("Selective prediction")
    st.caption("If the model may answer 'not sure' below a confidence threshold, how accurate is it on the cycles it does answer, and how many does it answer?")
    ev = evaluate(component)
    y, pred, proba = ev["y"], ev["pred"], ev["proba"]  # type: ignore[misc]
    conf = proba.max(axis=1)
    correct = pred == y
    grid = np.round(np.arange(0.30, 1.0001, 0.01), 2)
    sel = pd.DataFrame(
        [{"Threshold": t, "Coverage": float((conf >= t).mean()),
          "Accuracy on answered": float(correct[conf >= t].mean()) if (conf >= t).any() else np.nan} for t in grid]
    )
    thr = st.slider("Confidence threshold", 0.30, 1.00, 0.60, 0.01, key=f"sel_thr_{component}")
    long = sel.melt("Threshold", ["Coverage", "Accuracy on answered"], var_name="Metric", value_name="Value")
    lines = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Threshold:Q", title="confidence threshold"),
        y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title=None),
        color=alt.Color("Metric:N", scale=alt.Scale(domain=["Coverage", "Accuracy on answered"], range=[SERIES[0], "#22C55E"])),
        tooltip=["Metric", alt.Tooltip("Threshold:Q", format=".2f"), alt.Tooltip("Value:Q", format=".4f")],
    )
    rule = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(color="#F59E0B", strokeDash=[5, 4], strokeWidth=2).encode(x="t:Q")
    show_chart(lines + rule, 300)
    answered = conf >= thr
    stat_cards(
        [
            ("Answered", f"{answered.mean() * 100:.1f} %", f"{int(answered.sum())} of {len(y)} cycles", ""),
            ("Deferred to engineer", str(int((~answered).sum())), "below the threshold", ""),
            ("Accuracy when answering", f"{correct[answered].mean():.4f}" if answered.any() else "n/a", "on answered cycles only",
             tone_for(correct[answered].mean()) if answered.any() else ""),
            ("Errors still answered", str(int((~correct & answered).sum())), f"of {int((~correct).sum())} total errors", "good" if not (~correct & answered).any() else "bad"),
        ]
    )


# ==========================================================================
# Layout
# ==========================================================================
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    render_hero()
    component = render_sidebar()
    if not artefacts_ready():
        st.error(
            "Saved pipelines or the engineered feature table are missing.\n\n"
            "Run the training pipeline first:\n\n```bash\npython src/train.py\npython src/evaluate.py\n```"
        )
        return
    tabs = st.tabs(
        ["🏠 Overview", "🔮 Predict", "🏟️ Model arena", "🧩 Pipeline", "🧪 Evaluation lab", "🌐 Generalization",
         "🔍 Error analysis", "💡 Explainability", "🛡️ Robustness"]
    )
    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_predict(component)
    with tabs[2]:
        tab_arena()
    with tabs[3]:
        tab_pipeline(component)
    with tabs[4]:
        tab_evaluation(component)
    with tabs[5]:
        tab_generalization(component)
    with tabs[6]:
        tab_errors(component)
    with tabs[7]:
        tab_explain(component)
    with tabs[8]:
        tab_robustness(component)


if __name__ == "__main__":
    main()
