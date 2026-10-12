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
    plus a Results tab with cross-validated, temporal-block and test metrics.

WHAT IT DELIBERATELY DOES NOT DO
    It never trains anything. Every prediction comes from the pipelines saved by
    `src/train.py`, loaded once and cached. That is not just a speed decision:
    re-fitting a scaler on whatever data the app happens to hold would put the
    classifier's inputs on a different scale from training and silently corrupt
    every prediction.

STATE HANDLING
    Streamlit re-runs the whole script on every widget interaction, and
    `st.button` is only True on the single rerun triggered by the click. The
    selected cycle is therefore kept in `st.session_state`, so changing the
    explainability component or the plotted sensors does not wipe the result.

RUN
    streamlit run app.py
"""

from __future__ import annotations

import html as html_lib
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
    "Normal": "#22C55E",
    "Degraded": "#F59E0B",
    "Critical": "#EF4444",
    "Unknown": "#94A3B8",
}

def _streamlit_version() -> Tuple[int, int]:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return major, minor
    except ValueError:
        return 0, 0


# `use_container_width` is deprecated in newer Streamlit in favour of
# width="stretch"; pick whichever the installed version understands.
FULL_WIDTH: Dict[str, object] = (
    {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}
)

COMPONENT_ICONS = {"cooler": "❄️", "valve": "🔧", "pump": "⚙️", "accumulator": "🛢️"}


# ==========================================================================
# Styling
# ==========================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
  --accent: #38BDF8;
  --accent-2: #818CF8;
  --card: rgba(148, 163, 184, 0.08);
  --card-hover: rgba(148, 163, 184, 0.13);
  --line: rgba(148, 163, 184, 0.20);
  --muted: rgba(148, 163, 184, 0.95);
  --track: rgba(148, 163, 184, 0.16);
}

.stApp { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1400px; }

/* ---------- keyframes ---------- */
@keyframes fadeUp   { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes growX    { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes shimmer  { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes flow     { to { stroke-dashoffset: -48; } }
@keyframes pulse    {
  0%   { box-shadow: 0 0 0 0 color-mix(in srgb, var(--sev) 65%, transparent); }
  70%  { box-shadow: 0 0 0 10px transparent; }
  100% { box-shadow: 0 0 0 0 transparent; }
}
@keyframes glow     {
  0%, 100% { box-shadow: 0 0 0 1px color-mix(in srgb, var(--sev) 35%, transparent), 0 10px 30px -18px var(--sev); }
  50%      { box-shadow: 0 0 0 1px color-mix(in srgb, var(--sev) 70%, transparent), 0 14px 38px -12px var(--sev); }
}
@keyframes slideArrow { 0% { transform: translateX(-4px); opacity: .35; } 50% { transform: translateX(3px); opacity: 1; } 100% { transform: translateX(-4px); opacity: .35; } }

/* ---------- hero ---------- */
.hero {
  position: relative; overflow: hidden;
  padding: 1.8rem 2rem 1.3rem; border-radius: 20px; margin-bottom: 1.2rem;
  border: 1px solid var(--line);
  background: linear-gradient(120deg, rgba(56,189,248,.16), rgba(129,140,248,.14), rgba(34,197,94,.10), rgba(56,189,248,.16));
  background-size: 300% 300%;
  animation: shimmer 14s ease infinite, fadeUp .6s ease both;
}
.hero-eyebrow { font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
.hero-title { font-size: 2.05rem; font-weight: 800; line-height: 1.15; margin: .35rem 0 .45rem; }
.hero-title span {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { font-size: .98rem; opacity: .85; max-width: 60rem; }
.hero svg { width: 100%; height: 18px; margin-top: 1rem; display: block; }
.hero .pipe { stroke: var(--track); stroke-width: 8; stroke-linecap: round; fill: none; }
.hero .oil  { stroke: var(--accent); stroke-width: 3; stroke-linecap: round; fill: none;
              stroke-dasharray: 14 10; animation: flow 1.2s linear infinite; opacity: .9; }

/* ---------- section headers ---------- */
.sec { display: flex; align-items: center; gap: .9rem; margin: 1.4rem 0 .9rem; animation: fadeUp .5s ease both; }
.sec-num {
  flex: none; width: 2.4rem; height: 2.4rem; border-radius: 12px;
  display: grid; place-items: center; font-weight: 800; font-size: .95rem; color: #0B1220;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
}
.sec-title { font-size: 1.45rem; font-weight: 750; line-height: 1.2; }
.sec-sub { font-size: .88rem; color: var(--muted); }

/* ---------- stat cards ---------- */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: .8rem; margin: .4rem 0 1rem; }
.stat {
  background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.1rem;
  animation: fadeUp .55s ease both; transition: transform .2s ease, background .2s ease, border-color .2s ease;
}
.stat:hover { transform: translateY(-3px); background: var(--card-hover); border-color: color-mix(in srgb, var(--accent) 45%, transparent); }
.stat-label { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
.stat-value { font-size: 1.75rem; font-weight: 800; margin-top: .2rem; }
.stat-hint  { font-size: .8rem; color: var(--muted); margin-top: .1rem; }

/* ---------- feature funnel ---------- */
.funnel { display: flex; align-items: stretch; gap: .4rem; flex-wrap: wrap; margin: .3rem 0 1.1rem; }
.funnel-step {
  flex: 1 1 150px; background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: .9rem 1rem; animation: fadeUp .55s ease both; text-align: center;
}
.funnel-step b { display: block; font-size: 1.5rem; font-weight: 800; }
.funnel-step span { font-size: .8rem; color: var(--muted); }
.funnel-arrow { align-self: center; color: var(--accent); font-size: 1.3rem; animation: slideArrow 1.6s ease-in-out infinite; }

/* ---------- generic bars ---------- */
.bar-row { display: grid; grid-template-columns: minmax(120px, 1.1fr) 3fr minmax(58px, auto); gap: .7rem; align-items: center; margin: .38rem 0; font-size: .88rem; }
.bar-row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-row .mono { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .8rem; }
.bar-row .num { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.track { height: 10px; border-radius: 99px; background: var(--track); overflow: hidden; }
.fill  { height: 100%; border-radius: 99px; transform-origin: left center; animation: growX .9s cubic-bezier(.2,.8,.2,1) both; }
.bar-row.hit .name { font-weight: 700; }

/* ---------- status banner ---------- */
.banner {
  display: flex; align-items: center; gap: 1.2rem; flex-wrap: wrap;
  padding: 1.1rem 1.4rem; border-radius: 18px; margin: .4rem 0 1rem;
  border: 1px solid color-mix(in srgb, var(--sev) 45%, transparent);
  background: linear-gradient(100deg, color-mix(in srgb, var(--sev) 18%, transparent), transparent 70%);
  animation: fadeUp .5s ease both;
}
.banner-label { font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 600; }
.banner-value { font-size: 1.9rem; font-weight: 800; color: var(--sev); line-height: 1.1; }
.banner-sub   { font-size: .85rem; color: var(--muted); }
.banner-meta  { margin-left: auto; text-align: right; font-size: .8rem; color: var(--muted); }
.banner-meta b { display: block; font-size: 1.35rem; color: inherit; }

.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--sev); flex: none; }
.dot.live { animation: pulse 1.8s ease-out infinite; }
.dot.big  { width: 18px; height: 18px; }

/* ---------- component status cards ---------- */
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: .9rem; margin-bottom: 1rem; }
.status-card {
  position: relative; padding: 1.05rem 1.15rem 1rem; border-radius: 18px;
  background: var(--card); border: 1px solid var(--line);
  border-top: 4px solid var(--sev);
  animation: fadeUp .55s ease both; transition: transform .2s ease, background .2s ease;
}
.status-card:hover { transform: translateY(-4px); background: var(--card-hover); }
.status-card.critical { animation: fadeUp .55s ease both, glow 2.4s ease-in-out 0.6s infinite; }
.card-top { display: flex; align-items: center; gap: .5rem; font-size: .85rem; color: var(--muted); font-weight: 600; }
.card-icon { font-size: 1.1rem; }
.sev-word { font-size: 1.55rem; font-weight: 800; color: var(--sev); margin: .35rem 0 .1rem; }
.card-value { font-size: .92rem; }
.card-value b { font-size: 1.05rem; }
.card-foot { display: flex; justify-content: space-between; font-size: .76rem; color: var(--muted); margin-top: .35rem; }
.status-card .track { margin-top: .75rem; height: 7px; }
.status-card .fill { background: var(--sev); }

/* ---------- panels ---------- */
.panel-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: .9rem; margin-bottom: .9rem; }
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.15rem; animation: fadeUp .55s ease both; }
.panel h4 { margin: 0 0 .5rem; font-size: .98rem; font-weight: 700; }

/* ---------- ground-truth table ---------- */
.truth { width: 100%; border-collapse: separate; border-spacing: 0 .35rem; font-size: .9rem; }
.truth th { text-align: left; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; padding: 0 .7rem; }
.truth td { background: var(--card); padding: .6rem .7rem; }
.truth tr td:first-child { border-radius: 10px 0 0 10px; font-weight: 600; }
.truth tr td:last-child  { border-radius: 0 10px 10px 0; }
.truth tr { animation: fadeUp .45s ease both; }
.chip { display: inline-block; padding: .12rem .6rem; border-radius: 99px; font-size: .76rem; font-weight: 700; }
.chip.ok  { background: rgba(34,197,94,.18); color: #22C55E; }
.chip.bad { background: rgba(239,68,68,.18); color: #EF4444; }
.chip.neutral { background: var(--track); color: inherit; font-weight: 600; }

/* ---------- gain cards (results) ---------- */
.gain { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.1rem; animation: fadeUp .55s ease both; transition: transform .2s ease; }
.gain:hover { transform: translateY(-3px); }
.gain .g-title { font-size: .82rem; color: var(--muted); font-weight: 600; }
.gain .g-main { font-size: 1.45rem; font-weight: 800; margin: .25rem 0; }
.gain .g-main .arrow { color: var(--accent); margin: 0 .3rem; }
.gain .g-delta { font-size: .85rem; color: #22C55E; font-weight: 700; }
.legend { display: flex; gap: 1.1rem; flex-wrap: wrap; font-size: .8rem; color: var(--muted); margin: .2rem 0 .6rem; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: .35rem; vertical-align: -1px; }

/* ---------- block list ---------- */
.block-item { padding: .55rem 0; border-bottom: 1px solid var(--line); animation: fadeUp .5s ease both; }
.block-item:last-child { border-bottom: 0; }
.block-head { display: flex; align-items: center; gap: .5rem; font-weight: 700; font-size: .92rem; }
.block-head i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.block-head span { margin-left: auto; font-weight: 500; font-size: .8rem; color: var(--muted); }
.block-desc { font-size: .84rem; color: var(--muted); margin-top: .2rem; line-height: 1.45; }

/* ---------- sidebar ---------- */
.side-row { display: flex; justify-content: space-between; align-items: center; padding: .45rem 0; border-bottom: 1px solid var(--line); font-size: .88rem; }

/* ---------- native widgets ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: .55rem 1.05rem; transition: background .2s ease; font-weight: 600; }
.stTabs [data-baseweb="tab"]:hover { background: var(--card); }
.stTabs [data-baseweb="tab-panel"] { animation: fadeUp .45s ease both; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }
[data-testid="stExpander"] details { border-radius: 14px; border: 1px solid var(--line); }
.stButton > button { border-radius: 12px; font-weight: 600; transition: transform .15s ease, box-shadow .2s ease; }
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg, var(--accent), var(--accent-2)); border: 0; color: #0B1220;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  box-shadow: 0 10px 26px -10px var(--accent); color: #0B1220;
}

@media (max-width: 700px) {
  .hero { padding: 1.3rem 1.1rem 1rem; }
  .hero-title { font-size: 1.5rem; }
  .banner-meta { margin-left: 0; text-align: left; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


def render_html(markup: str) -> None:
    """Render raw HTML through st.markdown.

    Lines are stripped and joined because Markdown treats indented lines as a
    code block and a blank line as the end of an HTML block.
    """
    clean = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


def esc(value: object) -> str:
    return html_lib.escape(str(value))


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def section_header(number: str, title: str, subtitle: str = "") -> None:
    render_html(
        f"""
        <div class="sec">
          <div class="sec-num">{esc(number)}</div>
          <div><div class="sec-title">{esc(title)}</div>
          <div class="sec-sub">{esc(subtitle)}</div></div>
        </div>
        """
    )


def stat_cards(items: Iterable[Tuple[str, str, str]], min_width: int = 180) -> None:
    """A responsive grid of animated stat tiles: (label, value, hint)."""
    cells = "".join(
        f'<div class="stat" style="animation-delay:{i * 0.07:.2f}s">'
        f'<div class="stat-label">{esc(label)}</div>'
        f'<div class="stat-value">{esc(value)}</div>'
        f'<div class="stat-hint">{esc(hint)}</div></div>'
        for i, (label, value, hint) in enumerate(items)
    )
    render_html(
        f'<div class="stat-grid" style="grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr))">{cells}</div>'
    )


def bar_rows(
    rows: Iterable[Tuple[str, float, str, str]],
    mono: bool = False,
    highlight: Optional[str] = None,
    base_delay: float = 0.0,
) -> str:
    """HTML for animated horizontal bars.

    rows: (name, fraction 0..1 for bar length, right-hand text, bar colour).
    """
    out: List[str] = []
    for i, (name, fraction, right, colour) in enumerate(rows):
        width = max(0.0, min(1.0, float(fraction))) * 100
        hit = " hit" if highlight is not None and name == highlight else ""
        name_class = "name mono" if mono else "name"
        out.append(
            f'<div class="bar-row{hit}"><div class="{name_class}" title="{esc(name)}">{esc(name)}</div>'
            f'<div class="track"><div class="fill" style="width:{width:.2f}%;background:{colour};'
            f'animation-delay:{base_delay + i * 0.06:.2f}s"></div></div>'
            f'<div class="num">{esc(right)}</div></div>'
        )
    return "".join(out)


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


@st.cache_data(show_spinner=False)
def cached_cycle_features(cycle_id: int) -> pd.DataFrame:
    """Engineered features for a dataset cycle, computed once per cycle."""
    return features_from_cycle(cached_cycle(int(cycle_id)))


@st.cache_data(show_spinner=False)
def cached_test_sets() -> Dict[str, Set[int]]:
    """Held-out test cycle indices per component, from train.py's saved split."""
    sets: Dict[str, Set[int]] = {}
    for component in COMPONENTS:
        path = MODELS_DIR / component / "split_indices.npz"
        if path.exists():
            with np.load(path) as split:
                sets[component] = set(int(i) for i in split["test_idx"])
    return sets


@st.cache_resource(show_spinner="Loading the four saved pipelines...")
def cached_bundles() -> Dict[str, Dict[str, object]]:
    """Deserialise all four pipelines once per server process.

    `st.cache_resource` is Streamlit's cache for unserialisable shared objects
    such as fitted models: they are loaded once and shared across reruns and
    sessions instead of being copied.
    """
    return {component: load_component_bundle(component) for component in COMPONENTS}


def models_ready() -> bool:
    """True when all four saved pipelines are present."""
    return models_are_available()


def require_models() -> bool:
    """Show a clear instruction instead of a traceback when models are missing."""
    if models_ready():
        cached_bundles()
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
# Hero
# ==========================================================================
def render_hero() -> None:
    render_html(
        """
        <div class="hero">
          <div class="hero-eyebrow">Condition monitoring · UCI hydraulic test rig · ZeMA gGmbH</div>
          <div class="hero-title">Predictive Maintenance of <span>Industrial Hydraulic Systems</span></div>
          <div class="hero-sub">Four component-health classifiers - cooler, valve, pump and accumulator -
          driven by engineered features that compress 43,680 raw samples per cycle into a
          few hundred physically meaningful quantities.</div>
          <svg viewBox="0 0 1000 18" preserveAspectRatio="none" aria-hidden="true">
            <path class="pipe" d="M6 9 H994"/><path class="oil" d="M6 9 H994"/>
          </svg>
        </div>
        """
    )


# ==========================================================================
# Section 1 - Project overview
# ==========================================================================
def section_overview() -> None:
    section_header("01", "Project overview", "What the system does and why feature engineering is the core")

    metadata = cached_run_metadata()
    if metadata:
        blocks: Dict[str, int] = metadata["feature_blocks"]  # type: ignore[assignment]
        summary: Dict[str, object] = metadata["dataset_summary"]  # type: ignore[assignment]
        stat_cards(
            [
                ("Operating cycles", f"{int(summary['n_cycles']):,}", "60 s each"),
                ("Raw values / cycle", f"{int(summary['total_raw_dimensions']):,}", "17 sensors, 3 sampling rates"),
                ("Engineered features", f"{sum(blocks.values()):,}", "5 feature blocks"),
                ("Components predicted", str(len(COMPONENTS)), "cooler · valve · pump · accumulator"),
            ]
        )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            """
**Predictive maintenance** means judging the health of a machine from the signals
it produces while it runs, so that a component is serviced *before* it fails
rather than after. For a hydraulic power unit, an unplanned failure stops the
whole production line it feeds, so early warning is worth a great deal.

**The hydraulic system here** is a laboratory test rig with a primary working
circuit and a secondary cooling-filtration circuit, connected through the oil
tank. It repeats a constant 60-second load cycle while pressures, flows,
temperatures, motor power and vibration are recorded. Four components are
deliberately degraded to known levels: the **cooler**, a **valve**, the **pump**
(internal leakage) and the **hydraulic accumulator**.
            """
        )
    with right:
        st.markdown(
            """
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


# ==========================================================================
# Section 2 - Dataset explorer
# ==========================================================================
def section_dataset() -> None:
    section_header("02", "Dataset explorer", "Sensors, sampling rates and class balance")
    if not require_dataset():
        return

    summary = cached_dataset_summary()
    profile = cached_profile()
    assert summary is not None and profile is not None

    stat_cards(
        [
            ("Cycles", f"{int(summary['n_cycles']):,}", "rows in every sensor file"),
            ("Sensors", str(summary["n_sensors"]), "14 physical + 3 virtual"),
            ("Cycle duration", f"{summary['cycle_duration_s']} s", "constant load profile"),
            ("Raw dimensions / cycle", f"{int(summary['total_raw_dimensions']):,}", "before feature engineering"),
        ]
    )

    st.subheader("Sensors and sampling frequencies")
    sensor_frame = pd.DataFrame(summary["sensors"])  # type: ignore[arg-type]
    sensor_frame = sensor_frame.rename(
        columns={
            "sensor": "Sensor", "quantity": "Physical quantity", "unit": "Unit",
            "sampling_rate_hz": "Sampling rate [Hz]", "samples_per_cycle": "Samples per cycle",
            "group": "Group", "virtual": "Virtual channel",
        }
    )
    st.dataframe(sensor_frame, **FULL_WIDTH, hide_index=True)
    st.caption(
        "Virtual channels (CE, CP, SE) are quantities the rig computes rather than "
        "measures directly, so features built on them inherit the rig's own model."
    )

    st.subheader("Condition distributions")
    tabs = st.tabs([f"{COMPONENT_ICONS[key]} {TARGET_SPECS[key].display_name}" for key in COMPONENTS])
    for tab, key in zip(tabs, COMPONENTS):
        spec = TARGET_SPECS[key]
        with tab:
            counts = profile[spec.column].value_counts().sort_index()
            total = counts.sum()
            rows = [
                (
                    f"{int(v)} {spec.unit} - {spec.value_labels.get(int(v), 'unknown')}",
                    float(c) / float(counts.max()),
                    f"{int(c)} · {100.0 * c / total:.1f}%",
                    SEVERITY_COLOURS.get(spec.severity.get(int(v), "Unknown"), SEVERITY_COLOURS["Unknown"]),
                )
                for v, c in counts.items()
            ]
            ratio = counts.max() / counts.min()
            render_html(
                f"""
                <div class="panel">
                  <h4>{esc(spec.display_name)} - cycles per documented condition</h4>
                  {bar_rows(rows)}
                  <div class="sec-sub" style="margin-top:.6rem">Imbalance ratio (largest / smallest class):
                  <b>{ratio:.2f}</b>. Macro F1 weights every class equally, so it is the headline metric
                  rather than accuracy. Bar colour = documented severity.</div>
                </div>
                """
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
BLOCK_DESCRIPTIONS = {
    "statistical": "Distribution shape within the cycle: mean, median, std, variance, min, max, range, RMS, MAV, skewness, kurtosis, quartiles, IQR, coefficient of variation.",
    "time_domain": "How the signal evolves: peak-to-peak, peak amplitude, energy, first-difference statistics, linear slope, zero crossings, peak count.",
    "rolling": "Local behaviour in a 1-second window, sized from each sensor's own sampling rate. Only for channels at 10 Hz and above.",
    "frequency": "A deliberately small FFT summary: dominant frequency, spectral energy, spectral centroid, log-spaced band energy ratios. Only for channels at 10 Hz and above.",
    "domain": "Hydraulic relationships between channels: pressure differentials and ratios, flow imbalance, hydraulic power, electro-hydraulic efficiency, specific power, temperature spreads, cooling conductance, vibration per unit power.",
}

BLOCK_COLOURS = ["#38BDF8", "#818CF8", "#22C55E", "#F59E0B", "#F472B6"]


def section_features() -> None:
    section_header("03", "Feature engineering", "From 43,680 raw samples to a compact, physical feature vector")
    metadata = cached_run_metadata()
    if metadata is None:
        st.warning("Run `python src/train.py` to populate this section.")
        return

    summary: Dict[str, object] = metadata["dataset_summary"]  # type: ignore[assignment]
    blocks: Dict[str, int] = metadata["feature_blocks"]  # type: ignore[assignment]
    cleaning: Dict[str, object] = metadata["cleaning"]  # type: ignore[assignment]
    selected: Dict[str, Dict[str, object]] = metadata["selected_per_component"]  # type: ignore[assignment]

    raw = int(summary["total_raw_dimensions"])
    engineered = sum(blocks.values())
    per_model = [int(v["n_selected_features"]) for v in selected.values()]
    steps = [
        (f"{raw:,}", "raw samples per cycle"),
        (f"{engineered:,}", "engineered features"),
        (f"{int(cleaning['n_output']):,}", "after cleaning"),
        (f"{min(per_model)}-{max(per_model)}", "reach each classifier"),
    ]
    parts: List[str] = []
    for i, (value, label) in enumerate(steps):
        if i:
            parts.append('<div class="funnel-arrow">➜</div>')
        parts.append(
            f'<div class="funnel-step" style="animation-delay:{i * 0.12:.2f}s"><b>{esc(value)}</b><span>{esc(label)}</span></div>'
        )
    render_html(f'<div class="funnel">{"".join(parts)}</div>')
    st.caption(f"Compression: about {raw / max(engineered, 1):.0f}x fewer dimensions before any selection.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Feature blocks")
        largest = max(blocks.values())
        rows = [
            (block.replace("_", " ").capitalize(), count / largest, str(count), BLOCK_COLOURS[i % len(BLOCK_COLOURS)])
            for i, (block, count) in enumerate(blocks.items())
        ]
        render_html(f'<div class="panel">{bar_rows(rows)}</div>')

        st.subheader("Removed during cleaning")
        stat_cards(
            [
                ("Constant", str(cleaning["n_dropped_constant"]), "no information"),
                ("Near-zero variance", str(cleaning["n_dropped_near_zero_variance"]), "almost constant"),
                ("Exact duplicates", str(cleaning["n_dropped_duplicate"]), "identical columns"),
            ],
            min_width=120,
        )
    with right:
        st.subheader("What each block captures")
        items = "".join(
            f'<div class="block-item" style="animation-delay:{i * 0.07:.2f}s">'
            f'<div class="block-head"><i style="background:{BLOCK_COLOURS[i % len(BLOCK_COLOURS)]}"></i>'
            f'{esc(block.replace("_", " ").capitalize())}<span>{count} features</span></div>'
            f'<div class="block-desc">{esc(BLOCK_DESCRIPTIONS.get(block, ""))}</div></div>'
            for i, (block, count) in enumerate(blocks.items())
        )
        render_html(f'<div class="panel">{items}</div>')

    st.subheader("Selected model and features per component")
    cards = "".join(
        f'<div class="stat" style="animation-delay:{i * 0.08:.2f}s">'
        f'<div class="stat-label">{COMPONENT_ICONS[key]} {esc(TARGET_SPECS[key].display_name)}</div>'
        f'<div class="stat-value">{int(value["n_selected_features"])}<span class="stat-hint"> features</span></div>'
        f'<div class="stat-hint">{esc(str(value["model"]).replace("_", " "))} · {esc(value["feature_set"])} set · '
        f'CV macro F1 {float(value["cv_f1_macro"]):.4f}</div></div>'
        for i, (key, value) in enumerate(selected.items())
    )
    render_html(f'<div class="stat-grid">{cards}</div>')

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
            st.dataframe(subset, **FULL_WIDTH, hide_index=True)


# ==========================================================================
# Section 4 + 5 - Prediction and maintenance status
# ==========================================================================
def status_cards_html(predictions: Dict[str, Dict[str, object]]) -> str:
    cards: List[str] = []
    for i, key in enumerate(COMPONENTS):
        p = predictions[key]
        severity = str(p["severity"])
        colour = SEVERITY_COLOURS.get(severity, SEVERITY_COLOURS["Unknown"])
        confidence = p["confidence"]
        pct = float(confidence) * 100 if confidence is not None else 0.0
        conf_text = f"{pct:.1f}% confidence" if confidence is not None else "confidence n/a"
        extra = " critical" if severity == "Critical" else ""
        live = " live" if severity != "Normal" else ""
        delay = i * 0.12
        cards.append(
            f'<div class="status-card{extra}" style="--sev:{colour};animation-delay:{delay:.2f}s">'
            f'<div class="card-top"><span class="card-icon">{COMPONENT_ICONS[key]}</span>{esc(p["display_name"])}'
            f'<span class="dot{live}" style="margin-left:auto"></span></div>'
            f'<div class="sev-word">{esc(severity)}</div>'
            f'<div class="card-value"><b>{esc(p["predicted_value"])} {esc(p["unit"])}</b> - {esc(p["predicted_label"])}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;animation-delay:{delay + 0.25:.2f}s"></div></div>'
            f'<div class="card-foot"><span>{conf_text}</span><span>{esc(str(p["model"]).replace("_", " "))}</span></div>'
            f"</div>"
        )
    return f'<div class="card-grid">{"".join(cards)}</div>'


def probability_panels_html(predictions: Dict[str, Dict[str, object]]) -> str:
    panels: List[str] = []
    for i, key in enumerate(COMPONENTS):
        p = predictions[key]
        spec = TARGET_SPECS[key]
        probabilities: Optional[Dict[int, float]] = p["probabilities"]  # type: ignore[assignment]
        if not probabilities:
            body = '<div class="sec-sub">This model does not expose probabilities.</div>'
        else:
            predicted_name = f"{p['predicted_value']} {spec.unit} - {spec.value_labels.get(int(p['predicted_value']), '')}"
            rows = [
                (
                    f"{v} {spec.unit} - {spec.value_labels.get(int(v), '')}",
                    float(prob),
                    f"{float(prob) * 100:.1f}%",
                    SEVERITY_COLOURS.get(spec.severity.get(int(v), "Unknown"), SEVERITY_COLOURS["Unknown"]),
                )
                for v, prob in sorted(probabilities.items())
            ]
            body = bar_rows(rows, highlight=predicted_name, base_delay=i * 0.1)
        panels.append(
            f'<div class="panel" style="animation-delay:{i * 0.08:.2f}s">'
            f'<h4>{COMPONENT_ICONS[key]} {esc(p["display_name"])}</h4>{body}</div>'
        )
    return f'<div class="panel-grid">{"".join(panels)}</div>'


def truth_table_html(
    predictions: Dict[str, Dict[str, object]], true_labels: pd.Series, cycle_id: int
) -> Tuple[str, int]:
    test_sets = cached_test_sets()
    rows: List[str] = []
    matches = 0
    for i, key in enumerate(COMPONENTS):
        spec = TARGET_SPECS[key]
        actual = int(true_labels[spec.column])
        predicted = int(predictions[key]["predicted_value"])
        ok = predicted == actual
        matches += int(ok)
        in_test = cycle_id in test_sets.get(key, set())
        rows.append(
            f'<tr style="animation-delay:{i * 0.07:.2f}s">'
            f"<td>{COMPONENT_ICONS[key]} {esc(spec.display_name)}</td>"
            f"<td>{predicted} {esc(spec.unit)} <span class='sec-sub'>({esc(spec.value_labels.get(predicted, ''))})</span></td>"
            f"<td>{actual} {esc(spec.unit)} <span class='sec-sub'>({esc(spec.value_labels.get(actual, ''))})</span></td>"
            f"<td><span class='chip {'ok' if ok else 'bad'}'>{'✓ match' if ok else '✗ miss'}</span></td>"
            f"<td><span class='chip neutral'>{'held-out test' if in_test else 'training'}</span></td></tr>"
        )
    table = (
        '<table class="truth"><thead><tr><th>Component</th><th>Predicted</th><th>Recorded</th>'
        f'<th>Result</th><th>This cycle was</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )
    return table, matches


def _pick_random_cycle(n_cycles: int) -> None:
    """Button callback: choose and score a random cycle."""
    choice = random.randrange(n_cycles)
    st.session_state["cycle_input"] = choice
    st.session_state["active_cycle"] = choice


def _predict_selected_cycle() -> None:
    """Button callback: score the cycle currently in the number input."""
    st.session_state["active_cycle"] = int(st.session_state["cycle_input"])


def section_prediction() -> None:
    section_header("04", "Condition prediction", "Score one 60-second cycle with the four saved pipelines")
    if not require_models():
        return

    source = st.radio(
        "Input source",
        ["Select a cycle from the dataset", "Upload a raw-sensor CSV"],
        horizontal=True,
        key="input_source",
    )

    features: Optional[pd.DataFrame] = None
    cycle: Optional[Dict[str, np.ndarray]] = None
    true_labels: Optional[pd.Series] = None
    active_cycle: Optional[int] = None

    if source == "Select a cycle from the dataset":
        if not require_dataset():
            return
        profile = cached_profile()
        assert profile is not None
        n_cycles = len(profile)

        if "cycle_input" not in st.session_state:
            st.session_state["cycle_input"] = 0

        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom") if _supports_vertical_alignment() else st.columns([2, 1, 1])
        with c1:
            st.number_input(
                f"Cycle number (0 - {n_cycles - 1})", min_value=0, max_value=n_cycles - 1,
                step=1, key="cycle_input",
            )
        with c2:
            st.button("⚡ Predict this cycle", type="primary", **FULL_WIDTH,
                      on_click=_predict_selected_cycle)
        with c3:
            st.button("🎲 Random cycle", **FULL_WIDTH,
                      on_click=_pick_random_cycle, args=(n_cycles,))

        active_cycle = st.session_state.get("active_cycle")
        if active_cycle is not None:
            if int(active_cycle) != int(st.session_state["cycle_input"]):
                st.caption(
                    f"Showing results for cycle **{active_cycle}**. "
                    f"Press *Predict this cycle* to score cycle {st.session_state['cycle_input']}."
                )
            try:
                cycle = cached_cycle(int(active_cycle))
                with st.spinner("Extracting features..."):
                    features = cached_cycle_features(int(active_cycle))
                true_labels = profile.iloc[int(active_cycle)]
            except (FileNotFoundError, IndexError, ValueError) as error:
                st.error(f"Could not read cycle {active_cycle}: {error}")
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
                **FULL_WIDTH, hide_index=True,
            )
        uploaded = st.file_uploader("Raw-sensor CSV for one cycle", type=["csv"])
        if uploaded is not None:
            try:
                cycle = load_cycle_csv(uploaded)
                features = features_from_cycle(cycle)
            except ValueError as error:
                st.error(str(error))
                return
            except Exception as error:  # noqa: BLE001 - shown as readable text
                st.error(f"The file could not be processed: {error}")
                return

    if cycle is None or features is None:
        render_html(
            """
            <div class="panel" style="text-align:center;padding:2rem 1rem">
              <div style="font-size:2rem">🛠️</div>
              <div style="font-weight:700;margin:.3rem 0">No cycle scored yet</div>
              <div class="sec-sub">Pick a cycle and press <b>Predict</b>, try a <b>random cycle</b>, or upload a CSV.</div>
            </div>
            """
        )
        return

    # ---- Prediction -----------------------------------------------------
    try:
        result = predict_system_condition(features=features)
    except ModelNotTrainedError as error:
        st.error(str(error))
        return
    except ValueError as error:
        st.error(f"Prediction failed: {error}")
        return

    label = f"cycle {active_cycle}" if active_cycle is not None else "uploaded cycle"
    if st.session_state.get("last_toast") != label:
        st.session_state["last_toast"] = label
        st.toast(f"Scored {label}: {features.shape[1]} engineered features, 4 models.", icon="✅")

    # ---- Section 5: maintenance status ----------------------------------
    section_header("05", "Maintenance status", f"Result for {label}")
    status = str(result["system_status"])
    colour = SEVERITY_COLOURS.get(status, SEVERITY_COLOURS["Unknown"])
    mean_conf = result.get("mean_confidence")
    mean_text = f"{float(mean_conf) * 100:.1f}%" if mean_conf is not None else "n/a"
    live = " live" if status != "Normal" else ""
    render_html(
        f"""
        <div class="banner" style="--sev:{colour}">
          <span class="dot big{live}"></span>
          <div>
            <div class="banner-label">Overall system status</div>
            <div class="banner-value">{esc(status)}</div>
            <div class="banner-sub">Driven by {esc(result['driving_component'])} · worst-component rule</div>
          </div>
          <div class="banner-meta">Mean model confidence<b>{mean_text}</b></div>
        </div>
        """
    )
    st.caption(
        "The overall status is simply the worst individual component status - a system "
        "is only as healthy as its weakest part, and the dataset contains no "
        "maintenance-cost information that would justify a weighted health score."
    )

    predictions: Dict[str, Dict[str, object]] = result["components"]  # type: ignore[assignment]
    render_html(status_cards_html(predictions))

    with st.expander("Class probabilities", expanded=False):
        render_html(probability_panels_html(predictions))

    if true_labels is not None and active_cycle is not None:
        with st.expander("Compare with the dataset's recorded condition", expanded=True):
            table, matches = truth_table_html(predictions, true_labels, int(active_cycle))
            render_html(
                f'<div class="sec-sub" style="margin-bottom:.3rem">'
                f'<b style="font-size:1.1rem;color:inherit">{matches}/4</b> components match the recorded condition.</div>'
                + table
            )
            st.caption(
                "Most dataset cycles were used in training, so agreement here is not an "
                "unbiased accuracy estimate. The last column shows, per model, whether this "
                "cycle was held out. The honest numbers are on the Results tab."
            )

    # ---- Section 6: explainability --------------------------------------
    section_explainability(features, cycle)


def _supports_vertical_alignment() -> bool:
    """`st.columns(vertical_alignment=...)` exists from Streamlit 1.36."""
    return _streamlit_version() >= (1, 36)


def section_explainability(features: pd.DataFrame, cycle: Dict[str, np.ndarray]) -> None:
    section_header(
        "06", "Explainability",
        "An alarm that cannot be checked against the physics is not actionable",
    )

    component = st.selectbox(
        "Component", COMPONENTS,
        format_func=lambda key: f"{COMPONENT_ICONS[key]} {TARGET_SPECS[key].display_name}",
        key="explain_component",
    )
    bundle = cached_bundles()[component]
    metadata: Dict[str, object] = bundle["metadata"]  # type: ignore[assignment]

    try:
        explanation = explain_prediction(component, features, top_n=12)
        if explanation["importance"].notna().all() and len(explanation):
            top = float(explanation["importance"].max()) or 1.0
            rows = [
                (
                    str(r.feature_name),
                    float(r.importance) / top,
                    f"{float(r.value):.4g}",
                    "linear-gradient(90deg,#38BDF8,#818CF8)",
                )
                for r in explanation.itertuples()
            ]
            render_html(
                '<div class="panel"><h4>Top features the model relies on '
                '<span class="sec-sub">(bar = relative importance, right = value measured in this cycle)</span></h4>'
                + bar_rows(rows, mono=True)
                + "</div>"
            )
        else:
            st.dataframe(explanation, **FULL_WIDTH, hide_index=True)
    except (ModelNotTrainedError, ValueError) as error:
        st.warning(f"Explanation unavailable: {error}")

    st.caption(
        f"Model: {str(metadata['selected_model']).replace('_', ' ')} on the '{metadata['feature_set']}' "
        f"feature set, {metadata['n_selected_features']} features reaching the "
        f"classifier. Selection rule: {metadata['selection_rule']}."
    )

    st.subheader("Raw sensor traces for this cycle")
    chosen = st.multiselect(
        "Sensors to plot", list(SENSOR_NAMES), default=["PS1", "EPS1", "FS1", "TS1"],
        key="trace_sensors",
    )
    columns = st.columns(2)
    for i, sensor in enumerate(chosen):
        values = cycle.get(sensor)
        if values is None:
            continue
        spec = SENSOR_INDEX[sensor]
        frame = pd.DataFrame(
            {f"{sensor} [{spec.unit}]": values},
            index=np.linspace(0, 60, len(values)),
        )
        frame.index.name = "time within cycle [s]"
        with columns[i % 2]:
            st.line_chart(frame, height=190)


# ==========================================================================
# Results section (model performance)
# ==========================================================================
def section_results() -> None:
    section_header("R", "Model performance", "Optimistic vs realistic estimates, and what feature engineering bought")
    metrics = cached_metrics()
    cv: Optional[pd.DataFrame] = metrics["cv"]
    test: Optional[pd.DataFrame] = metrics["test"]
    if cv is None:
        st.warning("Run `python src/train.py` to populate this section.")
        return

    # ---- Headline: what feature engineering bought ---------------------
    st.subheader("What feature engineering bought")
    st.caption(
        "Logistic regression is the clearest test: a linear model cannot build its own "
        "features, so its jump from the 17 per-sensor means to the engineered set is the "
        "effect of the representation itself."
    )
    lr = cv[cv["model"] == "logistic_regression"].pivot_table(
        index="component", columns="feature_set", values="f1_macro"
    )
    gains: List[str] = []
    for i, key in enumerate(COMPONENTS):
        if key not in lr.index or "baseline" not in lr.columns:
            continue
        before = float(lr.loc[key, "baseline"])
        after_cols = [c for c in ("engineered", "selected") if c in lr.columns]
        after = float(lr.loc[key, after_cols].max()) if after_cols else before
        gains.append(
            f'<div class="gain" style="animation-delay:{i * 0.08:.2f}s">'
            f'<div class="g-title">{COMPONENT_ICONS[key]} {esc(TARGET_SPECS[key].display_name)} · logistic regression</div>'
            f'<div class="g-main">{before:.3f}<span class="arrow">➜</span>{after:.3f}</div>'
            f'<div class="g-delta">{after - before:+.3f} macro F1</div></div>'
        )
    if gains:
        render_html(f'<div class="stat-grid">{"".join(gains)}</div>')

    # ---- Honest performance --------------------------------------------
    if test is not None:
        st.subheader("Realistic vs optimistic performance")
        st.caption(
            "Neighbouring cycles were recorded seconds apart under the same settings, so a "
            "random split places near-duplicates on both sides. Temporal-block CV holds out "
            "whole blocks of 25 consecutive cycles and is the realistic estimate. "
            "Bars start at 0.85 to make the gaps visible."
        )
        series = [
            ("block_cv_f1_macro", "Block CV", "#22C55E"),
            ("cv_f1_macro", "Random CV", "#38BDF8"),
            ("f1_macro", "Test", "#818CF8"),
        ]
        legend_names = {
            "Block CV": "Block CV - 25-cycle temporal blocks held out (realistic)",
            "Random CV": "Random CV - stratified 5-fold on the training set",
            "Test": "Test - held-out 20 % random split (optimistic upper bound)",
        }
        legend = "".join(
            f'<span><i style="background:{c}"></i>{esc(legend_names[n])}</span>' for _, n, c in series
        )
        floor = 0.85
        panels: List[str] = []
        for i, row in enumerate(test.itertuples()):
            key = str(row.component)
            rows = [
                (name, (float(getattr(row, col)) - floor) / (1 - floor), f"{float(getattr(row, col)):.3f}", colour)
                for col, name, colour in series
            ]
            gap = float(row.cv_f1_macro) - float(row.block_cv_f1_macro)
            panels.append(
                f'<div class="panel" style="animation-delay:{i * 0.08:.2f}s">'
                f'<h4>{COMPONENT_ICONS.get(key, "")} {esc(row.display_name)} '
                f'<span class="sec-sub">· {esc(str(row.model).replace("_", " "))}, {esc(row.feature_set)} set · '
                f'random-split optimism {gap:+.3f}</span></h4>'
                + bar_rows(rows, base_delay=i * 0.1)
                + "</div>"
            )
        render_html(f'<div class="legend">{legend}</div><div class="panel-grid">{"".join(panels)}</div>')

        with st.expander("Held-out test set table (each model evaluated exactly once)"):
            st.dataframe(test.round(4), **FULL_WIDTH, hide_index=True)
    else:
        st.info("Run `python src/evaluate.py` to add test-set results.")

    # ---- Full grid -------------------------------------------------------
    st.subheader("Effect of feature engineering - best model per feature set")
    best = cv.groupby(["component", "feature_set"])["f1_macro"].max().unstack()
    ordered = [c for c in ("baseline", "engineered", "selected") if c in best.columns]
    st.dataframe(best[ordered].round(4), **FULL_WIDTH)

    with st.expander("Full cross-validation grid"):
        st.dataframe(cv.round(4), **FULL_WIDTH, hide_index=True)


# ==========================================================================
# Sidebar
# ==========================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🛠️ System check")
        dataset_ok = cached_dataset_summary() is not None
        models_ok = models_ready()
        metadata = cached_run_metadata()

        def row(label: str, ok: Optional[bool], text: str) -> str:
            colour = "#94A3B8" if ok is None else ("#22C55E" if ok else "#EF4444")
            return (
                f'<div class="side-row"><span>{esc(label)}</span>'
                f'<span><span class="dot" style="--sev:{colour};margin-right:.4rem"></span>{esc(text)}</span></div>'
            )

        rows = [
            row("Dataset", dataset_ok, "found" if dataset_ok else "missing"),
            row("Trained models", models_ok, "found" if models_ok else "missing"),
        ]
        if metadata:
            rows.append(row("Boosting backend", None,
                            "XGBoost" if metadata.get("xgboost_available") else "HistGradientBoosting"))
            rows.append(row("Random state", None, str(metadata.get("random_state"))))
            rows.append(row("Block size", None, f"{metadata.get('block_size')} cycles"))
        render_html("".join(rows))

        active = st.session_state.get("active_cycle")
        if active is not None:
            st.markdown(f"**Last scored cycle:** `{active}`")

        st.divider()
        st.caption(
            "The dashboard loads saved pipelines only. It never retrains, so the "
            "preprocessing and feature selection are exactly those used at "
            "training time."
        )


# ==========================================================================
# Layout
# ==========================================================================
def main() -> None:
    inject_css()
    render_hero()
    render_sidebar()

    tabs = st.tabs(
        ["🏠 Overview", "📊 Dataset", "🧪 Feature engineering", "🔮 Predict & status", "🏁 Results"]
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
