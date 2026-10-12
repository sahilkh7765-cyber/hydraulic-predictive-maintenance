"""
utils.py
========
Shared low-level helpers for the hydraulic predictive-maintenance project.

WHAT THIS MODULE DOES
    Holds everything that every other module needs: project paths, the global
    random seed, numerically safe arithmetic, small IO wrappers and console
    formatting helpers.

WHY IT IS NEEDED
    Feature engineering on sensor data constantly divides one physical quantity
    by another (pressure / flow, power / flow, cooling power / temperature
    difference). Any of those denominators can legitimately be zero in a real
    rig, and an unguarded division silently injects `inf` / `NaN` into the
    feature matrix, which later crashes scikit-learn. Centralising the guard
    here means every domain feature is protected by the same tested code path.

INPUT / OUTPUT
    This module has no side effects on import other than defining paths.
    Directories are created lazily through `ensure_dir()`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# A single seed is used for every stochastic step in the project: the
# train/test split, cross-validation shuffling, Random Forest / XGBoost tree
# construction and permutation importance. Re-running the pipeline on the same
# machine therefore reproduces the exact numbers reported in outputs/.
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
# All paths are derived from this file's location so the project can be moved
# or cloned anywhere without editing a single hard-coded string.
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
FEATURES_DIR: Path = OUTPUTS_DIR / "features"

# Component keys used consistently across models/, outputs/ and the dashboard.
COMPONENTS: tuple[str, ...] = ("cooler", "valve", "pump", "accumulator")


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Input : a Path pointing at a directory.
    Output: the same Path, guaranteed to exist on disk.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs() -> None:
    """Create the full output tree in one call.

    Called at the start of train.py / evaluate.py so that a fresh clone of the
    repository (which ships empty output folders) never fails on a missing
    directory halfway through a long run.
    """
    for directory in (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        FEATURES_DIR,
    ):
        ensure_dir(directory)
    for component in COMPONENTS:
        ensure_dir(MODELS_DIR / component)


# --------------------------------------------------------------------------
# Numerically safe arithmetic
# --------------------------------------------------------------------------
def safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    fill_value: float = 0.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """Element-wise division that can never produce inf or NaN.

    WHY: hydraulic ratio features (efficiency = hydraulic power / electrical
    power, flow ratio FS1/FS2, cooling power per Kelvin of temperature spread)
    are physically meaningful but their denominators approach zero whenever the
    rig is between load steps. Returning `fill_value` there is safer than
    propagating an infinity that would later be dropped together with the whole
    feature column.

    Input : two broadcastable arrays/scalars, a fill value for degenerate cases.
    Output: float64 array of the same broadcast shape, finite everywhere.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast(numerator, denominator).shape, float(fill_value)),
        where=np.abs(denominator) > eps,
    )
    return np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)


def sanitize_matrix(matrix: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Replace inf / NaN in a feature matrix with a finite constant.

    Used as the last line of defence before a feature table is written to disk,
    so that `outputs/features/engineered_features.csv` is always loadable.
    """
    return np.nan_to_num(
        np.asarray(matrix, dtype=np.float64),
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def save_json(obj: Dict[str, Any], path: Path) -> Path:
    """Write a dictionary to disk as indented JSON (used for model metadata)."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file written by `save_json`."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    """Make numpy scalars / arrays JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write a DataFrame to CSV, creating the parent directory if needed."""
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path


# --------------------------------------------------------------------------
# Console formatting
# --------------------------------------------------------------------------
def print_header(title: str, char: str = "=", width: int = 78) -> None:
    """Print a visually separated section header.

    The console transcript of `train.py` doubles as evidence for the report, so
    the required summaries (DATASET SUMMARY, FEATURE ENGINEERING SUMMARY, model
    comparison table) need to be easy to locate in a long log.
    """
    print()
    print(char * width)
    print(title)
    print(char * width)


def print_kv(key: str, value: Any, width: int = 38) -> None:
    """Print an aligned `key: value` line for the summary blocks."""
    print(f"  {key:<{width}} {value}")


class Timer:
    """Context manager that measures wall-clock seconds.

    Training time is one of the required columns of
    `outputs/metrics/model_results.csv`, and it must come from real execution
    rather than an estimate, so every model fit is wrapped in this timer.

    Usage:
        with Timer() as t:
            model.fit(X, y)
        elapsed_seconds = t.elapsed
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed = time.perf_counter() - self.start


def format_int(value: int) -> str:
    """Thousands-separated integer, for readable console summaries."""
    return f"{value:,}"


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """De-duplicate an iterable while keeping first-seen order.

    Feature-name lists are assembled from several generators; ordering must stay
    stable so that a saved column list still matches the matrix it was fitted on.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


# Keep matplotlib from trying to open a window on a headless machine.
os.environ.setdefault("MPLBACKEND", "Agg")
