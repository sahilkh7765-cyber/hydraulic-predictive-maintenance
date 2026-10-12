"""
core.py
=======
Artefact loading and warm-up for the API.

WHY A BACKGROUND WARM-UP
    Loading four pipelines, the 2,205 x 559 engineered feature table and
    scoring every held-out test set takes several seconds. Doing it inside the
    lifespan hook would make the server refuse connections until it finished;
    doing it per request would repeat it. Instead the server starts at once,
    `/api/health` reports progress, and every other endpoint answers
    503 `warming_up` until the artefacts are ready. Everything is loaded exactly
    once per process.

NOTHING HERE TRAINS OR RE-FITS ANYTHING
    Models come from `prediction.load_component_bundle`; the evaluation numbers
    are the saved pipelines' own predictions on the saved test indices.
"""

from __future__ import annotations

import threading
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import data_loader
import prediction
import utils
from data_loader import SENSOR_NAMES, TARGET_SPECS, TargetSpec
from utils import COMPONENTS, load_json

from .errors import ApiError

CYCLE_CACHE_SIZE = 32
RELIABILITY_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001)
METRIC_FILES = (
    "model_results.csv",
    "test_results.csv",
    "feature_experiment_effect.csv",
    "feature_importance.csv",
)


@dataclass(frozen=True)
class Paths:
    """Artefact locations, resolved from the src modules at warm-up time.

    Reading them from the modules (rather than hard-coding) lets the tests
    point the API at an empty directory to exercise the missing-model path.
    """

    models: Path
    raw: Path
    metrics: Path
    features: Path
    figures: Path
    reports: Path

    @classmethod
    def from_modules(cls) -> "Paths":
        return cls(
            models=Path(prediction.MODELS_DIR),
            raw=Path(data_loader.RAW_DATA_DIR),
            metrics=Path(utils.METRICS_DIR),
            features=Path(utils.FEATURES_DIR),
            figures=Path(utils.FIGURES_DIR),
            reports=Path(utils.REPORTS_DIR),
        )


@dataclass
class ComponentArtefacts:
    """One component's pipeline plus everything precomputed from it."""

    key: str
    spec: TargetSpec
    model: object
    encoder: object
    metadata: Dict[str, object]
    classes: List[int]
    input_columns: List[str]
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_set: frozenset
    # Filled once the feature table is available.
    test_y: Optional[np.ndarray] = None
    test_proba: Optional[np.ndarray] = None
    train_embedding: Optional[np.ndarray] = None
    reliability: List[Dict[str, float]] = field(default_factory=list)

    @property
    def test_pred(self) -> np.ndarray:
        assert self.test_proba is not None
        return self.test_proba.argmax(axis=1)

    def severity(self, value: int) -> str:
        return self.spec.severity.get(int(value), "Unknown")

    def label(self, value: int) -> str:
        return self.spec.value_labels.get(int(value), "unknown")


class AppState:
    """Process-wide artefact store with an explicit warm-up lifecycle."""

    STEPS = (
        "Checking artefacts",
        "Reading metrics and run metadata",
        "Loading the four pipelines",
        "Loading the engineered feature table",
        "Scoring the held-out test sets",
        "Indexing training cycles for neighbour search",
    )

    def __init__(self) -> None:
        self.paths = Paths.from_modules()
        self.status: str = "warming"
        self.message: Optional[str] = None
        self.steps: Dict[str, bool] = {name: False for name in self.STEPS}
        self.checks: Dict[str, object] = {
            "models": {c: False for c in COMPONENTS},
            "feature_table": False,
            "raw_data": False,
            "metrics": False,
        }
        self.run_meta: Optional[Dict[str, object]] = None
        self.metrics: Dict[str, pd.DataFrame] = {}
        self.components: Dict[str, ComponentArtefacts] = {}
        self.table: Optional[pd.DataFrame] = None
        self.profile: Optional[pd.DataFrame] = None
        self.block_of: Dict[str, str] = {}
        self._cycle_cache: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._compute_lock = threading.Lock()
        self.noise_cache: Dict[str, object] = {}
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._warm_up, name="warm-up", daemon=True)
        self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _done(self, step: str) -> None:
        self.steps[step] = True

    def _warm_up(self) -> None:
        try:
            self._check_files()
            self._done(self.STEPS[0])

            self._load_metadata()
            self._done(self.STEPS[1])

            models_ok = all(self.checks["models"].values())  # type: ignore[union-attr]
            if not models_ok:
                missing = [c for c, ok in self.checks["models"].items() if not ok]  # type: ignore[union-attr]
                self.status = "unavailable"
                self.message = (
                    f"No trained model for: {', '.join(missing)}. "
                    "Run `python src/train.py` then `python src/evaluate.py`."
                )
                return

            self._load_models()
            self._done(self.STEPS[2])

            if self.checks["feature_table"]:
                self.table = pd.read_csv(self.paths.features / "engineered_features.csv")
            self._done(self.STEPS[3])

            if self.table is not None:
                self._score_test_sets()
            self._done(self.STEPS[4])

            if self.table is not None:
                self._index_training_cycles()
            self._done(self.STEPS[5])

            self.status = "ready"
        except Exception as error:  # noqa: BLE001 - surfaced through /api/health
            traceback.print_exc()
            self.status = "unavailable"
            self.message = f"Warm-up failed: {error}"

    def _check_files(self) -> None:
        p = self.paths
        self.checks["models"] = {
            c: all(
                (p.models / c / name).exists()
                for name in ("model.joblib", "label_encoder.joblib", "metadata.json", "split_indices.npz")
            )
            for c in COMPONENTS
        }
        self.checks["feature_table"] = (p.features / "engineered_features.csv").exists()
        self.checks["raw_data"] = (p.raw / "profile.txt").exists() and all(
            (p.raw / f"{s}.txt").exists() for s in SENSOR_NAMES
        )
        self.checks["metrics"] = all((p.metrics / f).exists() for f in METRIC_FILES)

    def _load_metadata(self) -> None:
        p = self.paths
        run_meta_path = p.models / "run_metadata.json"
        self.run_meta = load_json(run_meta_path) if run_meta_path.exists() else None
        if self.run_meta:
            names: Dict[str, List[str]] = self.run_meta.get("feature_block_names", {})  # type: ignore[assignment]
            self.block_of = {f: block for block, feats in names.items() for f in feats}
        for name in METRIC_FILES:
            path = p.metrics / name
            if path.exists():
                self.metrics[name.removesuffix(".csv")] = pd.read_csv(path)
        if self.checks["raw_data"]:
            self.profile = data_loader.load_profile(p.raw)

    def _load_models(self) -> None:
        for c in COMPONENTS:
            bundle = prediction.load_component_bundle(c)
            with np.load(self.paths.models / c / "split_indices.npz") as split:
                train_idx = split["train_idx"].astype(int).copy()
                test_idx = split["test_idx"].astype(int).copy()
            metadata: Dict[str, object] = bundle["metadata"]  # type: ignore[assignment]
            self.components[c] = ComponentArtefacts(
                key=c,
                spec=TARGET_SPECS[c],
                model=bundle["model"],
                encoder=bundle["encoder"],
                metadata=metadata,
                classes=[int(v) for v in bundle["encoder"].classes_],  # type: ignore[union-attr]
                input_columns=list(metadata["input_feature_columns"]),  # type: ignore[arg-type]
                train_idx=train_idx,
                test_idx=test_idx,
                test_set=frozenset(int(i) for i in test_idx),
            )

    def _score_test_sets(self) -> None:
        for art in self.components.values():
            X, y = self.xy(art, art.test_idx)
            art.test_y = y
            art.test_proba = art.model.predict_proba(X)  # type: ignore[union-attr]
            conf = art.test_proba.max(axis=1)
            correct = art.test_pred == y
            bins = []
            for lo, hi in zip(RELIABILITY_EDGES[:-1], RELIABILITY_EDGES[1:]):
                mask = (conf >= lo) & (conf < hi)
                bins.append(
                    {
                        "lo": lo,
                        "hi": min(hi, 1.0),
                        "n": int(mask.sum()),
                        "accuracy": float(correct[mask].mean()) if mask.any() else None,
                    }
                )
            art.reliability = bins

    def _index_training_cycles(self) -> None:
        for art in self.components.values():
            X, _ = self.xy(art, art.train_idx)
            art.train_embedding = art.model[:-1].transform(X)  # type: ignore[index]

    # ------------------------------------------------------------------
    # Guards used by the routes
    # ------------------------------------------------------------------
    def require_ready(self) -> None:
        if self.status == "warming":
            raise ApiError(
                503,
                "warming_up",
                "The API is still loading the models. Retry in a few seconds.",
                headers={"Retry-After": "2"},
            )
        if self.status == "unavailable":
            raise ApiError(503, "models_unavailable", self.message or "Models are unavailable.")

    def require_metrics(self) -> None:
        self.require_ready()
        if not self.checks["metrics"]:
            raise ApiError(
                503,
                "metrics_unavailable",
                "outputs/metrics is incomplete. Run `python src/train.py` and `python src/evaluate.py`.",
            )

    def require_table(self) -> pd.DataFrame:
        self.require_ready()
        if self.table is None:
            raise ApiError(
                503,
                "feature_table_unavailable",
                "outputs/features/engineered_features.csv is missing. Run `python src/train.py`.",
            )
        return self.table

    def require_raw(self) -> None:
        self.require_ready()
        if not self.checks["raw_data"]:
            raise ApiError(
                503,
                "raw_data_unavailable",
                "The raw dataset was not found in data/raw/. Extract the UCI ZIP there.",
            )

    def component(self, key: str) -> ComponentArtefacts:
        self.require_ready()
        return self.components[key]

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------
    @property
    def n_cycles(self) -> int:
        if self.profile is not None:
            return len(self.profile)
        if self.table is not None:
            return len(self.table)
        return 0

    def labels_frame(self) -> pd.DataFrame:
        """Cycle-wise condition labels: profile.txt if present, else the feature table."""
        if self.profile is not None:
            return self.profile
        if self.table is not None:
            return self.table
        raise ApiError(503, "raw_data_unavailable", "No label source available.")

    def check_cycle(self, cycle_id: int) -> None:
        n = self.n_cycles
        if not 0 <= cycle_id < n:
            raise ApiError(404, "cycle_not_found", f"Cycle {cycle_id} does not exist (valid: 0 .. {n - 1}).")

    def xy(self, art: ComponentArtefacts, idx: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """Model-ready X and label-encoded y for the given cycle indices."""
        assert self.table is not None
        X = self.table.loc[idx, art.input_columns].to_numpy(dtype=np.float64)
        y = art.encoder.transform(self.table.loc[idx, art.spec.column].to_numpy())  # type: ignore[union-attr]
        return X, np.asarray(y)

    def raw_cycle(self, cycle_id: int) -> Dict[str, np.ndarray]:
        """One cycle's raw samples via `load_cycle_row`, with a small LRU cache."""
        with self._cache_lock:
            if cycle_id in self._cycle_cache:
                self._cycle_cache.move_to_end(cycle_id)
                return self._cycle_cache[cycle_id]
        cycle = prediction.load_cycle_row(cycle_id, raw_dir=self.paths.raw)
        with self._cache_lock:
            self._cycle_cache[cycle_id] = cycle
            while len(self._cycle_cache) > CYCLE_CACHE_SIZE:
                self._cycle_cache.popitem(last=False)
        return cycle

    def cached(self, key: str, compute):  # type: ignore[no-untyped-def]
        """Compute-once cache for expensive, deterministic results."""
        with self._compute_lock:
            if key not in self.noise_cache:
                self.noise_cache[key] = compute()
            return self.noise_cache[key]
