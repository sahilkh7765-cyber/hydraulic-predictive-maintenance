"""
visualization.py
================
Every figure written to outputs/figures/.

WHAT THIS MODULE DOES
    Provides one function per figure type: raw-signal and feature
    distributions, class balance, correlation structure, PCA projection, model
    comparison, confusion matrices, feature importance and ROC / PR curves.

WHY IT IS A SEPARATE MODULE
    Plotting code mixed into a training script makes the training logic hard to
    read and impossible to reuse from a notebook or from the dashboard. Keeping
    it separate means `train.py` stays a pipeline description, and the same
    functions render figures in `notebooks/` without duplicating code.

CONVENTIONS
    * The Agg backend is forced in utils, so the module is safe on a headless
      machine.
    * Every function returns the Path it wrote, so the caller can log it.
    * Figures are saved at 150 dpi with tight bounding boxes, which is legible
      both on screen and in a printed report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from utils import FIGURES_DIR, ensure_dir

# A single visual style for every figure in the report.
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "viridis"
DPI = 150


def _save(fig: plt.Figure, filename: str, subdir: Optional[str] = None) -> Path:
    """Write a figure to outputs/figures/ and close it.

    Closing matters: a long training run produces dozens of figures, and
    matplotlib keeps every unclosed figure in memory.
    """
    directory = ensure_dir(FIGURES_DIR / subdir) if subdir else ensure_dir(FIGURES_DIR)
    path = directory / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================================
# Raw data and label figures
# ==========================================================================
def plot_sensor_distributions(
    feature_df: pd.DataFrame,
    sensors: Sequence[str],
    filename: str = "sensor_distributions.png",
) -> Path:
    """Histogram of each sensor's cycle-mean value across all cycles.

    WHY: shows at a glance which channels are multi-modal. A clearly bimodal
    cycle-mean (as in the cooling channels) means the fault levels separate
    almost linearly in that sensor, which predicts an easy classification task.

    Input : engineered feature table (must contain `<sensor>_mean` columns).
    Output: path to the saved figure.
    """
    available = [s for s in sensors if f"{s}_mean" in feature_df.columns]
    n_cols = 4
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, sensor in zip(axes, available):
        sns.histplot(feature_df[f"{sensor}_mean"], bins=40, ax=ax, color="#2a6f97")
        ax.set_title(f"{sensor} (cycle mean)", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("cycles")
    for ax in axes[len(available) :]:
        ax.axis("off")

    fig.suptitle("Distribution of per-cycle mean value, by sensor", fontsize=13)
    fig.tight_layout()
    return _save(fig, filename)


def plot_raw_cycle_examples(
    cycles: Dict[str, np.ndarray],
    filename: str = "raw_cycle_examples.png",
) -> Path:
    """Plot the raw waveform of several sensors for one operating cycle.

    WHY: this is the figure that motivates the whole project. It shows that a
    cycle is a structured 60-second programme rather than a static reading, so
    a single mean value throws away the shape that the fault signatures live in.

    Input : dict sensor -> 1-D array of that sensor's samples for one cycle.
    Output: path to the saved figure.
    """
    n = len(cycles)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 * n), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (sensor, values) in zip(axes, cycles.items()):
        time_axis = np.linspace(0, 60, len(values))
        ax.plot(time_axis, values, linewidth=0.7, color="#1b4965")
        ax.set_ylabel(sensor, fontsize=9)
        ax.margins(x=0)
    axes[-1].set_xlabel("time within cycle [s]")
    fig.suptitle("One operating cycle (60 s) as seen by different sensors", fontsize=13)
    fig.tight_layout()
    return _save(fig, filename)


def plot_class_distributions(
    profile: pd.DataFrame,
    target_specs: Dict[str, object],
    filename: str = "class_distributions.png",
) -> Path:
    """Bar chart of class counts for the four component targets.

    WHY: class imbalance decides which metric is trustworthy. Showing that the
    valve target is 51 % one class is what justifies reporting macro-F1 rather
    than accuracy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, (key, spec) in zip(axes, target_specs.items()):
        counts = profile[spec.column].value_counts().sort_index()
        labels = [
            f"{value}\n{spec.value_labels.get(int(value), '')}" for value in counts.index
        ]
        sns.barplot(x=labels, y=counts.to_numpy(), ax=ax, hue=labels, legend=False, palette=PALETTE)
        ax.set_title(f"{spec.display_name} [{spec.unit}]", fontsize=11)
        ax.set_ylabel("cycles")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Class distribution of the four component-condition targets", fontsize=13)
    fig.tight_layout()
    return _save(fig, filename)


# ==========================================================================
# Feature-space figures
# ==========================================================================
def plot_correlation_heatmap(
    feature_df: pd.DataFrame,
    columns: Sequence[str],
    max_features: int = 45,
    filename: str = "correlation_heatmap.png",
) -> Path:
    """Correlation heatmap of a readable subset of engineered features.

    WHY A SUBSET: a 500x500 heatmap is an unreadable smear. Showing the highest-
    variance subset communicates the point - that engineered features are
    strongly collinear by construction - which is exactly what motivates the
    correlation-filtering stage.
    """
    subset = list(columns)[:max_features]
    corr = feature_df[subset].corr()
    fig, ax = plt.subplots(figsize=(0.30 * len(subset) + 4, 0.28 * len(subset) + 3))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"shrink": 0.6, "label": "Pearson r"})
    ax.set_title(f"Correlation structure of {len(subset)} engineered features", fontsize=12)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return _save(fig, filename)


def plot_feature_distributions(
    feature_df: pd.DataFrame,
    features: Sequence[str],
    target: Optional[pd.Series] = None,
    target_name: str = "",
    filename: str = "feature_distributions.png",
) -> Path:
    """Distribution of selected engineered features, optionally split by class.

    WHY: a feature whose per-class distributions barely overlap is doing real
    discriminative work. This figure is the visual justification for the
    features that the importance ranking puts on top.
    """
    features = list(features)[:8]
    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        if target is not None:
            for value in np.unique(target):
                sns.kdeplot(
                    feature_df.loc[target == value, feature],
                    ax=ax, fill=True, alpha=0.35, label=str(value), warn_singular=False,
                )
            ax.legend(fontsize=7, title=target_name, title_fontsize=7)
        else:
            sns.histplot(feature_df[feature], bins=40, ax=ax, color="#2a6f97")
        ax.set_title(feature, fontsize=9)
        ax.set_xlabel("")
    for ax in axes[len(features) :]:
        ax.axis("off")
    suffix = f" by {target_name}" if target_name else ""
    fig.suptitle(f"Engineered feature distributions{suffix}", fontsize=13)
    fig.tight_layout()
    return _save(fig, filename)


def plot_pca(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    filename: str,
    class_labels: Optional[Dict[int, str]] = None,
) -> Path:
    """Two-component PCA scatter of the engineered feature space.

    WHY PCA IS USED FOR VISUALISATION ONLY: principal components are linear
    mixtures of hundreds of features, so a model built on them cannot be
    explained to a maintenance engineer - which defeats the purpose of a
    condition-monitoring system. As a 2-D picture, though, PCA answers a useful
    question: are the condition classes separable at all in the engineered
    space? Well-separated clusters predict high accuracy before any model runs.

    IMPORTANT: X must already be scaled. PCA maximises variance, so unscaled
    features measured in watts would dominate features measured in bar.
    """
    pca = PCA(n_components=2, random_state=0)
    projected = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for value in np.unique(y):
        mask = y == value
        label = class_labels.get(int(value), str(value)) if class_labels else str(value)
        ax.scatter(projected[mask, 0], projected[mask, 1], s=12, alpha=0.65, label=label)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f} % variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f} % variance)")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, markerscale=1.6)
    fig.tight_layout()
    return _save(fig, filename, subdir="pca")


def plot_pca_explained_variance(
    X: np.ndarray, filename: str = "pca_explained_variance.png", n_components: int = 30
) -> Path:
    """Cumulative explained variance of the engineered feature space.

    WHY: quantifies how much redundancy the engineered features contain. If a
    handful of components explain most of the variance, the effective
    dimensionality is far below the nominal feature count - the same message
    the correlation filter acts on.
    """
    n_components = int(min(n_components, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_components, random_state=0).fit(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(np.arange(1, n_components + 1), cumulative * 100, marker="o", markersize=4)
    ax.axhline(95, color="crimson", linestyle="--", linewidth=1, label="95 % variance")
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance [%]")
    ax.set_title("How much redundancy is in the engineered feature space?", fontsize=12)
    ax.legend()
    fig.tight_layout()
    return _save(fig, filename, subdir="pca")


# ==========================================================================
# Model result figures
# ==========================================================================
def plot_model_comparison(
    results: pd.DataFrame,
    metric: str = "f1_macro",
    filename: str = "model_comparison.png",
) -> Path:
    """Grouped bar chart: model x feature set, one panel per component.

    WHY THIS IS THE HEADLINE FIGURE: the research question is whether engineered
    features improve condition monitoring. This figure answers it directly by
    putting baseline, engineered and selected bars side by side for every model
    and every component.
    """
    components = list(dict.fromkeys(results["component"]))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, component in zip(axes, components):
        subset = results[results["component"] == component]
        sns.barplot(
            data=subset, x="model", y=metric, hue="feature_set", ax=ax, palette="mako"
        )
        ax.set_title(component.capitalize(), fontsize=12)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " "))
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(fontsize=8, title="feature set", title_fontsize=8)
    fig.suptitle(
        f"Model comparison across feature sets ({metric.replace('_', ' ')}, cross-validated)",
        fontsize=13,
    )
    fig.tight_layout()
    return _save(fig, filename)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    display_labels: Sequence[str],
    title: str,
    filename: str,
) -> Path:
    """Confusion matrix with raw counts.

    WHY IT MATTERS MORE THAN ACCURACY HERE: the targets are ordered degradation
    levels. Confusing "optimal" with "close to total failure" is an operational
    disaster; confusing two adjacent degradation levels is a minor error. Only
    the confusion matrix shows which of the two is happening.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=list(display_labels), ax=ax,
        cmap="Blues", colorbar=False, values_format="d",
    )
    ax.set_title(title, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    return _save(fig, filename, subdir="confusion_matrices")


def plot_feature_importance(
    names: Sequence[str],
    importances: Sequence[float],
    title: str,
    filename: str,
    top_n: int = 20,
) -> Path:
    """Horizontal bar chart of the top-n most important features."""
    order = np.argsort(importances)[::-1][:top_n]
    selected_names = [names[i] for i in order]
    selected_values = [importances[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 0.38 * len(selected_names) + 2))
    sns.barplot(x=selected_values, y=selected_names, ax=ax, hue=selected_names,
                legend=False, palette=PALETTE)
    ax.set_xlabel("importance")
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _save(fig, filename, subdir="feature_importance")


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: Sequence[int],
    class_labels: Dict[int, str],
    title: str,
    filename: str,
) -> Optional[Path]:
    """One-vs-rest ROC curves for a multiclass target.

    WHY ONE-VS-REST: ROC is defined for binary problems. With more than two
    ordered condition levels, the standard extension is to treat each class as
    "this class vs everything else" and plot one curve per class.

    Returns None when the score matrix is unusable (for example a model without
    `predict_proba`, or a fold that contains a single class).
    """
    classes = list(classes)
    if y_score is None or len(classes) < 2:
        return None
    binarized = label_binarize(y_true, classes=classes)
    if binarized.shape[1] == 1:  # degenerate binary case
        binarized = np.hstack([1 - binarized, binarized])

    fig, ax = plt.subplots(figsize=(7, 6))
    for index, value in enumerate(classes):
        if binarized[:, index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binarized[:, index], y_score[:, index])
        ax.plot(fpr, tpr, linewidth=1.6,
                label=f"{class_labels.get(int(value), value)} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, filename, subdir="roc_curves")


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: Sequence[int],
    class_labels: Dict[int, str],
    title: str,
    filename: str,
) -> Optional[Path]:
    """One-vs-rest precision-recall curves.

    WHY IN ADDITION TO ROC: ROC can look flattering on an imbalanced target
    because the true-negative pool is large. Precision-recall focuses on the
    minority class, which here is the rare severe-degradation state - exactly
    the state a maintenance system must not miss.
    """
    classes = list(classes)
    if y_score is None or len(classes) < 2:
        return None
    binarized = label_binarize(y_true, classes=classes)
    if binarized.shape[1] == 1:
        binarized = np.hstack([1 - binarized, binarized])

    fig, ax = plt.subplots(figsize=(7, 6))
    for index, value in enumerate(classes):
        if binarized[:, index].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(binarized[:, index], y_score[:, index])
        ax.plot(recall, precision, linewidth=1.6,
                label=f"{class_labels.get(int(value), value)} (AP area {auc(recall, precision):.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return _save(fig, filename, subdir="pr_curves")


def plot_feature_block_contribution(
    selected_features: Sequence[str],
    block_map: Dict[str, List[str]],
    title: str,
    filename: str,
) -> Path:
    """How many features each engineering block contributed to the final set.

    WHY: this is the figure that answers "which kind of feature engineering was
    worth doing?". If the domain block contributes disproportionately relative
    to its size, hydraulic knowledge added value beyond generic statistics.
    """
    lookup = {name: block for block, names in block_map.items() for name in names}
    counts: Dict[str, int] = {block: 0 for block in block_map}
    for name in selected_features:
        block = lookup.get(name)
        if block:
            counts[block] += 1
    available = {block: len(names) for block, names in block_map.items()}

    blocks = list(block_map.keys())
    selected_counts = [counts[b] for b in blocks]
    rates = [
        100.0 * counts[b] / available[b] if available[b] else 0.0 for b in blocks
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    sns.barplot(x=blocks, y=selected_counts, ax=axes[0], hue=blocks, legend=False, palette=PALETTE)
    axes[0].set_title("Features selected, by engineering block")
    axes[0].set_ylabel("selected features")
    sns.barplot(x=blocks, y=rates, ax=axes[1], hue=blocks, legend=False, palette=PALETTE)
    axes[1].set_title("Selection rate (selected / generated)")
    axes[1].set_ylabel("%")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, filename)


def plot_feature_count_funnel(
    stage_counts: Dict[str, int], title: str, filename: str
) -> Path:
    """Funnel of feature counts from generation to the final selected set.

    WHY: the brief asks for the count at every stage. A funnel makes the
    reduction - and how aggressive each stage was - immediately readable.
    """
    stages = list(stage_counts.keys())
    counts = [stage_counts[s] for s in stages]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(stages, counts, color=sns.color_palette("mako", len(stages)))
    ax.bar_label(bars, fmt="%d", fontsize=9)
    ax.set_ylabel("number of features")
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", labelsize=9, rotation=15)
    fig.tight_layout()
    return _save(fig, filename)
