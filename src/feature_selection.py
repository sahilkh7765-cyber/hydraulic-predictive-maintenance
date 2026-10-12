"""
feature_selection.py
====================
Three-stage feature selection, implemented as scikit-learn transformers.

WHY TRANSFORMERS AND NOT SCRIPTS
    The obvious way to select features is to compute correlations and mutual
    information on the whole feature table and then keep the winners. That is
    also the single most common source of data leakage in student projects:
    the ranking has already seen the test labels, so the test score measures a
    selection that was tuned on the test set.

    Wrapping each stage in a `BaseEstimator` / `TransformerMixin` puts it inside
    the Pipeline. scikit-learn then calls `fit` on the training fold only and
    `transform` on the validation fold, so the selection is re-derived from
    scratch for every fold and the final test set is never consulted.

STAGE ORDER AND RATIONALE
    1. CorrelationFilter  - unsupervised, removes redundancy. Engineered
       features are highly collinear by construction (a sensor's mean, median
       and 50th percentile are nearly the same number). Removing redundancy
       first means the supervised stages rank a set of genuinely distinct
       candidates rather than twenty copies of one signal.
    2. MutualInformationSelector - supervised, univariate, model-free. Captures
       non-linear dependence between a single feature and the target, which a
       Pearson correlation with the label would miss.
    3. TreeImportanceSelector - supervised, multivariate. A Random Forest scores
       features in the context of all others, so it can keep a feature that is
       only useful in combination and drop one whose univariate signal is
       duplicated elsewhere.

    Cheap and unsupervised first, expensive and multivariate last.

INPUT  : numeric feature matrix (already imputed and scaled by the pipeline)
OUTPUT : the same matrix restricted to the selected columns
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted

from utils import RANDOM_STATE


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of every pair of features whose |Pearson r| exceeds a threshold.

    WHAT: computes the correlation matrix of the training fold and, scanning
    columns left to right, discards any feature that is too strongly correlated
    with a feature already kept.

    WHY: `PS1_mean`, `PS1_median` and `PS1_q50` measure almost the same thing.
    Keeping all three inflates the feature count, splits tree importance across
    interchangeable copies (making every one of them look unimportant) and
    makes Logistic Regression coefficients unstable. Removing redundancy costs
    almost no information because the retained twin carries the same signal.

    Parameters
    ----------
    threshold : absolute correlation above which the later feature is dropped.
                0.95 is strict enough to remove near-duplicates while keeping
                features that merely share a trend.

    Attributes after fit
    --------------------
    support_        : boolean mask of retained columns
    selected_index_ : integer indices of retained columns
    n_features_in_  : number of input columns
    """

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "CorrelationFilter":
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # np.corrcoef emits warnings and NaNs for zero-variance columns. Those
        # were already dropped in cleaning, but a training fold can still
        # contain a column that is constant *within that fold*, so the guard is
        # kept and such columns are treated as uncorrelated.
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)
        abs_corr = np.abs(corr)

        keep: List[int] = []
        for column in range(self.n_features_in_):
            if not keep:
                keep.append(column)
                continue
            # Compare only against features already accepted, so the decision
            # is order-deterministic and never removes both members of a pair.
            if abs_corr[column, keep].max() <= self.threshold:
                keep.append(column)

        self.selected_index_ = np.asarray(keep, dtype=int)
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_index_] = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "support_")
        return np.asarray(X, dtype=np.float64)[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return the retained-column mask (or indices), sklearn-style."""
        check_is_fitted(self, "support_")
        return self.selected_index_ if indices else self.support_


class MutualInformationSelector(BaseEstimator, TransformerMixin):
    """Keep the top fraction of features by mutual information with the target.

    WHAT: estimates I(feature; target) for every column on the training fold and
    keeps the highest-scoring `percentile` percent.

    WHY MUTUAL INFORMATION RATHER THAN F-TEST OR CORRELATION: mutual information
    measures any statistical dependence, not just a monotonic one. Several
    hydraulic relationships here are non-monotonic - for instance a pressure
    ratio can move in one direction for mild valve lag and back for severe lag -
    and a linear filter would score those near zero and discard them.

    Parameters
    ----------
    percentile     : percentage of features to retain (0-100).
    min_features   : never reduce below this count, so a small input set is not
                     emptied by an aggressive percentile.
    random_state   : mutual_info_classif uses a randomised nearest-neighbour
                     estimator; fixing the seed keeps the run reproducible.
    """

    def __init__(
        self,
        percentile: float = 50.0,
        min_features: int = 10,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.percentile = percentile
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MutualInformationSelector":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.n_features_in_ = X.shape[1]

        self.scores_ = mutual_info_classif(X, y, random_state=self.random_state)
        n_keep = int(np.ceil(self.n_features_in_ * self.percentile / 100.0))
        n_keep = int(np.clip(n_keep, min(self.min_features, self.n_features_in_), self.n_features_in_))

        # argsort on the negated scores gives descending order; the retained
        # indices are then re-sorted so column order stays stable.
        ranked = np.argsort(-self.scores_, kind="stable")[:n_keep]
        self.selected_index_ = np.sort(ranked)
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_index_] = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "support_")
        return np.asarray(X, dtype=np.float64)[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.selected_index_ if indices else self.support_


class TreeImportanceSelector(BaseEstimator, TransformerMixin):
    """Keep the top-k features by Random Forest impurity importance.

    WHAT: fits a compact Random Forest on the training fold and retains the k
    features with the largest mean decrease in impurity.

    WHY A TREE: unlike the two earlier stages this one is multivariate - each
    feature is scored in the presence of the others. A feature that is only
    informative jointly with another (say, a pressure differential that matters
    only at high flow) survives here but would score poorly univariately.

    WHY A SEPARATE FOREST FROM THE FINAL MODEL: the selector must be fitted on
    the same training fold as the model that follows it. Instantiating its own
    small forest (fewer, shallower trees) keeps the cost low while giving a
    stable ranking, and keeps selection independent of whichever classifier -
    Logistic Regression, SVM, XGBoost - comes next in the pipeline.

    Parameters
    ----------
    k            : number of features to keep. If it exceeds the input width,
                   all features are kept.
    n_estimators : trees in the scoring forest.
    class_weight : passed through so that rare degradation classes still
                   influence the importance ranking.
    """

    def __init__(
        self,
        k: int = 40,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: Optional[str] = "balanced",
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.k = k
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeImportanceSelector":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.n_features_in_ = X.shape[1]

        forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
        forest.fit(X, y)
        self.importances_ = forest.feature_importances_

        n_keep = int(min(self.k, self.n_features_in_))
        ranked = np.argsort(-self.importances_, kind="stable")[:n_keep]
        self.selected_index_ = np.sort(ranked)
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_index_] = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "support_")
        return np.asarray(X, dtype=np.float64)[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.selected_index_ if indices else self.support_


def trace_selected_feature_names(pipeline, input_names: List[str]) -> List[str]:
    """Map the surviving columns of a fitted pipeline back to feature names.

    WHY THIS IS NEEDED: each selector returns a bare numpy array, so after three
    chained selection stages the model sees anonymous column indices. To report
    *which* engineered features were selected - and to show them in the
    dashboard's explainability panel - the masks have to be composed in order.

    Input : a fitted Pipeline and the names of the columns fed into it.
    Output: the names of the columns that reach the final estimator, in order.
    """
    names = list(input_names)
    for _, step in pipeline.steps:
        if hasattr(step, "get_support") and hasattr(step, "support_"):
            mask = step.get_support()
            if len(mask) != len(names):
                # A step changed the width in a way that cannot be traced
                # (should not happen with the transformers above); fail loudly
                # rather than silently returning mismatched names.
                raise ValueError(
                    f"Selector expected {len(mask)} names but received {len(names)}."
                )
            names = [name for name, keep in zip(names, mask) if keep]
    return names


def selection_stage_counts(pipeline, n_input: int) -> dict:
    """Report how many features survive each selection stage of a fitted pipeline.

    Input : fitted Pipeline, number of features entering it.
    Output: dict mapping stage name -> feature count after that stage.
    """
    counts = {"input": n_input}
    current = n_input
    for name, step in pipeline.steps:
        if hasattr(step, "support_"):
            current = int(np.sum(step.get_support()))
            counts[name] = current
    counts["final"] = current
    return counts
