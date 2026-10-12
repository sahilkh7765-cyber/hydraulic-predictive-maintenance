"""
models.py
=========
The classifier zoo.

WHY THIS IS A SEPARATE MODULE
    Anything that ends up inside a saved `joblib` file must live in an
    importable module with a stable path. A custom estimator defined in
    `train.py` is recorded by pickle as `__main__.BalancedXGBClassifier` when
    that script is run directly, and `evaluate.py` or the Streamlit app then
    fails to load the model with "Can't get attribute". Defining estimators
    here means every process that imports `models` can deserialise the saved
    pipelines.

CONTENTS
    * BalancedXGBClassifier - XGBoost with automatic inverse-frequency sample
      weights, so it can honour the same "balanced" policy as the scikit-learn
      models.
    * make_model() - the single place where each classifier's hyperparameters
      are defined.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils import RANDOM_STATE

# XGBoost is the preferred gradient-boosting implementation, but the project
# must still run on a machine without it, so a scikit-learn fallback is wired in.
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    from sklearn.ensemble import HistGradientBoostingClassifier

    XGBOOST_AVAILABLE = False


if XGBOOST_AVAILABLE:

    class BalancedXGBClassifier(XGBClassifier):  # type: ignore[misc]
        """XGBoost that computes balanced sample weights inside `fit`.

        WHY THIS WRAPPER EXISTS: scikit-learn estimators accept
        `class_weight="balanced"`; XGBoost does not - it expects per-row
        `sample_weight` at fit time. Passing weights through a Pipeline inside
        `cross_validate` would need metadata routing, and the weights must be
        recomputed for every training fold anyway. Deriving them inside `fit`
        from that fold's own labels is simpler and cannot fall out of sync with
        the data.

        Input : the usual XGBClassifier arguments plus `balance_classes`.
        Output: a fitted booster.
        """

        def __init__(self, balance_classes: bool = True, **kwargs) -> None:
            super().__init__(**kwargs)
            self.balance_classes = balance_classes

        def fit(self, X, y, **kwargs):  # type: ignore[override]
            if self.balance_classes and "sample_weight" not in kwargs:
                classes, counts = np.unique(y, return_counts=True)
                # Inverse-frequency weighting, normalised so the mean weight is 1.
                weight_per_class = {
                    int(cls): len(y) / (len(classes) * count)
                    for cls, count in zip(classes, counts)
                }
                kwargs["sample_weight"] = np.array([weight_per_class[int(v)] for v in y])
            return super().fit(X, y, **kwargs)

else:  # pragma: no cover - only used when XGBoost is missing

    class BalancedXGBClassifier:  # type: ignore[no-redef]
        """Placeholder so the name always exists; never instantiated."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("XGBoost is not installed; make_model uses the fallback.")


def make_model(name: str, class_weight: Optional[str], with_proba: bool = False):
    """Instantiate one classifier.

    MODEL CHOICES AND WHY EACH IS HERE
        logistic_regression : linear reference point. If a linear boundary
            already separates the classes the problem is easy and a complex
            model is not justified. It is also the model that benefits most
            visibly from good features, which is exactly what this project
            sets out to measure.
        random_forest : non-linear, captures feature interactions, and supplies
            an interpretable importance ranking - which matters because a
            condition-monitoring system has to justify the alarms it raises.
        gradient_boosting : XGBoost, usually the strongest learner on tabular
            data; falls back to HistGradientBoostingClassifier if XGBoost is
            unavailable.
        svm : RBF-kernel, margin-based. A different inductive bias from the
            trees, so agreement between them is evidence that a result is a
            property of the features rather than of one model family.

    HYPERPARAMETERS are fixed rather than tuned per component. The brief asks
    for a feature-engineering study, so holding the models constant is what
    makes the three feature experiments comparable: any difference in score is
    attributable to the representation, not to a luckier hyperparameter search.

    Input : model key, class weighting strategy, whether probability estimates
            are required (only for the final refit - SVC probability estimation
            internally cross-validates and is expensive).
    Output: an unfitted estimator.
    """
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=3000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        if XGBOOST_AVAILABLE:
            return BalancedXGBClassifier(
                balance_classes=(class_weight == "balanced"),
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        return HistGradientBoostingClassifier(  # pragma: no cover
            max_iter=300,
            learning_rate=0.1,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
    if name == "svm":
        return SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight=class_weight,
            probability=with_proba,
            random_state=RANDOM_STATE,
        )
    raise ValueError(
        f"Unknown model '{name}'. Expected one of: logistic_regression, "
        f"random_forest, gradient_boosting, svm."
    )
