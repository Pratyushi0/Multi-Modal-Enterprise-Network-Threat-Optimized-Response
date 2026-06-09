"""
src/models.py
-------------
PROJECT MENTOR — Model Training & Scoring

All model logic lives here. Scripts import from this module.

Exports
-------
compute_evt_threshold(scores, contam_rate, risk_alpha) -> float
fit_isolation_forest(X_train, contam_rate, n_estimators) -> model
fit_ocsvm(X_train, nu) -> model
score_ensemble(if_scores, svm_scores, if_weight, svm_weight) -> np.ndarray
threshold_by_f1(y_true, ensemble_scores) -> (best_threshold, best_f1)
"""

import logging

import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)


def compute_evt_threshold(
    anomaly_scores: np.ndarray,
    contam_rate: float,
    risk_alpha: float = 0.015,
) -> float:
    """
    Derive an anomaly detection threshold from the tail of the IF score
    distribution using a Generalised Pareto Distribution (GPD) fit.

    Extreme Value Theory approach (Siffer et al., KDD 2017).

    Parameters
    ----------
    anomaly_scores : IF decision_function output.  Lower = more anomalous.
    contam_rate    : True contamination fraction.  Sets the tail cutoff.
    risk_alpha     : Target tail exceedance probability (default 0.015).

    Returns
    -------
    threshold : float in IF score space (negative).  Flag rows where
                score <= threshold as anomalies.
    """
    inverted = -anomaly_scores                            # higher = more anomalous
    t_cutoff = np.quantile(inverted, 1.0 - contam_rate)  # tail boundary

    excesses = inverted[inverted > t_cutoff] - t_cutoff
    if len(excesses) < 10:
        log.warning(
            "EVT: only %d tail samples — falling back to raw quantile threshold.",
            len(excesses),
        )
        return float(-t_cutoff)

    n   = len(anomaly_scores)
    N_u = len(excesses)

    shape, _loc, scale = stats.genpareto.fit(excesses)

    if abs(shape) < 1e-6:
        # Exponential limit (shape -> 0)
        log.debug("EVT: shape≈0 — using exponential-limit formula.")
        excess_threshold = scale * np.log(risk_alpha * (n / N_u))
    else:
        excess_threshold = (scale / shape) * (
            ((risk_alpha * (n / N_u)) ** (-shape)) - 1
        )

    threshold = float(-(t_cutoff + excess_threshold))
    log.debug("EVT: t_cutoff=%.6f  shape=%.4f  scale=%.4f  threshold=%.6f",
              t_cutoff, shape, scale, threshold)
    return threshold


def fit_isolation_forest(
    X_train: np.ndarray,
    contam_rate: float,
    n_estimators: int = 400,
    random_state: int = 42,
) -> IsolationForest:
    """
    Fit Isolation Forest on NORMAL training data only (no attacker rows).

    Parameters
    ----------
    X_train      : Scaled feature matrix — should contain NORMAL rows only.
    contam_rate  : Contamination passed to sklearn (controls internal threshold).
    n_estimators : Number of trees.

    Returns
    -------
    Fitted IsolationForest instance.
    """
    model = IsolationForest(
        contamination=contam_rate,
        n_estimators=n_estimators,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def fit_ocsvm(
    X_train: np.ndarray,
    nu: float,
) -> OneClassSVM:
    """
    Fit One-Class SVM on NORMAL training data only.

    Parameters
    ----------
    X_train : Scaled feature matrix — normal rows only.
    nu      : Upper bound on training error fraction.  Should be set to
              true_contamination_rate or slightly above.

    Returns
    -------
    Fitted OneClassSVM instance.
    """
    nu = float(np.clip(nu, 1e-4, 0.499))
    model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    model.fit(X_train)
    return model


def anomaly_probabilities(
    if_model: IsolationForest,
    svm_model: OneClassSVM,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normalised anomaly probability proxies in [0, 1] for both models.

    Lower IF decision_function = more anomalous -> invert & scale.
    Lower SVM decision_function = more anomalous -> invert & scale.

    Returns
    -------
    (if_prob, svm_prob) each in [0, 1], shape (n_samples,)
    """
    if_scores  = if_model.decision_function(X)
    svm_scores = svm_model.decision_function(X)

    if_prob  = MinMaxScaler().fit_transform((-if_scores).reshape(-1, 1)).flatten()
    svm_prob = MinMaxScaler().fit_transform((-svm_scores).reshape(-1, 1)).flatten()

    return if_prob, svm_prob


def score_ensemble(
    if_prob: np.ndarray,
    svm_prob: np.ndarray,
    if_weight: float = 0.65,
    svm_weight: float = 0.35,
) -> np.ndarray:
    """
    Weighted soft-vote ensemble score in [0, 1].

    Higher score = more anomalous.

    Parameters
    ----------
    if_prob, svm_prob : normalised probability proxies from anomaly_probabilities()
    if_weight         : weight for IF (default 0.65 — IF is more reliable on CERT)
    svm_weight        : weight for OC-SVM (default 0.35)

    Returns
    -------
    ensemble_score : np.ndarray, shape (n_samples,)
    """
    assert abs(if_weight + svm_weight - 1.0) < 1e-6, \
        "Weights must sum to 1.0"
    return if_weight * if_prob + svm_weight * svm_prob


def threshold_by_f1(
    y_true: np.ndarray,
    ensemble_scores: np.ndarray,
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """
    Find the ensemble threshold that maximises F1 on the provided data.

    Use only on the TRAINING fold — never on test data.

    Parameters
    ----------
    y_true          : Ground-truth binary labels.
    ensemble_scores : Continuous anomaly scores in [0, 1].
    n_thresholds    : Number of candidate thresholds to sweep.

    Returns
    -------
    (best_threshold, best_f1)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thr, best_f1 = 0.5, 0.0
    for thr in thresholds:
        y_pred = (ensemble_scores >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    log.info("F1-optimal threshold: %.4f  (train F1=%.4f)", best_thr, best_f1)
    return best_thr, best_f1