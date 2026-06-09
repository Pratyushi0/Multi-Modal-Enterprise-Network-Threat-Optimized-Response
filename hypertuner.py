"""
hypertuner.py
-------------
Hyperparameter grid search for the IF + EVT pillar.

Sweeps over n_estimators, contamination multiplier, EVT risk_alpha, and
soft-vote ensemble threshold.  Reports precision / recall / F1 for every
combination and saves the best configuration to reports/best_hyperparams.json
so evt_master_engine.py can optionally load it.

Since recall is the primary objective, the sweep ranks configurations by
recall first, then F1 (to filter out degenerate high-recall / zero-precision
configs), then precision.

Run independently:
    python hypertuner.py

Outputs:
    reports/hypertuner_results.csv
    reports/best_hyperparams.json
"""

import itertools
import json
import logging
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import OneClassSVM

from pillar1_ewma import build_features, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [HYPER-TUNER] - %(levelname)s - %(message)s",
)

# build_features and FEATURE_COLS imported from pillar1_ewma


# ---------------------------------------------------------------------------
# EVT threshold (self-contained)
# ---------------------------------------------------------------------------
def _evt_threshold(
    scores: np.ndarray,
    contam_rate: float,
    risk_alpha: float,
) -> float:
    inverted = -scores
    t_cutoff = np.quantile(inverted, 1.0 - contam_rate)
    excesses = inverted[inverted > t_cutoff] - t_cutoff

    if len(excesses) < 10:
        return float(-t_cutoff)

    n, N_u = len(scores), len(excesses)
    shape, _loc, scale = stats.genpareto.fit(excesses)

    if abs(shape) < 1e-6:
        excess_threshold = scale * np.log(risk_alpha * (n / N_u))
    else:
        excess_threshold = (scale / shape) * (
            ((risk_alpha * (n / N_u)) ** (-shape)) - 1
        )
    return float(-(t_cutoff + excess_threshold))


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def evaluate_hyperparameters(
    data_path: str            = "data/processed/cert_expanded_matrix.csv",
    results_csv: str          = "reports/hypertuner_results.csv",
    best_params_json: str     = "reports/best_hyperparams.json",
) -> pd.DataFrame:
    """
    Sweeps the search space and saves results.

    Search axes
    -----------
    n_estimators        : IF tree count
    contam_multiplier   : contam = true_rate × multiplier  (keeps it data-driven)
    risk_alpha          : EVT tail exceedance probability
    ensemble_threshold  : soft-vote cutoff for the combined SVM+IF score
    svm_weight          : weight for OC-SVM in the soft vote

    Returns the full results DataFrame.
    """
    logging.info(f"Loading dataset: {data_path}")
    raw_df = pd.read_csv(data_path)
    df     = build_features(raw_df)

    X      = df[FEATURE_COLS]
    y_true = df["is_attacker"].values

    true_contam = float(y_true.sum()) / len(y_true)
    logging.info(f"True contamination rate: {true_contam:.5f}")

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Search space
    # ------------------------------------------------------------------
    search_space = list(itertools.product(
        [200, 400],          # n_estimators
        [1.0, 2.0, 3.0],     # contam_multiplier
        [0.005, 0.015, 0.03],# risk_alpha
        [0.25, 0.30, 0.40],  # ensemble_threshold
        [0.30, 0.40],        # svm_weight
    ))

    total = len(search_space)
    logging.info(f"Grid size: {total} configurations")

    print("\n" + "=" * 100)
    print("          MENTOR HYPERPARAMETER GRID SWEEP  (recall-primary objective)")
    print("=" * 100)
    print(
        f"{'Trees':<7}{'CxMult':<8}{'Alpha':<8}{'Thr':<7}{'SVMw':<7}"
        f"{'Precision':<12}{'Recall':<10}{'F1':<10}{'TP':<8}{'FP':<8}"
    )
    print("-" * 100)

    results = []

    # Pre-fit SVM options to avoid refitting for every IF/EVT combo
    svm_cache: dict = {}

    for i, (n_trees, c_mult, alpha, ens_thresh, svm_w) in enumerate(search_space, 1):
        # IF contamination is capped at 0.5 (sklearn hard limit)
        contam = min(true_contam * c_mult, 0.499)
        svm_nu = min(true_contam * c_mult, 0.499)

        # ---- Isolation Forest ----------------------------------------
        clf = IsolationForest(
            contamination=contam,
            n_estimators=n_trees,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_scaled)
        if_scores = clf.decision_function(X_scaled)
        if_prob   = MinMaxScaler().fit_transform(
            (-if_scores).reshape(-1, 1)
        ).flatten()

        # EVT threshold (for standalone IF evaluation)
        evt_thresh = _evt_threshold(if_scores, contam, alpha)

        # ---- OC-SVM (cached by nu value) -----------------------------
        if svm_nu not in svm_cache:
            svm = OneClassSVM(kernel="rbf", nu=svm_nu, gamma="scale")
            svm.fit(X_scaled)
            svm_cache[svm_nu] = svm
        else:
            svm = svm_cache[svm_nu]

        svm_scores = svm.decision_function(X_scaled)
        svm_prob   = MinMaxScaler().fit_transform(
            (-svm_scores).reshape(-1, 1)
        ).flatten()

        # ---- Soft-vote ensemble --------------------------------------
        ens_score = svm_w * svm_prob + (1 - svm_w) * if_prob
        y_pred    = (ens_score >= ens_thresh).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred,    zero_division=0)
        f1   = f1_score(y_true, y_pred,        zero_division=0)
        tp   = int(((y_pred == 1) & (y_true == 1)).sum())
        fp   = int(((y_pred == 1) & (y_true == 0)).sum())

        print(
            f"{n_trees:<7}{c_mult:<8.1f}{alpha:<8.3f}{ens_thresh:<7.2f}{svm_w:<7.2f}"
            f"{prec:<12.4f}{rec:<10.4f}{f1:<10.4f}{tp:<8}{fp:<8}"
        )

        results.append({
            "n_estimators":        n_trees,
            "contam_multiplier":   c_mult,
            "contam":              round(contam, 6),
            "risk_alpha":          alpha,
            "ensemble_threshold":  ens_thresh,
            "svm_weight":          svm_w,
            "if_weight":           round(1 - svm_w, 2),
            "precision":           round(prec, 6),
            "recall":              round(rec,  6),
            "f1":                  round(f1,   6),
            "tp":                  tp,
            "fp":                  fp,
        })

    print("=" * 100)

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Best config: maximise recall, break ties by f1, then precision
    # (filter out configs where precision is literally 0)
    # ------------------------------------------------------------------
    valid = results_df[results_df["precision"] > 0].copy()
    if valid.empty:
        logging.warning("All configs had zero precision — relaxing filter.")
        valid = results_df.copy()

    best = valid.sort_values(
        ["recall", "f1", "precision"],
        ascending=[False, False, False],
    ).iloc[0]

    print(f"\n  OPTIMAL CONFIGURATION (recall-primary):")
    print(f"  Trees:              {int(best['n_estimators'])}")
    print(f"  Contam multiplier:  {best['contam_multiplier']}  →  contam={best['contam']:.6f}")
    print(f"  Risk alpha:         {best['risk_alpha']}")
    print(f"  Ensemble threshold: {best['ensemble_threshold']}")
    print(f"  SVM weight:         {best['svm_weight']}")
    print(f"  Recall:             {best['recall']:.4f}")
    print(f"  Precision:          {best['precision']:.4f}")
    print(f"  F1:                 {best['f1']:.4f}")
    print(f"  TP: {int(best['tp'])}  FP: {int(best['fp'])}")
    print("=" * 100 + "\n")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    logging.info(f"Full results saved: {results_csv}")

    best_params = {
        "n_estimators":       int(best["n_estimators"]),
        "contam_multiplier":  float(best["contam_multiplier"]),
        "risk_alpha":         float(best["risk_alpha"]),
        "ensemble_threshold": float(best["ensemble_threshold"]),
        "svm_weight":         float(best["svm_weight"]),
        "if_weight":          float(best["if_weight"]),
        "achieved_recall":    float(best["recall"]),
        "achieved_precision": float(best["precision"]),
        "achieved_f1":        float(best["f1"]),
    }
    with open(best_params_json, "w") as fh:
        json.dump(best_params, fh, indent=2)
    logging.info(f"Best params saved: {best_params_json}")

    return results_df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_hyperparameters(
        data_path="data/processed/cert_expanded_matrix.csv",
        results_csv="reports/hypertuner_results.csv",
        best_params_json="reports/best_hyperparams.json",
    )
