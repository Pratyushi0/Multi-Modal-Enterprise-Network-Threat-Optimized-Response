"""
Pillar2_3_validator.py  (renamed from Pillar2&3.py)
----------------------------------------------------
Standalone research / validation tool.

Fits Isolation Forest + EVT and OC-SVM independently on the expanded
matrix and prints a side-by-side comparison with the consensus ensemble.
Useful for ablation studies and paper table generation.

Does NOT affect the main pipeline — run independently:

    python Pillar2_3_validator.py

Outputs a comparison table to stdout and saves a CSV to
    reports/ablation_comparison.csv
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import OneClassSVM

from pillar1_ewma import build_features, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [VALIDATOR] - %(levelname)s - %(message)s",
)

# build_features and FEATURE_COLS imported from pillar1_ewma

# ---------------------------------------------------------------------------
# EVT helper (self-contained — no import from master engine)
# ---------------------------------------------------------------------------
def _compute_evt_threshold(
    scores: np.ndarray,
    contam_rate: float,
    tail_quantile: float = 0.97,
    risk_alpha: float = 0.015,
) -> float:
    """
    Fit a GPD to the upper tail of inverted IF scores.

    tail_quantile : fraction of data BELOW the tail cutoff.
                    0.97 keeps 3 % in the tail (more tail samples, less
                    extreme threshold) — good for a standalone comparison.
    """
    inverted  = -scores
    t_cutoff  = np.quantile(inverted, tail_quantile)
    excesses  = inverted[inverted > t_cutoff] - t_cutoff

    if len(excesses) < 10:
        logging.warning("EVT: insufficient tail samples — using raw quantile.")
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
# Validator class
# ---------------------------------------------------------------------------
class PillarValidator:
    """
    Fits IF+EVT, OC-SVM, and a soft-vote ensemble independently and
    produces a side-by-side ablation comparison.

    Parameters
    ----------
    risk_alpha          : EVT tail exceedance probability
    if_tail_quantile    : quantile used for EVT tail cutoff (0.97 = 3 % tail)
    svm_nu_multiplier   : OC-SVM nu = contam * multiplier (tune sensitivity)
    ensemble_threshold  : soft-vote threshold in [0,1]
    svm_weight          : weight for OC-SVM in soft vote
    """

    def __init__(
        self,
        risk_alpha: float         = 0.015,
        if_tail_quantile: float   = 0.97,
        svm_nu_multiplier: float  = 2.0,
        ensemble_threshold: float = 0.30,
        svm_weight: float         = 0.35,
    ):
        self.risk_alpha          = risk_alpha
        self.if_tail_quantile    = if_tail_quantile
        self.svm_nu_multiplier   = svm_nu_multiplier
        self.ensemble_threshold  = ensemble_threshold
        self.svm_weight          = svm_weight

    def run(
        self,
        data_path: str         = "data/processed/cert_expanded_matrix.csv",
        output_csv: str        = "reports/ablation_comparison.csv",
    ) -> pd.DataFrame:
        # ----------------------------------------------------------------
        # Load + feature-engineer
        # ----------------------------------------------------------------
        raw_df      = pd.read_csv(data_path)
        df          = build_features(raw_df)
        X           = df[FEATURE_COLS]
        y_true      = df["is_attacker"].values
        contam_rate = y_true.sum() / len(y_true)
        svm_nu      = min(contam_rate * self.svm_nu_multiplier, 0.5)

        logging.info(
            f"Validator: {len(df):,} rows | contamination={contam_rate:.5f} | "
            f"SVM nu={svm_nu:.5f}"
        )

        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # ----------------------------------------------------------------
        # Model A — Isolation Forest + EVT
        # ----------------------------------------------------------------
        logging.info("Fitting Isolation Forest …")
        if_model = IsolationForest(
            contamination=contam_rate, n_estimators=300, random_state=42, n_jobs=-1
        )
        if_model.fit(X_scaled)
        if_scores  = if_model.decision_function(X_scaled)
        evt_thresh = _compute_evt_threshold(
            if_scores, contam_rate, self.if_tail_quantile, self.risk_alpha
        )
        logging.info(f"EVT threshold: {evt_thresh:.6f}")
        if_preds = (if_scores <= evt_thresh).astype(int)
        if_prob  = MinMaxScaler().fit_transform((-if_scores).reshape(-1,1)).flatten()

        # ----------------------------------------------------------------
        # Model B — One-Class SVM
        # ----------------------------------------------------------------
        logging.info("Fitting OC-SVM …")
        svm_model  = OneClassSVM(kernel="rbf", nu=svm_nu, gamma="scale")
        svm_preds_ = svm_model.fit_predict(X_scaled)
        svm_preds  = np.where(svm_preds_ == -1, 1, 0)
        svm_scores = svm_model.decision_function(X_scaled)
        svm_prob   = MinMaxScaler().fit_transform((-svm_scores).reshape(-1,1)).flatten()

        # ----------------------------------------------------------------
        # Ensemble — soft vote
        # ----------------------------------------------------------------
        ensemble_score = self.svm_weight * svm_prob + (1 - self.svm_weight) * if_prob
        ens_preds      = (ensemble_score >= self.ensemble_threshold).astype(int)

        # ----------------------------------------------------------------
        # Build comparison table
        # ----------------------------------------------------------------
        rows = []
        for name, preds in [
            ("IForest + EVT",      if_preds),
            ("One-Class SVM",      svm_preds),
            ("Soft-Vote Ensemble", ens_preds),
        ]:
            p, r, f, _ = precision_recall_fscore_support(
                y_true, preds, average="binary", zero_division=0
            )
            cm = confusion_matrix(y_true, preds)
            rows.append({
                "Model":     name,
                "Precision": round(p, 4),
                "Recall":    round(r, 4),
                "F1":        round(f, 4),
                "TP":        int(cm[1][1]),
                "FP":        int(cm[0][1]),
                "FN":        int(cm[1][0]),
                "TN":        int(cm[0][0]),
            })

        comparison_df = pd.DataFrame(rows)

        # ----------------------------------------------------------------
        # Print report
        # ----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("     PILLAR ABLATION — STANDALONE MODEL COMPARISON")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        print()

        for name, preds in [
            ("IForest + EVT",      if_preds),
            ("One-Class SVM",      svm_preds),
            ("Soft-Vote Ensemble", ens_preds),
        ]:
            print(f"\n--- {name} ---")
            print(classification_report(y_true, preds, target_names=["Normal","Attacker"]))

        print("=" * 70 + "\n")

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        comparison_df.to_csv(output_csv, index=False)
        logging.info(f"Ablation CSV saved: {output_csv}")
        return comparison_df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    validator = PillarValidator(
        risk_alpha=0.015,
        if_tail_quantile=0.97,
        svm_nu_multiplier=2.0,
        ensemble_threshold=0.30,
        svm_weight=0.35,
    )
    validator.run(
        data_path="data/processed/cert_expanded_matrix.csv",
        output_csv="reports/ablation_comparison.csv",
    )
