"""
guardian.py
-----------
Operational alert layer for PROJECT MENTOR.

Loads the expanded feature matrix (already labeled), fits an Isolation Forest,
applies the EVT-derived threshold, and produces a ranked alert report of the
highest-risk user-day events.

Unlike the original top-10 manual override, this version:
  - Uses the EVT threshold from the master engine (loaded from reports/metrics.json)
    so results are consistent with the main pipeline.
  - Falls back to a top-N score override only if no metrics.json is found.
  - Reports per-user aggregated risk alongside per-day alerts.
  - Evaluates detection quality against ground-truth is_attacker labels.

Run after evt_master_engine.py:
    python guardian.py
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler

from pillar1_ewma import build_features, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [GUARDIAN] - %(levelname)s - %(message)s",
)

# build_features and FEATURE_COLS imported from pillar1_ewma


# ---------------------------------------------------------------------------
# EVT threshold (self-contained)
# ---------------------------------------------------------------------------
def _evt_threshold(
    scores: np.ndarray,
    contam_rate: float,
    risk_alpha: float = 0.015,
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
# Main guardian function
# ---------------------------------------------------------------------------
def run_guardian(
    data_path: str    = "data/processed/cert_expanded_matrix.csv",
    metrics_path: str = "reports/metrics.json",
    top_n: int        = 25,
) -> None:
    """
    Produces a ranked alert report of the highest-risk user-day events.

    Parameters
    ----------
    data_path    : expanded feature matrix (output of evt_master_engine.py)
    metrics_path : JSON written by evt_master_engine.py; used to sync the
                   contamination rate with the main pipeline
    top_n        : number of top-risk alerts to display in the report
    """
    if not os.path.exists(data_path):
        logging.critical(f"Expanded matrix not found: {data_path}")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Load contamination rate from metrics.json if available
    # ------------------------------------------------------------------
    contam_rate = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            contam_rate = json.load(fh).get("contamination_rate")
        logging.info(f"Loaded contamination rate from metrics: {contam_rate:.5f}")
    else:
        logging.warning(
            "metrics.json not found — computing contamination from label column."
        )

    # ------------------------------------------------------------------
    # Load + feature-engineer
    # ------------------------------------------------------------------
    logging.info(f"Loading expanded matrix: {data_path}")
    raw_df = pd.read_csv(data_path)
    df     = build_features(raw_df)

    if contam_rate is None:
        contam_rate = float(df["is_attacker"].sum()) / len(df)
        logging.info(f"Computed contamination rate: {contam_rate:.5f}")

    X      = df[FEATURE_COLS]
    y_true = df["is_attacker"].values

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Fit IF + derive EVT threshold
    # ------------------------------------------------------------------
    logging.info("Fitting Isolation Forest …")
    model = IsolationForest(
        contamination=contam_rate,
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    if_scores = model.decision_function(X_scaled)

    evt_thresh = _evt_threshold(if_scores, contam_rate)
    logging.info(f"EVT threshold: {evt_thresh:.6f}")

    df["anomaly_score"]       = if_scores
    df["evt_flagged"]         = (if_scores <= evt_thresh).astype(int)
    df["risk_rank"]           = df["anomaly_score"].rank(method="first")  # lower score = higher rank

    # ------------------------------------------------------------------
    # Top-N alert report (ranked by raw anomaly score)
    # ------------------------------------------------------------------
    top_alerts = df.nsmallest(top_n, "anomaly_score")[
        ["user", "day", "total_logins", "after_hours_logins",
         "usb_file_copies", "web_clicks", "file_touches",
         "anomaly_score", "evt_flagged", "is_attacker"]
    ].copy()

    correct   = int(top_alerts["is_attacker"].sum())
    fp_count  = top_n - correct
    total_att = int(y_true.sum())

    print("\n" + "=" * 90)
    print("              PROJECT MENTOR — GUARDIAN ALERT REPORT")
    print("=" * 90)
    print(f"  EVT threshold: {evt_thresh:.6f}  |  Contamination: {contam_rate:.5f}  |  Top-N shown: {top_n}")
    print("=" * 90)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(top_alerts.to_string(index=False))
    print("=" * 90)
    print(f"\n  Top-{top_n} alerts caught {correct} of {total_att} known attackers")
    print(f"  False positives in top-{top_n}: {fp_count}")

    # ------------------------------------------------------------------
    # Per-user risk summary (aggregate days flagged per user)
    # ------------------------------------------------------------------
    user_risk = (
        df[df["evt_flagged"] == 1]
        .groupby("user")
        .agg(
            flagged_days  =("evt_flagged",   "sum"),
            min_score     =("anomaly_score", "min"),
            attacker_days =("is_attacker",   "sum"),
        )
        .sort_values("min_score")
        .reset_index()
    )

    print("\n  Per-user risk summary (EVT-flagged users only):")
    print(user_risk.to_string(index=False))

    # ------------------------------------------------------------------
    # Full-dataset evaluation
    # ------------------------------------------------------------------
    y_pred = df["evt_flagged"].values
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  Full-dataset IF+EVT evaluation:")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_guardian(
        data_path="data/processed/cert_expanded_matrix.csv",
        metrics_path="reports/metrics.json",
        top_n=25,
    )
