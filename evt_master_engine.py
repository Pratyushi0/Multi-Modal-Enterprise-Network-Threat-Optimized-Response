"""
evt_master_engine.py
--------------------
PROJECT MENTOR — Consensus Ensemble Threat Detection Engine

Pipeline:
    1. Run multi-stream aggregation (HTTP + file telemetry)
    2. Pillar 1  — contextual behavioural baselining (EWM + velocity features)
    3. Pillar 2  — RobustScaler normalisation
    4. Pillar 3A — One-Class SVM anomaly scoring
    5. Pillar 3B — Isolation Forest + EVT dynamic threshold
    6. Ensemble  — soft-vote union (recall-maximising), tunable threshold
    7. Evaluation report + save metrics JSON for visualiser

Ensemble strategy (recall-maximised):
    Both models produce continuous anomaly probability proxies in [0, 1].
    The ensemble score is a weighted average; a sample is flagged as an
    attacker if the score exceeds `ensemble_threshold`.  The default
    threshold (0.30) strongly favours recall over precision — tune upward
    if false-positive fatigue becomes an operational concern.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import OneClassSVM

from multi_stream_aggregator import run_multi_domain_pipeline
from pillar1_ewma import build_features, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [MASTER-ENGINE] - %(levelname)s - %(message)s",
)

# FEATURE_COLS and build_features are imported from pillar1_ewma


# ---------------------------------------------------------------------------
# EVT threshold computation
# ---------------------------------------------------------------------------
def compute_evt_threshold(
    anomaly_scores: np.ndarray,
    contam_rate: float,
    risk_alpha: float = 0.015,
) -> float:
    """
    Derive a detection threshold from the tail of the IF score distribution
    using a Generalised Pareto Distribution fit (Extreme Value Theory).

    Parameters
    ----------
    anomaly_scores : raw IF decision_function output (lower = more anomalous)
    contam_rate    : true contamination fraction (used to set tail cutoff)
    risk_alpha     : target tail exceedance probability

    Returns
    -------
    threshold in native IF score space (negative float)
    """
    inverted = -anomaly_scores
    t_cutoff  = np.quantile(inverted, 1.0 - contam_rate)
    excesses  = inverted[inverted > t_cutoff] - t_cutoff

    if len(excesses) < 10:
        logging.warning(
            "EVT: too few tail samples — falling back to raw quantile threshold."
        )
        return float(-t_cutoff)

    n   = len(anomaly_scores)
    N_u = len(excesses)

    shape, _loc, scale = stats.genpareto.fit(excesses)

    if abs(shape) < 1e-6:
        logging.warning("EVT: shape ≈ 0 — using exponential-limit formula.")
        excess_threshold = scale * np.log(risk_alpha * (n / N_u))
    else:
        excess_threshold = (scale / shape) * (
            ((risk_alpha * (n / N_u)) ** (-shape)) - 1
        )

    return float(-(t_cutoff + excess_threshold))


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------
class MENTOREnsembleEngine:
    """
    Parameters
    ----------
    true_contam        : contamination rate computed from actual label distribution
    risk_alpha         : EVT tail exceedance probability (lower = more selective IF)
    ensemble_threshold : soft-vote score cutoff in [0, 1].
                         Lower = more aggressive recall.  Default 0.30 gives
                         maximum recall at the cost of more false positives.
    svm_weight         : weight given to OC-SVM in the ensemble (IF gets 1 - svm_weight)
    """

    def __init__(
        self,
        true_contam: float,
        risk_alpha: float = 0.015,
        ensemble_threshold: float = 0.30,
        svm_weight: float = 0.35,
    ):
        self.contam             = true_contam
        self.risk_alpha         = risk_alpha
        self.ensemble_threshold = ensemble_threshold
        self.svm_weight         = svm_weight
        self.if_weight          = 1.0 - svm_weight

        self.scaler = RobustScaler()
        self.prob_scaler = MinMaxScaler()  # normalises IF scores to [0,1]

        # OC-SVM: nu is set to 2× the true contamination rate so the
        # hypersphere is large enough to contain most normal behaviour
        # without being so tight that it misses attacker deviations.
        svm_nu = min(self.contam * 2.0, 0.5)
        self.svm_model = OneClassSVM(kernel="rbf", nu=svm_nu, gamma="scale")

        self.if_model = IsolationForest(
            contamination=self.contam,
            n_estimators=400,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )

        logging.info(
            f"Engine initialised — true contamination: {self.contam:.5f} "
            f"({self.contam * 100:.3f}%)  |  SVM nu={svm_nu:.5f}  |  "
            f"ensemble_threshold={self.ensemble_threshold}"
        )

    def execute_contextual_baselining(
        self, df: pd.DataFrame, ewm_span: int = 14
    ) -> pd.DataFrame:
        logging.info("Pillar 1: Extracting multi-domain historical behavioural trends …")
        return build_features(df, ewm_span=ewm_span)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        data_path: str,
        metrics_output_path: str = "reports/metrics.json",
    ) -> None:
        raw_df = pd.read_csv(data_path)
        assert "is_attacker" in raw_df.columns, (
            "CRITICAL: is_attacker column missing from expanded matrix!"
        )

        processed_df = self.execute_contextual_baselining(raw_df)

        X      = processed_df[FEATURE_COLS]
        y_true = processed_df["is_attacker"].values

        # Pillar 2 — normalisation
        logging.info("Pillar 2: Normalising features via RobustScaler …")
        X_scaled = self.scaler.fit_transform(X)

        # ----------------------------------------------------------
        # Pillar 3A — OC-SVM (anomaly probability proxy)
        # ----------------------------------------------------------
        logging.info("Pillar 3A: Fitting OC-SVM hyperplane …")
        svm_preds  = self.svm_model.fit_predict(X_scaled)
        # OC-SVM decision_function: higher = more normal
        svm_scores = self.svm_model.decision_function(X_scaled)
        # Invert so that higher value = more anomalous, then scale to [0,1]
        svm_prob = MinMaxScaler().fit_transform(
            (-svm_scores).reshape(-1, 1)
        ).flatten()

        # ----------------------------------------------------------
        # Pillar 3B — Isolation Forest + EVT
        # ----------------------------------------------------------
        logging.info("Pillar 3B: Fitting Isolation Forest + EVT threshold …")
        self.if_model.fit(X_scaled)
        if_scores = self.if_model.decision_function(X_scaled)

        evt_thresh = compute_evt_threshold(if_scores, self.contam, self.risk_alpha)
        logging.info(f"Dynamic EVT horizon established at: {evt_thresh:.6f}")

        # Normalise IF scores to [0, 1] (lower IF score = higher anomaly prob)
        if_prob = self.prob_scaler.fit_transform(
            (-if_scores).reshape(-1, 1)
        ).flatten()

        # ----------------------------------------------------------
        # Ensemble — weighted soft vote (recall-maximised)
        # ----------------------------------------------------------
        logging.info("Ensemble: Applying weighted soft-vote union …")
        ensemble_score = (
            self.svm_weight * svm_prob +
            self.if_weight  * if_prob
        )
        y_pred = (ensemble_score >= self.ensemble_threshold).astype(int)
        processed_df["predicted_attacker"] = y_pred
        processed_df["ensemble_score"]     = ensemble_score

        # ----------------------------------------------------------
        # Diagnostic — show score separation between classes
        # ----------------------------------------------------------
        att_scores  = ensemble_score[y_true == 1]
        norm_scores = ensemble_score[y_true == 0]
        logging.info(
            f"Score separation — attacker mean: {att_scores.mean():.4f}  "
            f"normal mean: {norm_scores.mean():.4f}"
        )

        # ----------------------------------------------------------
        # Evaluation report
        # ----------------------------------------------------------
        cm     = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Attacker"],
            output_dict=True,
        )

        print("\n" + "=" * 70)
        print("     PROJECT MENTOR: CONSENSUS ENSEMBLE EVALUATION REPORT")
        print("=" * 70)
        print(f"\nTrue contamination rate: {self.contam:.5f} ({self.contam * 100:.3f}%)")
        print(f"Ensemble threshold:      {self.ensemble_threshold}")
        print(f"SVM weight: {self.svm_weight}  |  IF weight: {self.if_weight}")
        print("\nCLASSIFICATION REPORT:")
        print(
            classification_report(
                y_true, y_pred, target_names=["Normal", "Attacker"]
            )
        )
        print("CONFUSION MATRIX:")
        print(f" -> True Negatives   (Innocents confirmed):   {cm[0][0]:>10,}")
        print(f" -> False Positives  (System noise fatigue):  {cm[0][1]:>10,}")
        print(f" -> False Negatives  (Missed threat surface): {cm[1][0]:>10,}")
        print(f" -> True Positives   (Threats intercepted):   {cm[1][1]:>10,}")
        print("=" * 70 + "\n")

        # ----------------------------------------------------------
        # Save metrics JSON for visualiser
        # ----------------------------------------------------------
        metrics = {
            "contamination_rate": float(self.contam),
            "ensemble_threshold": float(self.ensemble_threshold),
            "svm_weight":         float(self.svm_weight),
            "if_weight":          float(self.if_weight),
            "evt_threshold":      float(evt_thresh),
            "attacker_score_mean":  float(att_scores.mean()),
            "attacker_score_std":   float(att_scores.std()),
            "normal_score_mean":    float(norm_scores.mean()),
            "normal_score_std":     float(norm_scores.std()),
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
            "attacker": {
                "precision": report["Attacker"]["precision"],
                "recall":    report["Attacker"]["recall"],
                "f1":        report["Attacker"]["f1-score"],
                "support":   report["Attacker"]["support"],
            },
            "normal": {
                "precision": report["Normal"]["precision"],
                "recall":    report["Normal"]["recall"],
                "f1":        report["Normal"]["f1-score"],
                "support":   report["Normal"]["support"],
            },
            # Per-model breakdown (for standalone comparison in visualiser)
            "model_breakdown": {
                "IForest+EVT": {
                    "precision": float(
                        (y_true[(if_scores <= evt_thresh)] == 1).sum() /
                        max((if_scores <= evt_thresh).sum(), 1)
                    ),
                    "recall": float(
                        (y_true[(if_scores <= evt_thresh)] == 1).sum() /
                        max(y_true.sum(), 1)
                    ),
                },
                "OC-SVM": {
                    "precision": float(
                        (y_true[svm_preds == -1] == 1).sum() /
                        max((svm_preds == -1).sum(), 1)
                    ),
                    "recall": float(
                        (y_true[svm_preds == -1] == 1).sum() /
                        max(y_true.sum(), 1)
                    ),
                },
            },
        }

        # Derive F1 for per-model breakdown
        for model_name, m in metrics["model_breakdown"].items():
            p, r = m["precision"], m["recall"]
            m["f1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        # Add ensemble entry to model_breakdown for visualiser
        metrics["model_breakdown"]["Ensemble"] = {
            "precision": report["Attacker"]["precision"],
            "recall":    report["Attacker"]["recall"],
            "f1":        report["Attacker"]["f1-score"],
        }

        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        with open(metrics_output_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logging.info(f"Metrics saved: {metrics_output_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ZIP_ARCHIVE      = "data/raw/dataset.zip"
    LABELED_MATRIX   = "data/processed/cert_labeled_matrix.csv"
    EXPANDED_MATRIX  = "data/processed/cert_expanded_matrix.csv"
    METRICS_OUT      = "reports/metrics.json"
    BEST_PARAMS_PATH = "reports/best_hyperparams.json"

    print("⚡ Starting Integrated MENTOR Consensus Ensemble Threat Pipeline …")

    ok = run_multi_domain_pipeline(ZIP_ARCHIVE, LABELED_MATRIX, EXPANDED_MATRIX)
    if not ok or not os.path.exists(EXPANDED_MATRIX):
        logging.critical("Pipeline stopped: expanded matrix missing.")
        raise SystemExit(1)

    label_df    = pd.read_csv(EXPANDED_MATRIX, usecols=["is_attacker"])
    true_contam = float(label_df["is_attacker"].sum()) / len(label_df)

    # Load tuned hyperparams if hypertuner.py has been run, else use defaults
    risk_alpha         = 0.015
    ensemble_threshold = 0.30
    svm_weight         = 0.35

    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as fh:
            best = json.load(fh)
        risk_alpha         = best.get("risk_alpha",         risk_alpha)
        ensemble_threshold = best.get("ensemble_threshold", ensemble_threshold)
        svm_weight         = best.get("svm_weight",         svm_weight)
        logging.info(
            f"Loaded tuned hyperparams from {BEST_PARAMS_PATH}: "
            f"alpha={risk_alpha}  threshold={ensemble_threshold}  svm_w={svm_weight}"
        )
    else:
        logging.info("No best_hyperparams.json found — using defaults.")

    engine = MENTOREnsembleEngine(
        true_contam=true_contam,
        risk_alpha=risk_alpha,
        ensemble_threshold=ensemble_threshold,
        svm_weight=svm_weight,
    )
    engine.run_pipeline(EXPANDED_MATRIX, metrics_output_path=METRICS_OUT)
