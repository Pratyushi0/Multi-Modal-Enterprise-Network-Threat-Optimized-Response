import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BENCHMARKER] - %(levelname)s - %(message)s')


def compute_evt_threshold(anomaly_scores, contam_rate, risk_alpha=0.015):
    """
    FIX: n and N_u defined before fit. Shape ≈ 0 guard added. Tail size fallback added.
    """
    inverted_scores = -anomaly_scores
    t_cutoff = np.quantile(inverted_scores, 1.0 - contam_rate)
    tail_excesses = inverted_scores[inverted_scores > t_cutoff] - t_cutoff

    if len(tail_excesses) < 10:
        logging.warning("EVT: Insufficient tail samples — falling back to raw quantile threshold.")
        return -t_cutoff

    n   = len(anomaly_scores)
    N_u = len(tail_excesses)
    shape, loc, scale = stats.genpareto.fit(tail_excesses)

    if abs(shape) < 1e-6:
        logging.warning("EVT: shape ≈ 0 — using exponential limit formula.")
        return -(t_cutoff + scale * np.log(risk_alpha * (n / N_u)))

    calculated_excess = t_cutoff + (scale / shape) * (
        ((risk_alpha * (n / N_u)) ** (-shape)) - 1
    )
    return -calculated_excess


def build_advanced_features(df, ewm_span=14):
    logging.info("Pillar 1: Engineering multi-domain rolling velocity tensors...")
    df = df.sort_values(by=['user', 'day']).copy()

    for col, alias in [
        ('total_logins', 'logins'),
        ('usb_file_copies', 'usb'),
        ('web_clicks', 'web'),
        ('file_touches', 'file')
    ]:
        df[f'ewm_mean_{alias}'] = df.groupby('user')[col].transform(
            lambda x: x.ewm(span=ewm_span).mean()
        )
        df[f'ewm_std_{alias}'] = df.groupby('user')[col].transform(
            lambda x: x.ewm(span=ewm_span).std()
        ).fillna(1.0)

    df['login_deviation'] = (df['total_logins']     - df['ewm_mean_logins']) / df['ewm_std_logins']
    df['usb_deviation']   = (df['usb_file_copies']  - df['ewm_mean_usb'])    / df['ewm_std_usb']
    df['web_deviation']   = (df['web_clicks']        - df['ewm_mean_web'])    / df['ewm_std_web']
    df['file_deviation']  = (df['file_touches']      - df['ewm_mean_file'])   / df['ewm_std_file']
    df['after_hours_velocity'] = df['after_hours_logins']

    dev_cols = ['login_deviation', 'usb_deviation', 'web_deviation', 'file_deviation']
    df[dev_cols] = df[dev_cols].replace([np.inf, -np.inf], 0).fillna(0)

    df['login_dev_3d_velocity']   = df.groupby('user')['login_deviation'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['usb_dev_3d_velocity']     = df.groupby('user')['usb_deviation'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['usb_dev_7d_velocity']     = df.groupby('user')['usb_deviation'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df['after_hours_3d_velocity'] = df.groupby('user')['after_hours_velocity'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['web_dev_3d_velocity']     = df.groupby('user')['web_deviation'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['web_dev_7d_velocity']     = df.groupby('user')['web_deviation'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df['file_dev_3d_velocity']    = df.groupby('user')['file_deviation'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['file_dev_7d_velocity']    = df.groupby('user')['file_deviation'].transform(lambda x: x.rolling(7, min_periods=1).sum())

    df.fillna(0, inplace=True)
    return df


def run_benchmarks(data_path="data/processed/cert_expanded_matrix.csv"):
    logging.info(f"Loading full expanded data matrix from {data_path}...")
    raw_df = pd.read_csv(data_path)

    assert 'is_attacker' in raw_df.columns, "CRITICAL: is_attacker column missing!"

    df = build_advanced_features(raw_df)
    y_true = df['is_attacker'].values

    # FIX: Compute true contamination from actual labels — never hardcode
    TRUE_CONTAM = float(y_true.sum()) / len(y_true)
    logging.info(f"True contamination rate: {TRUE_CONTAM:.5f} ({int(y_true.sum())} attacks / {len(y_true):,} rows)")

    feature_cols = [
        'login_deviation', 'usb_deviation', 'web_deviation', 'file_deviation',
        'after_hours_velocity', 'login_dev_3d_velocity', 'usb_dev_3d_velocity',
        'usb_dev_7d_velocity', 'after_hours_3d_velocity', 'web_dev_3d_velocity',
        'web_dev_7d_velocity', 'file_dev_3d_velocity', 'file_dev_7d_velocity'
    ]

    X = df[feature_cols]

    logging.info("Pillar 2: Executing Robust Scaling across feature dimensions...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # --- MODEL 1: Isolation Forest + EVT ---
    logging.info("Running Model 1/4: Isolation Forest + EVT...")
    if_model = IsolationForest(
        contamination=TRUE_CONTAM,
        n_estimators=400,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )
    if_model.fit(X_scaled)
    if_scores = if_model.decision_function(X_scaled)
    evt_thresh = compute_evt_threshold(if_scores, TRUE_CONTAM)
    y_pred_evt = (if_scores <= evt_thresh).astype(int)

    # --- MODEL 2: One-Class SVM ---
    logging.info("Running Model 2/4: One-Class SVM (RBF Kernel)...")
    # FIX: nu set to TRUE_CONTAM instead of hardcoded 0.01
    oc_svm = OneClassSVM(kernel='rbf', nu=TRUE_CONTAM, gamma='scale')
    raw_ocsvm = oc_svm.fit_predict(X_scaled)
    y_pred_ocsvm = np.where(raw_ocsvm == -1, 1, 0)

    # --- MODEL 3: Local Outlier Factor ---
    logging.info("Running Model 3/4: Local Outlier Factor...")
    # FIX: n_neighbors increased from 20→50 to suppress duplicate-value warning
    lof = LocalOutlierFactor(n_neighbors=50, contamination=TRUE_CONTAM, n_jobs=-1)
    raw_lof = lof.fit_predict(X_scaled)
    y_pred_lof = np.where(raw_lof == -1, 1, 0)

    # --- MODEL 4: MENTOR Consensus Ensemble (IF ∩ OC-SVM) ---
    # FIX: Added consensus model as 4th row so the table is self-contained
    logging.info("Running Model 4/4: MENTOR Consensus Ensemble (IF+EVT ∩ OC-SVM)...")
    y_pred_consensus = np.bitwise_and(y_pred_evt, y_pred_ocsvm)

    # --- EVALUATION ENGINE ---
    models = {
        "Isolation Forest + EVT":        y_pred_evt,
        "One-Class SVM (RBF)":           y_pred_ocsvm,
        "Local Outlier Factor (LOF)":    y_pred_lof,
        "MENTOR Consensus (IF ∩ OC-SVM)": y_pred_consensus,
    }

    print("\n" + "=" * 90)
    print("         PROJECT MENTOR: CROSS-MODEL BENCHMARK EVALUATION REPORT")
    print("=" * 90)
    print(f"  True contamination rate: {TRUE_CONTAM:.5f} ({int(y_true.sum())} attacks / {len(y_true):,} total rows)")
    print("=" * 90)
    print(f"{'Classifier Model':<35}{'Precision':<12}{'Recall':<10}{'F1-Score':<12}{'TP':<8}{'FP':<8}")
    print("-" * 90)

    for name, preds in models.items():
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)
        cm   = confusion_matrix(y_true, preds)
        tp   = cm[1][1]
        fp   = cm[0][1]
        marker = " ← MENTOR" if "Consensus" in name else ""
        print(f"{name:<35}{prec:<12.3f}{rec:<10.3f}{f1:<12.3f}{tp:<8}{fp:<8}{marker}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_benchmarks()
