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

def build_advanced_features(df, ewm_span=14):
    logging.info("Pillar 1: Engineering multi-domain rolling velocity tensors...")
    df = df.sort_values(by=['user', 'day']).copy()
    
    # 1. Base Historical Baseline Calculations (Logins, USB, Web, and Files)
    df['ewm_mean_logins'] = df.groupby('user')['total_logins'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_logins'] = df.groupby('user')['total_logins'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    
    df['ewm_mean_usb'] = df.groupby('user')['usb_file_copies'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_usb'] = df.groupby('user')['usb_file_copies'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    
    df['ewm_mean_web'] = df.groupby('user')['web_clicks'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_web'] = df.groupby('user')['web_clicks'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    
    df['ewm_mean_file'] = df.groupby('user')['file_touches'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_file'] = df.groupby('user')['file_touches'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    
    # 2. Extract Native Deviations (Z-Scores)
    df['login_deviation'] = (df['total_logins'] - df['ewm_mean_logins']) / df['ewm_std_logins']
    df['usb_deviation'] = (df['usb_file_copies'] - df['ewm_mean_usb']) / df['ewm_std_usb']
    df['web_deviation'] = (df['web_clicks'] - df['ewm_mean_web']) / df['ewm_std_web']
    df['file_deviation'] = (df['file_touches'] - df['ewm_mean_file']) / df['ewm_std_file']
    df['after_hours_velocity'] = df['after_hours_logins']
    
    replace_cols = ['login_deviation', 'usb_deviation', 'web_deviation', 'file_deviation']
    df[replace_cols] = df[replace_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Multi-Day Temporal Memory Velocity Cascades
    df['login_dev_3d_velocity'] = df.groupby('user')['login_deviation'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df['usb_dev_3d_velocity'] = df.groupby('user')['usb_deviation'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df['usb_dev_7d_velocity'] = df.groupby('user')['usb_deviation'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    df['after_hours_3d_velocity'] = df.groupby('user')['after_hours_velocity'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df['web_dev_3d_velocity'] = df.groupby('user')['web_deviation'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df['web_dev_7d_velocity'] = df.groupby('user')['web_deviation'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    df['file_dev_3d_velocity'] = df.groupby('user')['file_deviation'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df['file_dev_7d_velocity'] = df.groupby('user')['file_deviation'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    
    df.fillna(0, inplace=True)
    return df

def run_benchmarks(data_path="data/processed/cert_expanded_matrix.csv"):
    logging.info(f"Loading full expanded data matrix from {data_path}...")
    raw_df = pd.read_csv(data_path)
    df = build_advanced_features(raw_df)
    
    feature_cols = [
        'login_deviation', 'usb_deviation', 'web_deviation', 'file_deviation', 'after_hours_velocity',
        'login_dev_3d_velocity', 'usb_dev_3d_velocity', 'usb_dev_7d_velocity', 
        'after_hours_3d_velocity', 'web_dev_3d_velocity', 'web_dev_7d_velocity',
        'file_dev_3d_velocity', 'file_dev_7d_velocity'
    ]
    
    X = df[feature_cols]
    y_true = df['is_attacker'].values
    
    logging.info("Pillar 2: Executing Robust Scaling across feature dimensions...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # -------------------------------------------------------------------------
    # MODEL 1: Optimized Isolation Forest + Extreme Value Theory (EVT)
    # -------------------------------------------------------------------------
    logging.info("Running Model 1/3: Isolation Forest + EVT...")
    if_model = IsolationForest(contamination=0.01, n_estimators=400, max_features=1.0, random_state=42, n_jobs=-1)
    if_model.fit(X_scaled)
    if_scores = if_model.decision_function(X_scaled)
    
    # EVT Pareto Tail Fit
    inverted_scores = -if_scores
    t_cutoff = np.quantile(inverted_scores, 0.99)
    tail_excesses = inverted_scores[inverted_scores > t_cutoff] - t_cutoff
    shape, loc, scale = stats.genpareto.fit(tail_excesses)
    
    risk_alpha = 0.015
    n = len(if_scores)
    N_u = len(tail_excesses)
    calculated_excess = t_cutoff + (scale / shape) * (((risk_alpha * (n / N_u)) ** (-shape)) - 1)
    
    y_pred_evt = (if_scores <= -calculated_excess).astype(int)
    
    # -------------------------------------------------------------------------
    # MODEL 2: One-Class Support Vector Machine (OC-SVM)
    # -------------------------------------------------------------------------
    logging.info("Running Model 2/3: One-Class SVM (RBF Kernel)...")
    # Using a 1% contamination budget for consistency
    oc_svm = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
    # Sampling subset for memory/speed protection if needed, otherwise full fit
    y_pred_ocsvm = oc_svm.fit_predict(X_scaled)
    y_pred_ocsvm = np.where(y_pred_ocsvm == -1, 1, 0) # Convert -1 mapping to anomaly flag (1)
    
    # -------------------------------------------------------------------------
    # MODEL 3: Local Outlier Factor (LOF)
    # -------------------------------------------------------------------------
    logging.info("Running Model 3/3: Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, n_jobs=-1)
    y_pred_lof = lof.fit_predict(X_scaled)
    y_pred_lof = np.where(y_pred_lof == -1, 1, 0)
    
    # --- EVALUATION ENGINE ---
    models = {
        "Isolation Forest + EVT": y_pred_evt,
        "One-Class SVM (RBF)": y_pred_ocsvm,
        "Local Outlier Factor (LOF)": y_pred_lof
    }
    
    print("\n" + "="*85)
    print("         HARVARD ACADEMIC BENCHMARK REPORT: CROSS-MODEL EVALUATION")
    print("="*85)
    print(f"{'Classifier Model':<30}{'Precision':<12}{'Recall':<10}{'F1-Score':<12}{'TP':<8}{'FP':<8}")
    print("-"*85)
    
    for name, preds in models.items():
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds)
        tp = cm[1][1]
        fp = cm[0][1]
        print(f"{name:<30}{prec:<12.3f}{rec:<10.3f}{f1:<12.3f}{tp:<8}{fp:<8}")
        
    print("="*85 + "\n")

if __name__ == "__main__":
    run_benchmarks()