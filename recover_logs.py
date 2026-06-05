import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os

# Define file saving destination
LOG_OUTPUT_PATH = "reports/phd_metrics_evolution.txt"
os.makedirs(os.path.dirname(LOG_OUTPUT_PATH), exist_ok=True)

print("⏳ Running comprehensive simulation sweep across all 5 research phases...")
raw_df = pd.read_csv("data/processed/cert_expanded_matrix.csv")

# --- PILLAR 1: ENGINE THE DEVIATIONS IN MEMORY ---
def build_advanced_features(df, ewm_span=14):
    df = df.sort_values(by=['user', 'day']).copy()
    
    # Base Historical Baseline Calculations
    df['ewm_mean_logins'] = df.groupby('user')['total_logins'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_logins'] = df.groupby('user')['total_logins'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    
    df['ewm_mean_usb'] = df.groupby('user')['usb_file_copies'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_usb'] = df.groupby('user')['usb_file_copies'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    
    df['ewm_mean_web'] = df.groupby('user')['web_clicks'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_web'] = df.groupby('user')['web_clicks'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    
    df['ewm_mean_file'] = df.groupby('user')['file_touches'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_file'] = df.groupby('user')['file_touches'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    
    # Extract Native Deviations
    df['login_deviation'] = (df['total_logins'] - df['ewm_mean_logins']) / df['ewm_std_logins']
    df['usb_deviation'] = (df['usb_file_copies'] - df['ewm_mean_usb']) / df['ewm_std_usb']
    df['web_deviation'] = (df['web_clicks'] - df['ewm_mean_web']) / df['ewm_std_web']
    df['file_deviation'] = (df['file_touches'] - df['ewm_mean_file']) / df['ewm_std_file']
    df['after_hours_velocity'] = df['after_hours_logins']
    
    replace_cols = ['login_deviation', 'usb_deviation', 'web_deviation', 'file_deviation']
    df[replace_cols] = df[replace_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Multi-Day Temporal Cascades
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

# Calculate features in memory
df = build_advanced_features(raw_df)

# Define feature spaces
base_features = [
    'login_deviation', 'usb_deviation', 'web_deviation', 'after_hours_velocity',
    'login_dev_3d_velocity', 'usb_dev_3d_velocity', 'usb_dev_7d_velocity', 
    'after_hours_3d_velocity', 'web_dev_3d_velocity', 'web_dev_7d_velocity'
]
expanded_features = base_features + ['file_deviation', 'file_dev_3d_velocity', 'file_dev_7d_velocity']

y_true = df['is_attacker'].values
scaler = RobustScaler()
log_lines = []

def log_and_print(text_line):
    print(text_line)
    log_lines.append(text_line)

log_and_print("="*95)
log_and_print("         HARVARD DISSERTATION RESEARCH: HISTORICAL PARADIGM PERFORMANCE LOG")
log_and_print("="*95)
log_and_print(f"{'Architecture Evolutionary Phase':<35} | Precision | Recall | F1-Score | True Pos  | False Pos")
log_and_print("-"*95)

def evaluate_and_log_phase(phase_name, y_pred):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1][1]
    fp = cm[0][1]
    row_string = f"{phase_name:<35} | {prec:<9.3f} | {rec:<6.3f} | {f1:<8.3f} | {tp:<9,d} | {fp:<9,d}"
    log_and_print(row_string)

# --- PHASE 1: UNTUNED BASELINE ---
X_p1 = scaler.fit_transform(df[base_features])
if_p1 = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1).fit(X_p1)
preds_p1 = np.where(if_p1.predict(X_p1) == -1, 1, 0)
evaluate_and_log_phase("Phase 1: Untuned Baseline", preds_p1)

# --- PHASE 2: HYPERPARAMETER TUNED ---
if_p2 = IsolationForest(contamination=0.01, n_estimators=400, random_state=42, n_jobs=-1).fit(X_p1)
scores_p2 = if_p2.decision_function(X_p1)
inv_scores = -scores_p2
t_cutoff = np.quantile(inv_scores, 0.99)
tail_excesses = inv_scores[inv_scores > t_cutoff] - t_cutoff
shape, loc, scale = stats.genpareto.fit(tail_excesses)
calc_excess = t_cutoff + (scale / shape) * (((0.015 * (len(scores_p2) / len(tail_excesses))) ** (-shape)) - 1)
preds_p2 = (scores_p2 <= -calc_excess).astype(int)
evaluate_and_log_phase("Phase 2: Hyperparameter Tuned", preds_p2)

# --- PHASE 3: EXPANDED MULTI-MODAL ---
X_p3 = scaler.fit_transform(df[expanded_features])
if_p3 = IsolationForest(contamination=0.01, n_estimators=400, random_state=42, n_jobs=-1).fit(X_p3)
scores_p3 = if_p3.decision_function(X_p3)
inv_scores_p3 = -scores_p3
t_cutoff_p3 = np.quantile(inv_scores_p3, 0.99)
tail_excesses_p3 = inv_scores_p3[inv_scores_p3 > t_cutoff_p3] - t_cutoff_p3
shape_p3, loc_p3, scale_p3 = stats.genpareto.fit(tail_excesses_p3)
calc_excess_p3 = t_cutoff_p3 + (scale_p3 / shape_p3) * (((0.015 * (len(scores_p3) / len(tail_excesses_p3))) ** (-shape_p3)) - 1)
preds_p3 = (scores_p3 <= -calc_excess_p3).astype(int)
evaluate_and_log_phase("Phase 3: Expanded Multi-Modal", preds_p3)

# --- PHASE 4: OPTIMIZED CLASSIFIER (OC-SVM) ---
svm_p4 = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale').fit(X_p3)
preds_p4 = np.where(svm_p4.predict(X_p3) == -1, 1, 0)
evaluate_and_log_phase("Phase 4: Optimized Classifier", preds_p4)

# --- PHASE 5: CONSENSUS ENSEMBLE ---
preds_p5 = np.bitwise_and(preds_p3, preds_p4)
evaluate_and_log_phase("Phase 5: Consensus Ensemble", preds_p5)

log_and_print("="*95)

with open(LOG_OUTPUT_PATH, "w") as f:
    f.write("\n".join(log_lines) + "\n")

print(f"\n✨ Permanent academic log asset generated completely at: {LOG_OUTPUT_PATH}")