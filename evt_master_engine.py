import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging
import os

# INTEGRATION: Import the multi-stream data aggregation engine
from multi_stream_aggregator import run_multi_domain_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MASTER-ENGINE] - %(levelname)s - %(message)s')

class PhDEmsembleEngine:
    def __init__(self, nu_svm=0.01, contam_if=0.01, risk_alpha=0.015):
        self.nu_svm = nu_svm
        self.contam_if = contam_if
        self.risk_alpha = risk_alpha
        self.scaler = RobustScaler()
        
        # Initialize both distinct structural model types
        self.svm_model = OneClassSVM(kernel='rbf', nu=self.nu_svm, gamma='scale')
        self.if_model = IsolationForest(contamination=self.contam_if, n_estimators=400, max_features=1.0, random_state=42, n_jobs=-1)

    def execute_contextual_baselining(self, df, ewm_span=14):
        logging.info("Pillar 1: Extracting multi-domain historical behavioral trends...")
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

    def compute_evt_threshold(self, anomaly_scores):
        inverted_magnitudes = -anomaly_scores
        t_cutoff = np.quantile(inverted_magnitudes, 1.0 - self.contam_if)
        tail_excesses = inverted_magnitudes[inverted_magnitudes > t_cutoff] - t_cutoff
        
        shape, loc, scale = stats.genpareto.fit(tail_excesses)
        n = len(anomaly_scores)
        N_u = len(tail_excesses)
        calculated_excess = t_cutoff + (scale / shape) * (((self.risk_alpha * (n / N_u)) ** (-shape)) - 1)
        
        return -calculated_excess

    def run_pipeline(self, data_path):
        raw_df = pd.read_csv(data_path)
        processed_df = self.execute_contextual_baselining(raw_df)
        
        feature_cols = [
            'login_deviation', 'usb_deviation', 'web_deviation', 'file_deviation', 'after_hours_velocity',
            'login_dev_3d_velocity', 'usb_dev_3d_velocity', 'usb_dev_7d_velocity', 
            'after_hours_3d_velocity', 'web_dev_3d_velocity', 'web_dev_7d_velocity',
            'file_dev_3d_velocity', 'file_dev_7d_velocity'
        ]
        X = processed_df[feature_cols]
        
        logging.info("Pillar 2: Normalizing multi-domain features via RobustScaler...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Engine 1: One-Class SVM Execution
        logging.info("Pillar 3A: Fitting non-linear SVM Hyperplane topology...")
        svm_preds = self.svm_model.fit_predict(X_scaled)
        svm_anomalies = np.where(svm_preds == -1, 1, 0)
        
        # Engine 2: Isolation Forest + EVT Execution
        logging.info("Pillar 3B: Fitting Isolation Forest Forest topology...")
        self.if_model.fit(X_scaled)
        if_scores = self.if_model.decision_function(X_scaled)
        evt_threshold = self.compute_evt_threshold(if_scores)
        if_anomalies = (if_scores <= evt_threshold).astype(int)
        
        # CONSENSUS ENSEMBLE LAYER: Trigger flag ONLY if both models agree
        logging.info("Ensemble Layer: Running intersection logic over model boundaries...")
        processed_df['predicted_attacker'] = np.bitwise_and(svm_anomalies, if_anomalies)
        
        print("\n" + "="*70)
        print("          HARVARD ACADEMIC EVALUATION METRICS REPORT (CONSENSUS ENSEMBLE)")
        print("="*70)
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(processed_df['is_attacker'], processed_df['predicted_attacker'], target_names=['Normal', 'Attacker']))
        
        print("CONFUSION MATRIX:")
        cm = confusion_matrix(processed_df['is_attacker'], processed_df['predicted_attacker'])
        print(f" -> True Negatives  (Innocents Confirmed): {cm[0][0]}")
        print(f" -> False Positives (System Noise Fatigue): {cm[0][1]}")
        print(f" -> False Negatives (Missed Threat Surface): {cm[1][0]}")
        print(f" -> True Positives  (Threats Successfully Intercepted): {cm[1][1]}")
        print("="*70 + "\n")

if __name__ == "__main__":
    ZIP_ARCHIVE = "data/raw/dataset.zip"
    LABELED_MATRIX = "data/processed/cert_labeled_matrix.csv"
    EXPANDED_MATRIX = "data/processed/cert_expanded_matrix.csv"
    
    print("⚡ Starting Integrated Consensus Ensemble Threat Pipeline...")
    
    # Check data status automatically
    update_success = run_multi_domain_pipeline(ZIP_ARCHIVE, LABELED_MATRIX, EXPANDED_MATRIX)
    
    if update_success and os.path.exists(EXPANDED_MATRIX):
        engine = PhDEmsembleEngine(nu_svm=0.01, contam_if=0.01, risk_alpha=0.015)
        engine.run_pipeline(EXPANDED_MATRIX)
    else:
        logging.critical("Pipeline initialization stopped due to missing data assets.")