import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [HYPER-TUNER] - %(levelname)s - %(message)s')

def build_features_in_place(df, ewm_span=14):
    logging.info("Pillar 1: Re-generating multi-day contextual velocity tensors for tuning space...")
    df = df.sort_values(by=['user', 'day']).copy()
    
    # 1. Base Historical Baseline Calculations
    df['ewm_mean_logins'] = df.groupby('user')['total_logins'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_logins'] = df.groupby('user')['total_logins'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    df['ewm_mean_usb'] = df.groupby('user')['usb_file_copies'].transform(lambda x: x.ewm(span=self.ewm_span if 'self' in locals() else ewm_span).mean())
    df['ewm_std_usb'] = df.groupby('user')['usb_file_copies'].transform(lambda x: x.ewm(span=self.ewm_span if 'self' in locals() else ewm_span).std()).fillna(1.0)
    df['ewm_mean_web'] = df.groupby('user')['web_clicks'].transform(lambda x: x.ewm(span=ewm_span).mean())
    df['ewm_std_web'] = df.groupby('user')['web_clicks'].transform(lambda x: x.ewm(span=ewm_span).std()).fillna(1.0)
    
    # 2. Extract Native Deviations
    df['login_deviation'] = (df['total_logins'] - df['ewm_mean_logins']) / df['ewm_std_logins']
    df['usb_deviation'] = (df['usb_file_copies'] - df['ewm_mean_usb']) / df['ewm_std_usb']
    df['web_deviation'] = (df['web_clicks'] - df['ewm_mean_web']) / df['ewm_std_web']
    df['after_hours_velocity'] = df['after_hours_logins']
    
    replace_cols = ['login_deviation', 'usb_deviation', 'web_deviation']
    df[replace_cols] = df[replace_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Temporal Memory Velocity Cascades
    df['login_dev_3d_velocity'] = df.groupby('user')['login_deviation'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df['usb_dev_3d_velocity'] = df.groupby('user')['usb_deviation'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df['usb_dev_7d_velocity'] = df.groupby('user')['usb_deviation'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    df['after_hours_3d_velocity'] = df.groupby('user')['after_hours_velocity'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df['web_dev_3d_velocity'] = df.groupby('user')['web_deviation'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df['web_dev_7d_velocity'] = df.groupby('user')['web_deviation'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    
    df.fillna(0, inplace=True)
    return df

def evaluate_hyperparameters(data_path="data/processed/cert_expanded_matrix.csv"):
    logging.info(f"Loading raw multi-modal dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    
    # Inject feature space structures before tuning
    df = build_features_in_place(raw_df)
    
    feature_cols = [
        'login_deviation', 'usb_deviation', 'web_deviation', 'after_hours_velocity',
        'login_dev_3d_velocity', 'usb_dev_3d_velocity', 'usb_dev_7d_velocity', 
        'after_hours_3d_velocity', 'web_dev_3d_velocity', 'web_dev_7d_velocity'
    ]
    
    X = df[feature_cols]
    y_true = df['is_attacker'].values
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Grid optimization search parameters
    n_trees_options = [200, 400]
    contamination_options = [0.01, 0.02]
    alpha_options = [0.005, 0.015]
    
    search_space = list(itertools.product(n_trees_options, contamination_options, alpha_options))
    results = []
    
    print("\n" + "="*80)
    print("          STARTING PhD HYPERPARAMETER GRID VALIDATION SWEEP")
    print("="*80)
    print(f"{'Trees':<8}{'Contam':<10}{'Alpha':<10}{'Precision':<12}{'Recall':<10}{'F1-Score':<10}")
    print("-"*80)
    
    for n_trees, contam, alpha in search_space:
        clf = IsolationForest(contamination=contam, n_estimators=n_trees, max_features=1.0, random_state=42, n_jobs=-1)
        clf.fit(X_scaled)
        
        scores = clf.decision_function(X_scaled)
        inverted_scores = -scores
        
        t_cutoff = np.quantile(inverted_scores, 1.0 - contam)
        tail_excesses = inverted_scores[inverted_scores > t_cutoff] - t_cutoff
        
        shape, loc, scale = stats.genpareto.fit(tail_excesses)
        n = len(scores)
        N_u = len(tail_excesses)
        
        calculated_excess = t_cutoff + (scale / shape) * (((alpha * (n / N_u)) ** (-shape)) - 1)
        evt_threshold = -calculated_excess
        
        y_pred = (scores <= evt_threshold).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"{n_trees:<8}{contam:<10.3f}{alpha:<10.3f}{prec:<12.3f}{rec:<10.3f}{f1:<10.3f}")
        results.append({'trees': n_trees, 'contam': contam, 'alpha': alpha, 'f1': f1, 'prec': prec, 'rec': rec})
        
    print("="*80)
    best_run = max(results, key=lambda x: x['f1'])
    print(f"OPTIMAL CONFIGURATION FOUND:\n -> Trees: {best_run['trees']}\n -> Contamination: {best_run['contam']}\n -> Alpha: {best_run['alpha']}\n -> Peak F1-Score: {best_run['f1']:.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    evaluate_hyperparameters()