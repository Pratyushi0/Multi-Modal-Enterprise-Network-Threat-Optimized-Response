import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import os

from multi_stream_aggregator import run_multi_domain_pipeline
from benchmarker import build_advanced_features

def generate_dissertation_plots(data_path="data/processed/cert_expanded_matrix.csv", output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("📈 Loading data and initializing visualization array...")
    
    raw_df = pd.read_csv(data_path)
    df = build_advanced_features(raw_df)
    
    feature_cols = [
        'login_deviation', 'usb_deviation', 'web_deviation', 'file_deviation', 'after_hours_velocity',
        'login_dev_3d_velocity', 'usb_dev_3d_velocity', 'usb_dev_7d_velocity', 
        'after_hours_3d_velocity', 'web_dev_3d_velocity', 'web_dev_7d_velocity',
        'file_dev_3d_velocity', 'file_dev_7d_velocity'
    ]
    
    X = df[feature_cols]
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("🌲 Fitting Isolation Forest for EVT distribution mapping...")
    if_model = IsolationForest(contamination=0.01, n_estimators=400, random_state=42, n_jobs=-1)
    if_model.fit(X_scaled)
    if_scores = if_model.decision_function(X_scaled)
    
    # -------------------------------------------------------------------------
    # PLOT 1: EXTREME VALUE THEORY PARETO TAIL ALIGNMENT
    # -------------------------------------------------------------------------
    print("🎨 Generating Plot 1: EVT Pareto Distribution Tail Curve...")
    inverted_scores = -if_scores
    t_cutoff = np.quantile(inverted_scores, 0.99)
    tail_excesses = inverted_scores[inverted_scores > t_cutoff] - t_cutoff
    
    shape, loc, scale = stats.genpareto.fit(tail_excesses)
    risk_alpha = 0.015
    n = len(if_scores)
    N_u = len(tail_excesses)
    calculated_excess = t_cutoff + (scale / shape) * (((risk_alpha * (n / N_u)) ** (-shape)) - 1)
    evt_threshold = -calculated_excess
    
    plt.figure(figsize=(10, 6))
    sns.histplot(inverted_scores, bins=100, kde=True, color='#2c3e50', stat='density', label='Sample Inverted Score Anomaly Distribution')
    
    # Plot tail fit
    x_tail = np.linspace(t_cutoff, max(inverted_scores), 200)
    y_tail = stats.genpareto.pdf(x_tail - t_cutoff, shape, loc, scale) * (N_u / n)
    plt.plot(x_tail, y_tail, color='#e74c3c', lw=3, label='Generalized Pareto Distribution (GPD) Fit')
    
    # FIXED: Added raw 'r' prefix to strings containing LaTeX symbols
    plt.axvline(t_cutoff, color='#f1c40f', linestyle='--', lw=2, label=r'Initial Threshold Cutoff ($u$)')
    plt.axvline(-evt_threshold, color='#2ecc71', linestyle='-', lw=3, label=r'Dynamic EVT Horizon Line ($\zeta_\alpha$)')
    
    plt.title('Extreme Value Theory (EVT) Tail Modeling Over Isolation Forest Output', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel(r'Inverted Anomaly Magnitude ($-s(x)$)', fontsize=10)
    plt.ylabel('Density Magnitude', fontsize=10)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot1_path = os.path.join(output_dir, "evt_gpd_tail_fit.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved Figure 1 to: {plot1_path}")

    # -------------------------------------------------------------------------
    # PLOT 2: COMPARATIVE PERFORMANCE DISSERTATION MATRIX
    # -------------------------------------------------------------------------
    print("🎨 Generating Plot 2: Performance Trade-off Metrics Comparison...")
    
    metrics_data = {
        'Model': ['iForest + EVT', 'One-Class SVM', 'Consensus Ensemble'],
        'Precision': [0.165, 0.218, 0.320],
        'Recall': [0.510, 0.532, 0.390],
        'F1-Score': [0.250, 0.309, 0.350]
    }
    metrics_df = pd.melt(pd.DataFrame(metrics_data), id_vars='Model', var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Model', y='Value', hue='Metric', palette=['#34495e', '#3498db', '#2ecc71'], edgecolor='black', linewidth=0.7)
    
    plt.title('Comparative Cyber-ML Architecture Performance Analysis', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Algorithm Paradigm Configuration', fontsize=10)
    plt.ylabel('Performance Metric Value (0.00 - 1.00)', fontsize=10)
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True)
    
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.gca().annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height + 0.02),
                                ha='center', va='center', fontsize=8, fontweight='semibold', color='#2c3e50')
            
    plot2_path = os.path.join(output_dir, "model_performance_comparison.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved Figure 2 to: {plot2_path}\n✨ All dissertation graphics rendered completely!")

if __name__ == "__main__":
    generate_dissertation_plots()