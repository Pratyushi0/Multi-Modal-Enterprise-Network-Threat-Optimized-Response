import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

class PhDValidationEngine:
    def __init__(self, risk_alpha=0.0005):
        self.scaler = RobustScaler()
        self.model = IsolationForest(contamination=0.01, n_estimators=300, random_state=42, n_jobs=-1)
        self.risk_alpha = risk_alpha

    def calibrate_base(self, X_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        
    def fit_evt_boundary(self, X_val):
        """
        Fits a Generalized Pareto Distribution to the tail end of the anomaly scores.
        """
        X_scaled = self.scaler.transform(X_val)
        scores = self.model.decision_function(X_scaled)
        
        # Invert scores: lower isolation forest values indicate high risk
        inverted_magnitudes = -scores
        cutoff_quantile = np.quantile(inverted_magnitudes, 0.97)
        tail_excess = inverted_magnitudes[inverted_magnitudes > cutoff_quantile] - cutoff_quantile
        
        # Fit Pareto distribution params: [shape, location, scale]
        c, loc, scale = stats.genpareto.fit(tail_excess)
        
        # Compute mathematical anomaly threshold
        n = len(scores)
        N_u = len(tail_excess)
        evt_cutoff = cutoff_quantile + (scale / c) * (((self.risk_alpha * (n / N_u)) ** (-c)) - 1)
        
        return -evt_cutoff  # Return to native Isolation Forest format