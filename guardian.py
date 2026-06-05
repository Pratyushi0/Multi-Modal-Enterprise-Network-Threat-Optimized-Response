import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

print("Step 1: Loading dataset...")
df = pd.read_csv("employee_activity.csv")

features = ['total_logins', 'after_hours_logins', 'usb_file_copies']
X = df[features]

print("Step 2: Scaling features using Robust Median Math...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print("Step 3: Extracting Raw Structural Anomaly Scores...")
# We use a wider contamination pool to train the internal trees thoroughly
model = IsolationForest(contamination=0.01, n_estimators=300, random_state=42)
model.fit(X_scaled)

# decision_function returns raw scores. Negative scores mean highly anomalous!
df['anomaly_score'] = model.decision_function(X_scaled)

# PRODUCTION OVERRIDE: Instead of letting the AI blindly choose,
# we sort the entire company by the most severe anomaly scores and flag the top 10 rows.
flagged_alerts = df.nsmallest(10, 'anomaly_score')

print("\n" + "="*80)
print("             AI DIGITAL GUARDIAN ALERTS (MANUAL SCORE OVERRIDE)        ")
print("="*80)
print(flagged_alerts[['user', 'day', 'total_logins', 'after_hours_logins', 'usb_file_copies', 'anomaly_score', 'is_attacker']])
print("="*80)

correct_detections = flagged_alerts['is_attacker'].sum()
false_positives = len(flagged_alerts) - correct_detections

print(f"\n[EVALUATION] The AI isolated the top {len(flagged_alerts)} highest-risk events in the company.")
print(f"[EVALUATION] -> Successfully caught {correct_detections} out of 2 actual hidden threats!")
print(f"[EVALUATION] -> Generated {false_positives} verification alerts for human analyst review.")

if correct_detections == 2:
    print("\n👑 NETWORK SECURED: The score override successfully dragged both hackers out of the dark!")