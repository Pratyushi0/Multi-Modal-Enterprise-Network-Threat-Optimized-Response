import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# 1. Load Data & Run Model
df = pd.read_csv("employee_activity.csv")
features = ['total_logins', 'after_hours_logins', 'usb_file_copies']
X = df[features]

model = IsolationForest(contamination=0.01, n_estimators=100, random_state=42)
df['ai_prediction'] = model.fit_predict(X)

# 2. Categorize data points for clear visibility
df['visualization_group'] = 'Normal Employee'
df.loc[df['ai_prediction'] == -1, 'visualization_group'] = 'False Positive (AI Mistake)'
df.loc[df['is_attacker'] == 1, 'visualization_group'] = 'Missed Threat'
df.loc[(df['ai_prediction'] == -1) & (df['is_attacker'] == 1), 'visualization_group'] = 'Caught Threat'

# 3. Print Behavior Breakdown Analysis
print("\n" + "="*60)
print("             BEHAVIOR PATTERN DIAGNOSTIC REPORTS              ")
print("="*60)
for name, group in df.groupby('visualization_group'):
    print(f"\nGroup: {name} (Total Events: {len(group)})")
    print(f" -> Avg Logins: {group['total_logins'].mean():.1f}")
    print(f" -> Max After-Hours Logins: {group['after_hours_logins'].max()}")
    print(f" -> Max USB File Copies: {group['usb_file_copies'].max()}")
print("="*60)

# 4. Generate 3D Mathematical Graph
fig = plt.figure(figsize=(12, 8))
ax = fig.add_dash_unit_3d if hasattr(fig, 'add_dash_unit_3d') else fig.add_subplot(111, projection='3d')

colors = {
    'Normal Employee': '#E0E0E0',           # Light Grey
    'False Positive (AI Mistake)': '#FF9900', # Orange
    'Caught Threat': '#00CC44',              # Bright Green
    'Missed Threat': '#FF0000'               # Crimson Red
}

for group_name, group_data in df.groupby('visualization_group'):
    ax.scatter(
        group_data['total_logins'], 
        group_data['after_hours_logins'], 
        group_data['usb_file_copies'],
        c=colors[group_name],
        label=group_name,
        s=150 if 'Threat' in group_name else 25,
        alpha=1.0 if 'Threat' in group_name else 0.4,
        edgecolors='black' if 'Threat' in group_name else 'none'
    )

ax.set_title("How the AI Sees Network Behavior (3D Anomaly Mapping)", fontsize=14, pad=20)
ax.set_xlabel("Total Daily Logins")
ax.set_ylabel("After-Hours Logins")
ax.set_zlabel("USB File Copies")
ax.legend(loc='upper left')

plt.tight_layout()
print("\n[DISPLAY] Rendering 3D behavior plot window... Look at your Mac dock for the Python window!")
plt.show()
