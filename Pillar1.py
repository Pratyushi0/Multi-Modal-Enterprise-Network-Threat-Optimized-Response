"""
pillar1_ewma.py
---------------
Standalone Pillar 1 feature engineering module for PROJECT MENTOR.

Provides a single importable function `build_features` that computes:
  - Per-user EWM baselines (mean + std) for all 4 signal domains
  - Z-score deviations relative to individual user history
  - Multi-day temporal velocity cascades (3-day and 7-day rolling sums)

Used by:
    evt_master_engine.py    (via execute_contextual_baselining — same logic)
    hypertuner.py           (build_features import)
    guardian.py             (build_features import)
    Pillar2_3_validator.py  (build_features import)
    visualizer.py           (build_features import)

All files previously duplicated this logic internally. Import from here instead.

Quick standalone use (EDA):
    from pillar1_ewma import build_features
    df = build_features(pd.read_csv("data/processed/cert_expanded_matrix.csv"))
"""

import numpy as np
import pandas as pd

# EWM span used across the entire pipeline — change here to affect all modules
DEFAULT_EWM_SPAN = 14

# Velocity windows applied to each deviation signal
VELOCITY_SPEC = [
    ("login_deviation",      3, "login_dev_3d_velocity"),
    ("usb_deviation",        3, "usb_dev_3d_velocity"),
    ("usb_deviation",        7, "usb_dev_7d_velocity"),
    ("after_hours_velocity", 3, "after_hours_3d_velocity"),
    ("web_deviation",        3, "web_dev_3d_velocity"),
    ("web_deviation",        7, "web_dev_7d_velocity"),
    ("file_deviation",       3, "file_dev_3d_velocity"),
    ("file_deviation",       7, "file_dev_7d_velocity"),
]

# All feature columns produced by build_features — importable by other modules
FEATURE_COLS = [
    "login_deviation",
    "usb_deviation",
    "web_deviation",
    "file_deviation",
    "after_hours_velocity",
    "login_dev_3d_velocity",
    "usb_dev_3d_velocity",
    "usb_dev_7d_velocity",
    "after_hours_3d_velocity",
    "web_dev_3d_velocity",
    "web_dev_7d_velocity",
    "file_dev_3d_velocity",
    "file_dev_7d_velocity",
]


def build_features(df: pd.DataFrame, ewm_span: int = DEFAULT_EWM_SPAN) -> pd.DataFrame:
    """
    Compute all Pillar 1 behavioural features in-place on a copy of df.

    Parameters
    ----------
    df       : DataFrame containing at minimum:
                 user, day, total_logins, after_hours_logins,
                 usb_file_copies, web_clicks, file_touches
    ewm_span : exponentially weighted moving average span (default 14 days)

    Returns
    -------
    DataFrame with all FEATURE_COLS appended.
    """
    df = df.sort_values(by=["user", "day"]).copy()

    # ------------------------------------------------------------------
    # 1. Per-user EWM baselines
    # ------------------------------------------------------------------
    signal_map = [
        ("total_logins",    "logins"),
        ("usb_file_copies", "usb"),
        ("web_clicks",      "web"),
        ("file_touches",    "file"),
    ]
    for col, alias in signal_map:
        df[f"ewm_mean_{alias}"] = df.groupby("user")[col].transform(
            lambda x: x.ewm(span=ewm_span).mean()
        )
        df[f"ewm_std_{alias}"] = (
            df.groupby("user")[col]
            .transform(lambda x: x.ewm(span=ewm_span).std())
            .fillna(1.0)
            .clip(lower=1e-6)       # prevents division by zero for constant users
        )

    # ------------------------------------------------------------------
    # 2. Z-score deviations relative to individual user history
    # ------------------------------------------------------------------
    df["login_deviation"] = (
        (df["total_logins"]    - df["ewm_mean_logins"]) / df["ewm_std_logins"]
    )
    df["usb_deviation"] = (
        (df["usb_file_copies"] - df["ewm_mean_usb"])    / df["ewm_std_usb"]
    )
    df["web_deviation"] = (
        (df["web_clicks"]      - df["ewm_mean_web"])    / df["ewm_std_web"]
    )
    df["file_deviation"] = (
        (df["file_touches"]    - df["ewm_mean_file"])   / df["ewm_std_file"]
    )
    df["after_hours_velocity"] = df["after_hours_logins"].astype(float)

    dev_cols = [
        "login_deviation", "usb_deviation",
        "web_deviation",   "file_deviation",
    ]
    df[dev_cols] = df[dev_cols].replace([np.inf, -np.inf], 0).fillna(0)

    # ------------------------------------------------------------------
    # 3. Multi-day temporal velocity cascades
    # ------------------------------------------------------------------
    for src, window, dest in VELOCITY_SPEC:
        df[dest] = df.groupby("user")[src].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).sum()
        )

    df.fillna(0, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Backwards-compatible alias for the original function name in Pillar1.py
# ---------------------------------------------------------------------------
def run_contextual_ewma(matrix_path: str, ewm_span: int = DEFAULT_EWM_SPAN) -> pd.DataFrame:
    """
    Original Pillar1.py interface — loads CSV from path and returns
    a feature-engineered DataFrame.

    Prefer calling build_features(df) directly when you already have
    the DataFrame in memory.
    """
    df = pd.read_csv(matrix_path)
    return build_features(df, ewm_span=ewm_span)
