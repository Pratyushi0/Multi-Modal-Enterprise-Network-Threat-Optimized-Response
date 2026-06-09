"""
src/features.py
---------------
PROJECT MENTOR — Canonical Feature Engineering

Single source of truth for ALL feature engineering across the pipeline.
Every script imports from here. No copy-pasting.

Features produced
-----------------
Deviation z-scores (EWM-normalised, per user):
    login_deviation, usb_deviation, web_deviation, file_deviation

Temporal velocities (rolling sum of deviations, per user):
    {signal}_dev_{3d|7d}_velocity   — 8 features

After-hours signal:
    after_hours_velocity

NEW — Cross-modal composite features:
    multi_channel_spike    : geometric mean of |deviations| (flags co-occurring spikes)
    spike_ratio_logins     : today / personal 90th-pct (detects burst events)
    spike_ratio_usb        : same for USB
    spike_ratio_web        : same for web
    spike_ratio_file       : same for file

NEW — Temporal encoding (avoids string sort problems):
    day_of_week_sin, day_of_week_cos  (cyclical, no ordinal artifact)
    is_weekend                         (binary)

Total: 13 original + 8 new = 21 features
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Public constant: ordered feature list used by all models ─────────────────
FEATURE_COLS: list[str] = [
    # EWM deviation z-scores
    "login_deviation",
    "usb_deviation",
    "web_deviation",
    "file_deviation",
    # After-hours raw signal
    "after_hours_velocity",
    # 3-day rolling velocities
    "login_dev_3d_velocity",
    "usb_dev_3d_velocity",
    "after_hours_3d_velocity",
    "web_dev_3d_velocity",
    "file_dev_3d_velocity",
    # 7-day rolling velocities
    "usb_dev_7d_velocity",
    "web_dev_7d_velocity",
    "file_dev_7d_velocity",
    # Cross-modal composite
    "multi_channel_spike",
    # Personal spike ratios
    "spike_ratio_logins",
    "spike_ratio_usb",
    "spike_ratio_web",
    "spike_ratio_file",
    # Temporal patterns
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
]


def build_features(df: pd.DataFrame, ewm_span: int = 14) -> pd.DataFrame:
    """
    Engineer all features from a raw aggregated matrix.

    Parameters
    ----------
    df       : DataFrame with columns [day, user, total_logins,
               after_hours_logins, usb_file_copies, web_clicks, file_touches]
    ewm_span : Half-life for exponentially weighted moving statistics.
               14 days corresponds to ~2-week behavioural memory.

    Returns
    -------
    df with FEATURE_COLS added in-place (copy returned).

    Notes
    -----
    - sort_values on 'day' is safe only if day is ISO-format string (YYYY-MM-DD)
      or a date/datetime object.  Both aggregation_pipeline.py and
      multi_stream_aggregator.py produce ISO strings via .dt.date.astype(str).
    - EWM std is clipped to 1e-6 BEFORE division to prevent inf/NaN propagation.
    - All velocity lambdas use the `win=w` default-arg pattern to avoid
      closure-over-loop-variable bugs.
    """
    df = df.sort_values(["user", "day"]).copy()

    # ── Ensure required raw columns exist ───────────────────────────────────
    for col in ("total_logins", "usb_file_copies", "web_clicks",
                "file_touches", "after_hours_logins"):
        if col not in df.columns:
            log.warning("Column '%s' missing — filling with 0.", col)
            df[col] = 0

    # ── EWM baselines ────────────────────────────────────────────────────────
    _signal_map = [
        ("total_logins",    "logins"),
        ("usb_file_copies", "usb"),
        ("web_clicks",      "web"),
        ("file_touches",    "file"),
    ]
    for col, alias in _signal_map:
        df[f"ewm_mean_{alias}"] = (
            df.groupby("user")[col]
            .transform(lambda x: x.ewm(span=ewm_span).mean())
        )
        df[f"ewm_std_{alias}"] = (
            df.groupby("user")[col]
            .transform(lambda x: x.ewm(span=ewm_span).std())
            .fillna(1.0)
            .clip(lower=1e-6)          # ← prevents division-by-zero
        )

    # ── Per-user deviation z-scores ──────────────────────────────────────────
    df["login_deviation"] = (
        (df["total_logins"]    - df["ewm_mean_logins"]) / df["ewm_std_logins"]
    )
    df["usb_deviation"]   = (
        (df["usb_file_copies"] - df["ewm_mean_usb"])    / df["ewm_std_usb"]
    )
    df["web_deviation"]   = (
        (df["web_clicks"]      - df["ewm_mean_web"])    / df["ewm_std_web"]
    )
    df["file_deviation"]  = (
        (df["file_touches"]    - df["ewm_mean_file"])   / df["ewm_std_file"]
    )
    df["after_hours_velocity"] = df["after_hours_logins"].astype(float)

    _dev_cols = ["login_deviation", "usb_deviation",
                 "web_deviation",   "file_deviation"]
    df[_dev_cols] = df[_dev_cols].replace([np.inf, -np.inf], 0).fillna(0)

    # ── Multi-day temporal velocity cascades ─────────────────────────────────
    # IMPORTANT: use `win=w` default arg to avoid closure-over-loop-variable.
    _velocity_spec = [
        ("login_deviation",      3, "login_dev_3d_velocity"),
        ("usb_deviation",        3, "usb_dev_3d_velocity"),
        ("usb_deviation",        7, "usb_dev_7d_velocity"),
        ("after_hours_velocity", 3, "after_hours_3d_velocity"),
        ("web_deviation",        3, "web_dev_3d_velocity"),
        ("web_deviation",        7, "web_dev_7d_velocity"),
        ("file_deviation",       3, "file_dev_3d_velocity"),
        ("file_deviation",       7, "file_dev_7d_velocity"),
    ]
    for src, w, dest in _velocity_spec:
        df[dest] = (
            df.groupby("user")[src]
            .transform(lambda x, win=w: x.rolling(win, min_periods=1).sum())
        )

    # ── Cross-modal composite spike ──────────────────────────────────────────
    # Geometric mean of absolute deviations — high only when MULTIPLE channels
    # deviate simultaneously (the insider threat signature).
    abs_devs = (
        df[["login_deviation", "usb_deviation",
            "web_deviation",   "file_deviation"]]
        .abs()
        .clip(lower=0)
    )
    # Add 1 before geometric mean to avoid zero-product collapse
    df["multi_channel_spike"] = (
        (abs_devs + 1.0)
        .prod(axis=1)
        .pow(1.0 / 4.0)
        - 1.0
    )

    # ── Personal spike ratios ────────────────────────────────────────────────
    # today's value / (personal 90th percentile + ε)
    # Detects burst events that EWM may smooth over.
    _pct90_cols = [
        ("total_logins",    "spike_ratio_logins"),
        ("usb_file_copies", "spike_ratio_usb"),
        ("web_clicks",      "spike_ratio_web"),
        ("file_touches",    "spike_ratio_file"),
    ]
    for col, dest in _pct90_cols:
        personal_p90 = (
            df.groupby("user")[col]
            .transform(lambda x: x.expanding(min_periods=1).quantile(0.90))
            .clip(lower=1e-6)
        )
        df[dest] = (df[col] / personal_p90).clip(upper=50.0)  # cap extreme ratios

    # ── Temporal encoding ────────────────────────────────────────────────────
    # Parse day column; if it fails, day_of_week defaults to 0.
    try:
        day_dt = pd.to_datetime(df["day"], errors="coerce")
        dow = day_dt.dt.dayofweek.fillna(0).astype(float)  # 0=Mon … 6=Sun
        df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)
        df["is_weekend"]      = (dow >= 5).astype(float)
    except Exception:
        log.warning("Could not parse 'day' for temporal encoding — using zeros.")
        df["day_of_week_sin"] = 0.0
        df["day_of_week_cos"] = 0.0
        df["is_weekend"]      = 0.0

    df.fillna(0, inplace=True)
    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split.

    Splits on unique calendar days (not rows) so the 70/30 ratio applies
    to the time axis, not the user axis.  This is the correct split for
    sequential behavioural data and avoids future-leaking into training.

    Parameters
    ----------
    df         : full feature matrix with a 'day' column (ISO string)
    train_frac : fraction of days used for training

    Returns
    -------
    (train_df, test_df)  — test contains the latest (1 - train_frac) days
    """
    sorted_days = sorted(df["day"].unique())
    cutoff_idx  = int(len(sorted_days) * train_frac)
    cutoff_day  = sorted_days[cutoff_idx]

    train_df = df[df["day"] <  cutoff_day].copy()
    test_df  = df[df["day"] >= cutoff_day].copy()

    log.info(
        "Temporal split: train=%d rows (%d days) | test=%d rows (%d days) | "
        "cutoff=%s",
        len(train_df), cutoff_idx,
        len(test_df),  len(sorted_days) - cutoff_idx,
        cutoff_day,
    )
    return train_df, test_df