"""
visualizer.py
-------------
Generates dissertation-quality figures using live metrics from
reports/metrics.json (written by evt_master_engine.py after each run).

Run AFTER evt_master_engine.py:
    python evt_master_engine.py   # writes reports/metrics.json
    python visualizer.py          # reads it and renders figures

Outputs:
    reports/figures/evt_gpd_tail_fit.png
    reports/figures/model_performance_comparison.png
    reports/figures/score_distribution.png
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from multi_stream_aggregator import run_multi_domain_pipeline
from pillar1_ewma import build_features, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [VISUALISER] - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
PALETTE = {
    "dark":   "#2c3e50",
    "red":    "#e74c3c",
    "yellow": "#f1c40f",
    "green":  "#2ecc71",
    "blue":   "#3498db",
    "purple": "#9b59b6",
    "bg":     "#fafafa",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     ":",
    "grid.alpha":         0.5,
    "figure.facecolor":   PALETTE["bg"],
    "axes.facecolor":     PALETTE["bg"],
})

# build_features and FEATURE_COLS imported from pillar1_ewma

# ---------------------------------------------------------------------------
# Plot 1 — EVT / GPD tail fit
# ---------------------------------------------------------------------------
def _plot_evt_tail(
    if_scores: np.ndarray,
    metrics: dict,
    output_path: str,
) -> None:
    logging.info("Generating Plot 1: EVT GPD tail curve …")
    contam_rate = metrics["contamination_rate"]
    evt_thresh  = metrics["evt_threshold"]

    inverted = -if_scores
    t_cutoff = np.quantile(inverted, 1.0 - contam_rate)
    excesses = inverted[inverted > t_cutoff] - t_cutoff
    shape, loc, scale = stats.genpareto.fit(excesses)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(
        inverted, bins=120, kde=True, color=PALETTE["dark"],
        stat="density", alpha=0.55, label="Inverted IF score distribution", ax=ax,
    )

    x_tail  = np.linspace(t_cutoff, np.quantile(inverted, 0.9995), 300)
    n, N_u  = len(if_scores), len(excesses)
    y_tail  = stats.genpareto.pdf(x_tail - t_cutoff, shape, loc, scale) * (N_u / n)
    ax.plot(x_tail, y_tail, color=PALETTE["red"], lw=2.5,
            label="Generalised Pareto Distribution (GPD) fit")

    # evt_thresh is in native IF score space (negative float).
    # Inverting it gives the correct position on the -s(x) axis.
    evt_inverted = -evt_thresh
    ax.axvline(t_cutoff,     color=PALETTE["yellow"], ls="--", lw=2,
               label=r"Initial tail cutoff ($u$)")
    ax.axvline(evt_inverted, color=PALETTE["green"],  ls="-",  lw=2.5,
               label=r"EVT dynamic horizon ($\zeta_\alpha$)")

    ax.set_title(
        "Extreme Value Theory — GPD tail fit over Isolation Forest scores",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel(r"Inverted anomaly magnitude  ($-s(x)$)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(frameon=True, fontsize=9)
    ax.set_xlim(left=inverted.min())

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Per-model performance comparison (live from metrics.json)
# ---------------------------------------------------------------------------
def _plot_model_comparison(metrics: dict, output_path: str) -> None:
    logging.info("Generating Plot 2: model performance comparison …")

    breakdown = metrics["model_breakdown"]
    models    = list(breakdown.keys())
    metric_names = ["Precision", "Recall", "F1"]
    key_map      = {"Precision": "precision", "Recall": "recall", "F1": "f1"}

    x    = np.arange(len(models))
    w    = 0.22
    colours = [PALETTE["dark"], PALETTE["blue"], PALETTE["green"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, colour) in enumerate(zip(metric_names, colours)):
        vals = [breakdown[m][key_map[metric]] for m in models]
        bars = ax.bar(x + i * w, vals, width=w, label=metric,
                      color=colour, alpha=0.88, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.annotate(
                    f"{v:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015),
                    ha="center", va="bottom", fontsize=8, color=PALETTE["dark"],
                )

    ax.set_xticks(x + w)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title(
        "Per-model attacker detection performance (live from engine run)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Model configuration", fontsize=11)
    ax.set_ylabel("Performance metric value", fontsize=11)
    ax.legend(frameon=True, fontsize=9)

    # Annotate TP/FP/FN for ensemble
    ens     = breakdown["Ensemble"]
    tp, fp  = metrics["tp"], metrics["fp"]
    fn      = metrics["fn"]
    ax.text(
        0.97, 0.04,
        f"Ensemble — TP: {tp:,}  FP: {fp:,}  FN: {fn:,}",
        transform=ax.transAxes, ha="right", fontsize=8,
        color=PALETTE["dark"], style="italic",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Ensemble score distributions (attackers vs normals)
# ---------------------------------------------------------------------------
def _plot_score_distribution(metrics: dict, output_path: str) -> None:
    logging.info("Generating Plot 3: score distribution separation …")

    att_mean  = metrics["attacker_score_mean"]
    att_std   = metrics["attacker_score_std"]
    norm_mean = metrics["normal_score_mean"]
    norm_std  = metrics["normal_score_std"]
    threshold = metrics["ensemble_threshold"]

    x = np.linspace(0, 1, 500)
    att_dist  = stats.norm.pdf(x, att_mean,  att_std  + 1e-6)
    norm_dist = stats.norm.pdf(x, norm_mean, norm_std + 1e-6)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(x, norm_dist, alpha=0.45, color=PALETTE["blue"],  label="Normal users")
    ax.fill_between(x, att_dist,  alpha=0.55, color=PALETTE["red"],   label="Attackers")
    ax.plot(x, norm_dist, color=PALETTE["blue"], lw=1.5)
    ax.plot(x, att_dist,  color=PALETTE["red"],  lw=1.5)
    ax.axvline(
        threshold, color=PALETTE["green"], ls="--", lw=2,
        label=f"Ensemble threshold ({threshold:.2f})",
    )

    ax.set_title(
        "Ensemble score distribution — attacker vs normal separation",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Ensemble anomaly score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_dissertation_plots(
    data_path: str    = "data/processed/cert_expanded_matrix.csv",
    metrics_path: str = "reports/metrics.json",
    output_dir: str   = "reports/figures",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load live metrics
    # ------------------------------------------------------------------
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}\n"
            "Run evt_master_engine.py first to generate it."
        )
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    logging.info(f"Loaded live metrics from {metrics_path}")

    # ------------------------------------------------------------------
    # 2. Re-fit IF on the expanded matrix to get raw scores for EVT plot
    # ------------------------------------------------------------------
    print("📈 Loading data and building features …")
    raw_df = pd.read_csv(data_path)
    df     = build_features(raw_df)
    X      = df[FEATURE_COLS]

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print("🌲 Fitting Isolation Forest for EVT tail mapping …")
    if_model = IsolationForest(
        contamination=metrics["contamination_rate"],
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    if_model.fit(X_scaled)
    if_scores = if_model.decision_function(X_scaled)

    # ------------------------------------------------------------------
    # 3. Render plots
    # ------------------------------------------------------------------
    print("🎨 Generating Plot 1: EVT GPD tail curve …")
    _plot_evt_tail(
        if_scores, metrics,
        os.path.join(output_dir, "evt_gpd_tail_fit.png"),
    )

    print("🎨 Generating Plot 2: model performance comparison …")
    _plot_model_comparison(
        metrics,
        os.path.join(output_dir, "model_performance_comparison.png"),
    )

    print("🎨 Generating Plot 3: ensemble score distribution …")
    _plot_score_distribution(
        metrics,
        os.path.join(output_dir, "score_distribution.png"),
    )

    print("✨ All dissertation figures rendered successfully.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_dissertation_plots()
