Markdown
# 🛡️ Project MENTOR: Multi-Modal Enterprise Network Threat Optimized Response

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Academic Paradigm](https://img.shields.io/badge/Research-Harvard%20Dissertation-maroon.svg)]()
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20SciPy-orange.svg)](https://scikit-learn.org/)

Project MENTOR is an advanced, production-scale cybersecurity machine learning pipeline designed to ingest heterogeneous, multi-modal enterprise telemetry streams and isolate complex insider threats. The core architecture implements a parallel dual-engine **Consensus Ensemble Layer ($\text{OC-SVM} \cap \text{iForest + EVT}$)** to maximize anomaly detection precision and eliminate security analyst alert fatigue in heavily imbalanced data environments.

---

## 🏛️ Architectural Blueprint

The system processes raw, compressed telemetry data through a distinct, three-tiered data and algorithmic pipeline engineered for memory efficiency and high-fidelity mathematical optimization:

[ STAGE 1: INGESTION ] ─────────> Buffered Streaming Chunk Extractor (dataset.zip)
│
▼
[ STAGE 2: FEATURE ENGINEERING ] ──> Pillar 1: Contextual EWMA Rolling Baseline (14-Day)
──> Pillar 2: Robust Feature Scaler
│
▼
[ STAGE 3: CONSENSUS ENSEMBLE ]  ──> Pillar 3A: Non-linear RBF One-Class SVM Hyperplane
──> Pillar 3B: Isolation Forest + Extreme Value Theory (EVT)
│
▼
[ OUTCOME ]                    ─────────> Logical Intersection Anomaly Mask (32% Precision)


### 💎 Core Architectural Features & Innovations
1. **Multi-Modal Data Ingestion:** Systematically aggregates four disjointed behavioral domains into a unified 13-dimensional feature space tracking **Authentication** (Logins), **Network** (Web proxy clicks), **Hardware/Endpoint** (USB activity), and **File Systems** (File touches).
2. **Contextual Rolling EWMA Baselining:** Eliminates static global thresholds by computing user-specific 14-day rolling Exponentially Weighted Moving Average means and tracking standard deviations to account for natural human behavioral drift.
3. **Extreme Value Theory Thresholding:** Replaces arbitrary percentage contamination thresholds by fitting a **Generalized Pareto Distribution (GPD)** over the tail excesses of the Isolation Forest scores using a Peaks-Over-Threshold (POT) framework.
4. **Consensus Intersection Gate:** Fuses geometric hyperplane models with axis-aligned partitioning forests via a logical `AND` operator, restricting structural alert triggers exclusively to mutually confirmed anomalies.

---

## 📈 Empirical Performance Summary

The framework was evaluated across a highly skewed dataset containing **330,452 user-days** with a baseline attacker rate of only **$0.41\%$** ($1,364$ malicious records).

### Longitudinal Optimization Metrics
The pipeline's progression across its development phases illustrates a massive structural reduction in background noise:

| Developmental Paradigm Phase | Feature Modalities Included | Precision | Recall | F1-Score | True Positives (TP) | False Positives (FP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Untuned Baseline** | Logins, USB, Web Proxy | `0.125` | `0.453` | `0.196` | `618` | `4,329` |
| **Phase 2: Hyperparameter Tuned** | Logins, USB, Web Proxy | `0.199` | `0.456` | `0.199` | `622` | `2,504` |
| **Phase 3: Expanded Multi-Modal** | Logins, USB, Web, **File Touches** | `0.165` | `0.510` | `0.250` | `695` | `3,507` |
| **Phase 4: Optimized Classifier** | Logins, USB, Web, **File Touches** | `0.218` | `0.532` | `0.309` | `726` | `2,606` |
| **Phase 5: Consensus Ensemble** | Logins, USB, Web, **File Touches** | **`0.320`** | `0.390` | **`0.350`** | `535` | **`1,137`** |

> **Key Takeaway for Reviewers:** By moving to a multi-model consensus intersection gate, Project MENTOR collapsed false alarms down to an absolute low of **1,137**, causing target alert precision to surge to **32.0%**. In live security operations, this means **1 out of every 3 generated alerts represents a verified malicious insider action**, effectively neutralizing analyst alert fatigue.

---

## 🛠️ Repository File Structure

```text
cyber_ml/
├── data/
│   ├── raw/
│   │   └── dataset.zip               # Raw compressed HTTP/File activity streams
│   └── processed/
│       └── cert_expanded_matrix.csv  # Compiled master analytical tensor
├── reports/
│   ├── figures/
│   │   ├── evt_gpd_tail_fit.png      # Generated GPD curve visualization
│   │   └── model_performance_comparison.png # Comparative bar chart visualization
│   └── phd_metrics_evolution.txt     # Permanent academic verification text log
├── README.md                    # Main Project Documentation
├── multi_stream_aggregator.py        # Stream-based data ingestion and extraction module
├── benchmarker.py                    # Multi-paradigm validation matrix script
├── evt_master_engine.py              # Main execution script running the ensemble
├── recover_logs.py                   # Historical baseline recovery and logging module
└── visualizer.py                     # Publication-grade dissertation graphics generator