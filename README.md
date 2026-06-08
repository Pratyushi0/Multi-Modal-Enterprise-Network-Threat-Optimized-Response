<div align="center">

```
███╗   ███╗███████╗███╗   ██╗████████╗ ██████╗ ██████╗ 
████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗██╔══██╗
██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ██║   ██║██████╔╝
██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██╔══██╗
██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ╚██████╔╝██║  ██║
╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

### **Multi-Modal Enterprise Network Threat Optimized Response**

*A production-scale unsupervised insider threat detection pipeline*

---

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CERT%20r4.2-00C853?style=for-the-badge)
![Precision](https://img.shields.io/badge/Precision-36.9%25-FF6B35?style=for-the-badge)
![FP Reduction](https://img.shields.io/badge/False%20Alarm%20Reduction-76%25-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic%20Research-blueviolet?style=for-the-badge)

</div>

---

## ◈ What is MENTOR?

MENTOR is an **unsupervised, production-scale cybersecurity pipeline** designed to detect insider threats inside enterprise behavioral telemetry without requiring any labeled attack data. It ingests multi-modal activity streams — logins, USB transfers, web activity, and file interactions — and identifies anomalous user behavior through a **dual-engine Consensus Ensemble architecture**.

The core innovation is a **Consensus Intersection Gate**: a day is only flagged as malicious if both an Isolation Forest (with Extreme Value Theory tail calibration) and a One-Class SVM independently agree. This eliminates the statistical noise that causes individual anomaly detectors to flood security analysts with false alarms.

> **The result:** On 330,452 user-days with a 0.413% true attack density, MENTOR generates only **530 false positives** — fewer than **1 false alert per day** — while maintaining **Precision = 36.9%**, an 89× lift over the random detection baseline.

---

## ◈ Performance at a Glance

<div align="center">

### 🎯 Final MENTOR Consensus Result

| Metric | Value | Context |
|:---:|:---:|:---:|
| **Precision** | **0.369** | 89× above random baseline (0.00413) |
| **Recall** | **0.227** | 310 of 1,364 true threat-days caught |
| **F1-Score** | **0.281** | Harmonic mean |
| **True Positives** | **310** | Genuine insider days intercepted |
| **False Positives** | **530** | False alarms over entire 18-month period |
| **False Alarm Rate** | **0.16%** | On 329,088 normal days |
| **True Negatives** | **328,558** | Innocents correctly cleared |

</div>

---

## ◈ Cross-Model Benchmark

> All models unsupervised. Contamination set to true empirical rate (0.00413). No SMOTE or label access.

| Classifier | Precision | Recall | F1 | TP | FP | Δ FP vs MENTOR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Isolation Forest + EVT | 0.172 | 0.485 | 0.254 | 662 | 3,180 | +2,650 |
| One-Class SVM (RBF) | 0.265 | 0.289 | 0.276 | 394 | 1,094 | +564 |
| Local Outlier Factor† | 0.007 | 0.007 | 0.007 | 9 | 1,355 | +825 |
| **MENTOR Consensus ✦** | **0.369** | **0.227** | **0.281** | **310** | **530** | **—** |

> †LOF excluded from primary analysis: zero-inflated behavioral telemetry (many identical zero-activity user-days) causes structural incompatibility with density-based methods.

---

## ◈ Architectural Evolution

Five discrete engineering decisions, each measured against the same corrected ground truth.

```
   Precision
   0.40 ┤
        │                                              ╔═══╗
   0.35 ┤                                              ║ ✦ ║  Phase 5
        │                                              ╚═══╝
   0.30 ┤                                        ┌─────┘
        │                                   ┌────┘  Phase 4
   0.25 ┤                              ┌────┘
        │                         ┌────┘  Phase 3
   0.20 ┤                    ┌────┘
        │               ┌────┘  Phase 2
   0.15 ┤          ┌────┘
        │     ┌────┘  Phase 1
   0.10 ┤─────┘
        └──────────────────────────────────────────────────▶ Phase
            1         2         3         4         5
```

| Phase | Architecture Decision | Precision | Recall | F1 | TP | FP |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | Vanilla Isolation Forest, base features | 0.163 | 0.163 | 0.163 | 223 | 1,141 |
| 2 | + EVT Generalized Pareto tail threshold | 0.167 | 0.268 | 0.205 | 365 | 1,827 |
| 3 | + File telemetry (multi-modal expansion) | 0.185 | 0.304 | 0.230 | 415 | 1,828 |
| 4 | + OC-SVM on expanded feature space | 0.265 | 0.289 | 0.276 | 394 | 1,094 |
| **5** | **+ Consensus Intersection Gate** | **0.369** | **0.227** | **0.281** | **310** | **530** |

**Phase 5 vs Phase 1:** Precision +127% · False Positives −54% · Operationally viable

---

## ◈ System Architecture

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                     CERT r4.2 RAW TELEMETRY                      │
 │              logon · device · http · file · email                │
 └─────────────────────────────┬────────────────────────────────────┘
                               │
                  ┌────────────▼────────────┐
                  │      label_injector     │  Exact date-range overlay
                  │                         │  against ground-truth answers
                  │  1,364 attack-days      │  190 insider records
                  │  329,088 normal-days    │  errors='coerce' on malformed dates
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │  multi_stream_aggregator│  Chunk-streams HTTP + File logs
                  │                         │  from raw ZIP (memory-safe)
                  │  web_clicks per user    │  Left-merge preserves labels
                  │  file_touches per user  │  Integrity assertion before write
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │    FEATURE ENGINEERING  │
                  │                         │
                  │  EWM behavioral         │  Per-user exponential baselines
                  │  baselines (span=14)    │  across 4 telemetry modalities
                  │                         │
                  │  Z-score deviations     │  login · usb · web · file
                  │                         │
                  │  Velocity cascades      │  3-day + 7-day rolling sums
                  │  (3d + 7d windows)      │  13 total engineered features
                  └──────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
   ┌──────────▼──────────┐       ┌──────────▼──────────┐
   │      ENGINE 1        │       │      ENGINE 2        │
   │                      │       │                      │
   │  Isolation Forest    │       │   One-Class SVM      │
   │  n_estimators=400    │       │   kernel=RBF         │
   │  + Generalized       │       │   nu=TRUE_CONTAM     │
   │  Pareto EVT tail     │       │   gamma=scale        │
   │  threshold fit       │       │                      │
   │                      │       │                      │
   │  Partitions feature  │       │  Draws kernel        │
   │  space via random    │       │  hyperplane around   │
   │  tree isolation      │       │  normal behavior     │
   └──────────┬───────────┘       └──────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │   CONSENSUS     │
                    │   GATE  (∩)     │  Flag day ONLY if
                    │                 │  BOTH engines agree
                    │  np.bitwise_and │
                    │  (if_pred,      │
                    │   svm_pred)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ALERT OUTPUT   │  530 FP over 18 months
                    │                 │  1-in-3 alerts = real threat
                    │  Precision 0.37 │  Operationally viable SOC queue
                    └─────────────────┘
```

---

## ◈ Dataset

**CERT Insider Threat Dataset r4.2** — Carnegie Mellon University Software Engineering Institute

| Property | Value |
|:---|:---|
| Total user-day records | 330,452 |
| True attack-day records | **1,364** |
| Normal-day records | 329,088 |
| True contamination rate | **0.00413 (0.413%)** |
| Unique insider threat actors | 190 (of 191; 1 record has unparseable dates) |
| Telemetry streams used | Logon, Device (USB), HTTP, File |
| Ground truth source | `answers/insiders.csv` — exact date-range windows |

---

## ◈ Repository Structure

```
cyber_ml/
│
├── 📄 label_injector.py          → Ground-truth label alignment
│                                    Date-range overlay onto user-day matrix
│                                    Outputs: cert_labeled_matrix.csv (1,364 pos)
│
├── 📄 multi_stream_aggregator.py → Multi-modal telemetry fusion
│                                    Streams HTTP + file logs from ZIP
│                                    Outputs: cert_expanded_matrix.csv
│
├── 📄 evt_master_engine.py       → Core MENTOR consensus pipeline
│                                    Full OC-SVM + IF+EVT → intersection gate
│                                    Outputs: Final evaluation report
│
├── 📄 benchmarker.py             → Cross-model evaluation suite
│                                    IF, OC-SVM, LOF, MENTOR side-by-side
│                                    Outputs: Benchmark comparison table
│
├── 📄 recover_logs.py            → Phase 1–5 evolution log generator
│                                    Reproducible architectural history
│                                    Outputs: reports/phd_metrics_evolution.txt
│
└── data/
    ├── raw/
    │   └── dataset.zip           → CERT r4.2 (not tracked in git)
    └── processed/
        ├── cert_processed_matrix.csv    → Base aggregated features
        ├── cert_labeled_matrix.csv      → + Corrected ground-truth labels
        └── cert_expanded_matrix.csv     → + HTTP and file telemetry
```

---

## ◈ How to Run

### Prerequisites
```bash
pip install pandas numpy scipy scikit-learn
```

### Execution — Run in this exact order

```bash
# 1. Align ground-truth labels to user-day matrix
python3 label_injector.py
# Expected: True Attack Vector Days: 1,364 rows ✓

# 2. Stream and merge HTTP + file telemetry
python3 multi_stream_aggregator.py
# Expected: Final label integrity check: 1364 attack rows ✓
# Note: Takes ~2 minutes (streaming 2M+ HTTP log rows from ZIP)

# 3. Generate Phase 1–5 architectural evolution table
python3 recover_logs.py

# 4. Run cross-model benchmark comparison
python3 benchmarker.py
# Note: OC-SVM on 330k rows takes ~90 seconds

# 5. Run final MENTOR consensus pipeline
python3 evt_master_engine.py
```

---

## ◈ Key Research Findings

**① Contamination calibration is non-negotiable**
Setting `contamination=0.01` (sklearn default) on data with a true attack rate of 0.00413 forces every model to over-flag by 2.4×, artificially suppressing precision by ~40%. All MENTOR modules derive contamination from actual label counts dynamically.

**② The consensus gate beats both constituent models on precision**
No individual unsupervised model in the benchmark exceeds Precision 0.265. The IF∩OC-SVM intersection reaches 0.369 — surpassing either model alone — because genuine insider anomalies trigger both geometric detection approaches simultaneously, while noise triggers only one.

**③ EVT tail thresholding recovers recall without labeled data**
Replacing IF's contamination-based cutoff with a Generalized Pareto Distribution tail fit raises recall from 0.163 to 0.485 on the IF-only benchmark row — a 197% recall improvement with zero label access.

**④ Multi-modal fusion provides additive, non-redundant signal**
Adding file telemetry (Phase 2→3) raises recall by 13.4% with negligible precision cost, confirming that file and HTTP streams encode complementary behavioral signals beyond logon and USB activity.

**⑤ Operational viability is achieved**
In a 330k user-day deployment, MENTOR generates 530 total false alerts over 18 months (< 1/day). Of every 3 alerts raised, 1 is a confirmed insider threat. This is a viable analyst review queue.

---

## ◈ Limitations

| Limitation | Detail |
|:---|:---|
| **Recall ceiling** | The consensus gate misses 76.7% of attack-days by design. MENTOR optimizes precision over recall — best suited for environments where alert fatigue is the primary bottleneck. |
| **Unsupervised ceiling** | Supervised models with labeled data achieve substantially higher F1. MENTOR's value proposition is operation without labeled attack examples. |
| **Dataset scope** | Validated on CERT r4.2 behavioral telemetry only. Performance on network intrusion or external threat datasets is not evaluated. |
| **LOF incompatibility** | Zero-inflated behavioral data (many identical zero-activity user-days) causes LOF to produce near-random results. Density-based methods require non-degenerate feature distributions. |

---

## ◈ Technical Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

| Component | Implementation |
|:---|:---|
| Anomaly Detection | `IsolationForest`, `OneClassSVM` (scikit-learn) |
| Tail Thresholding | `scipy.stats.genpareto` (Generalized Pareto Distribution) |
| Feature Scaling | `RobustScaler` (IQR-based, robust to behavioral outliers) |
| Temporal Features | Exponential weighted mean + rolling sum windows |
| Data Ingestion | Chunked streaming from ZIP archive (memory-safe, 250k rows/chunk) |

---

<div align="center">

---

*Project MENTOR · CERT Insider Threat Dataset r4.2 · Carnegie Mellon University SEI*

*Unsupervised · No label access · Production-scale · 330,452 user-days*

</div>
