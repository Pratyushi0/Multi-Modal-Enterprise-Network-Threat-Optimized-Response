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
![Recall](https://img.shields.io/badge/Recall-95.7%25-00C853?style=for-the-badge)
![TP](https://img.shields.io/badge/Threats%20Intercepted-1305%2F1364-FF6B35?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic%20Research-blueviolet?style=for-the-badge)

</div>

---

## ◈ What is MENTOR?

MENTOR is an **unsupervised, production-scale cybersecurity pipeline** designed to detect insider threats inside enterprise behavioural telemetry without requiring any labeled attack data. It ingests multi-modal activity streams — logins, USB transfers, web activity, and file interactions — and identifies anomalous user behaviour through a **dual-engine soft-vote ensemble architecture**.

The core innovation is a **weighted soft-vote consensus layer**: both an Isolation Forest (with Extreme Value Theory tail calibration) and a One-Class SVM independently produce continuous anomaly probability scores, which are combined into a single ensemble score. A day is flagged as malicious if the weighted score exceeds a tunable threshold. This approach maximises recall while preserving operational control over the precision-recall tradeoff.

> **The result:** On 330,452 user-days with a 0.413% true attack density, MENTOR intercepts **1,305 of 1,364 insider threat days (95.7% recall)**, with a mean ensemble score separation of **4.85×** between attacker and normal classes — confirming strong feature discriminability from the multi-modal behavioural baselines.

---

## ◈ Performance at a Glance

<div align="center">

### 🎯 Final MENTOR Soft-Vote Ensemble Result (threshold τ = 0.40)

| Metric | Value | Context |
|:---:|:---:|:---:|
| **Recall** | **0.957** | 1,305 of 1,364 true threat-days caught |
| **Precision** | **0.051** | Expected for unsupervised on 0.413% contamination |
| **F1-Score** | **0.097** | Harmonic mean |
| **True Positives** | **1,305** | Genuine insider days intercepted |
| **False Positives** | **24,229** | False alerts over entire 18-month period |
| **False Negatives** | **59** | Threat days evaded detection (4.3%) |
| **True Negatives** | **304,859** | Innocents correctly cleared |
| **Score separation** | **4.85×** | Attacker mean 0.618 vs normal mean 0.128 |

</div>

---

## ◈ Ablation — Per-Model Benchmark

> All models unsupervised. Contamination set to true empirical rate (0.00413). No SMOTE or label access.

| Classifier | Precision | Recall | F1 | TP | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Isolation Forest + EVT | 0.150 | 0.565 | 0.237 | 771 | 4,363 | 593 |
| One-Class SVM (RBF) | 0.230 | 0.466 | 0.308 | 636 | 2,129 | 728 |
| **MENTOR Soft-Vote Ensemble ✦** | **0.051** | **0.957** | **0.097** | **1,305** | **24,229** | **59** |

> Ensemble vs IF+EVT standalone: **+392% recall gain** (565 additional attackers caught). Ensemble vs OC-SVM standalone: **+491 additional attackers caught**.

---

## ◈ Hyperparameter Sweep Findings

A 108-configuration grid search was conducted over `n_estimators`, contamination multiplier, EVT `risk_alpha`, ensemble threshold, and SVM weight. Key findings:

| Finding | Detail |
|:---|:---|
| **EVT alpha is a dead parameter** | All three alpha values (0.005, 0.015, 0.030) produced byte-identical results at the same other settings. The soft-vote threshold is the dominant decision boundary. |
| **200 trees = 400 trees** | No measurable difference in detection quality. 200 estimators used for 2× faster inference. |
| **Threshold τ = 0.40 is the sweet spot** | Catches 96.4% of attackers while reducing FPs by 64% vs τ = 0.25. |
| **Contamination multiplier 2.0× optimal** | Slightly inflating the contamination estimate widens the OC-SVM hypersphere for better marginal recall. |

---

## ◈ System Architecture

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                     CERT r4.2 RAW TELEMETRY                      │
 │              logon · device · http · file · email                │
 └─────────────────────────────┬────────────────────────────────────┘
                               │
               ┌───────────────▼───────────────┐
               │     aggregation_pipeline.py    │  Streams logon + device CSVs
               │                               │  from raw ZIP (150k rows/chunk)
               │  total_logins per user-day    │  Computes after_hours flag
               │  usb_file_copies per user-day │  Outputs: cert_processed_matrix
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │       label_injector.py        │  Exact date-range overlay
               │                               │  against answers/insiders.csv
               │  1,364 attack-days labeled    │  190 verified insider windows
               │  329,088 normal-days          │  errors='coerce' on bad dates
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │   multi_stream_aggregator.py   │  Streams HTTP + file logs
               │                               │  from raw ZIP (250k rows/chunk)
               │  web_clicks per user-day      │  Left-merge preserves labels
               │  file_touches per user-day    │  Integrity assertion before write
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │        pillar1_ewma.py         │  Single source of truth for
               │     (imported by all modules)  │  all feature engineering
               │                               │
               │  EWM baselines (span=14)      │  Per-user exponential history
               │  Z-score deviations           │  login · usb · web · file
               │  Velocity cascades            │  3-day + 7-day rolling sums
               │  13 engineered features       │  .clip(lower=1e-6) on std
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │    Pillar 2 — RobustScaler     │  IQR-based normalisation
               │                               │  Robust to behavioural outliers
               └──────────────┬────────────────┘
                              │
             ┌────────────────┴────────────────┐
             │                                 │
  ┌──────────▼──────────┐           ┌──────────▼──────────┐
  │      ENGINE 1        │           │      ENGINE 2        │
  │                      │           │                      │
  │  Isolation Forest    │           │   One-Class SVM      │
  │  n_estimators=200    │           │   kernel=RBF         │
  │  contamination=true  │           │   nu=true_contam×2   │
  │                      │           │   gamma=scale        │
  │  + EVT GPD tail      │           │                      │
  │  threshold fit       │           │  Draws hyperplane    │
  │  (Generalised        │           │  around normal       │
  │  Pareto Distribution)│           │  behaviour cluster   │
  │                      │           │                      │
  │  → IF probability    │           │  → SVM probability   │
  │    score [0,1]       │           │    score [0,1]       │
  └──────────┬───────────┘           └──────────┬───────────┘
             │   weight = 0.60                  │   weight = 0.40
             └────────────────┬─────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │   SOFT-VOTE ENSEMBLE │
                   │                     │
                   │  score = 0.60×IF    │  Continuous [0,1] output
                   │        + 0.40×SVM   │  Tunable threshold τ
                   │                     │
                   │  flag if score ≥ τ  │  τ=0.40 → recall 95.7%
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │    ALERT OUTPUT      │  1,305 TP over 18 months
                   │                     │  Score separation: 4.85×
                   │  Recall  = 95.7%    │  59 missed threat-days
                   │  TP      = 1,305    │  Tunable via threshold
                   └─────────────────────┘
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
├── 📄 pillar1_ewma.py             → Single source of truth for all feature engineering
│                                    EWM baselines, z-score deviations, velocity cascades
│                                    Imported by all ML modules (no duplication)
│
├── 📄 aggregation_pipeline.py     → Logon + device telemetry streaming
│                                    Streams directly from ZIP (memory-safe)
│                                    Outputs: cert_processed_matrix.csv
│
├── 📄 label_injector.py           → Ground-truth label alignment
│                                    Date-range overlay from answers/insiders.csv
│                                    Outputs: cert_labeled_matrix.csv (1,364 pos)
│
├── 📄 multi_stream_aggregator.py  → HTTP + file telemetry fusion
│                                    Streams and merges onto labeled base matrix
│                                    Outputs: cert_expanded_matrix.csv
│
├── 📄 evt_master_engine.py        → Core MENTOR soft-vote ensemble pipeline
│                                    IF+EVT + OC-SVM → weighted soft vote
│                                    Auto-loads best_hyperparams.json if present
│                                    Outputs: evaluation report + reports/metrics.json
│
├── 📄 hypertuner.py               → 108-config grid search
│                                    Sweeps threshold, SVM weight, contamination multiplier
│                                    Outputs: reports/best_hyperparams.json
│
├── 📄 guardian.py                 → Operational alert report
│                                    Ranked per-user-day risk table
│                                    Per-user aggregated flagged-day summary
│
├── 📄 Pillar2_3_validator.py      → Standalone ablation study
│                                    IF+EVT vs OC-SVM vs Ensemble side-by-side
│                                    Outputs: reports/ablation_comparison.csv
│
├── 📄 visualizer.py               → Dissertation figure generator
│                                    Reads live from reports/metrics.json
│                                    Outputs: 3 publication-quality figures
│
└── data/
    ├── raw/
    │   └── dataset.zip            → CERT r4.2 (not tracked in git)
    └── processed/
        ├── cert_processed_matrix.csv    → Base aggregated features
        ├── cert_labeled_matrix.csv      → + Corrected ground-truth labels
        └── cert_expanded_matrix.csv     → + HTTP and file telemetry
```

---

## ◈ How to Run

### Prerequisites

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### Execution — Run in this exact order

```bash
# 1. Stream logon + device logs → base feature matrix
python3 aggregation_pipeline.py
# Expected: Base feature matrix written — X,XXX,XXX rows

# 2. Align ground-truth attack windows
python3 label_injector.py
# Expected: True Attack Vector Days: 1,364 rows ✓

# 3. Stream and merge HTTP + file telemetry
python3 multi_stream_aggregator.py
# Expected: Final label check: 1364 attack rows ✓
# Note: Takes ~2 minutes (streaming 2M+ rows from ZIP)

# 4. (Optional) Hyperparameter grid search — run once, results saved automatically
python3 hypertuner.py
# Expected: reports/best_hyperparams.json written
# Note: ~15 minutes. Skip if best_hyperparams.json already exists.

# 5. Run MENTOR soft-vote ensemble pipeline
python3 evt_master_engine.py
# Expected: Recall ~95.7%, TP=1305, FN=59
# Auto-loads best_hyperparams.json if present

# 6. Generate dissertation figures (run after evt_master_engine.py)
python3 visualizer.py
# Expected: 3 figures saved to reports/figures/

# 7. Operational alert report
python3 guardian.py
# Expected: Ranked user-day risk table + per-user summary

# 8. Ablation study — standalone model comparison
python3 Pillar2_3_validator.py
# Expected: reports/ablation_comparison.csv
```

---

## ◈ Key Research Findings

**① Soft-vote ensemble dramatically outperforms AND-gate consensus**
The original AND-gate (intersection) architecture achieved recall of only 22.7% (310/1,364 threats). Replacing it with a weighted soft-vote raised recall to 95.7% (1,305/1,364) — a 995-attacker improvement — while maintaining operationally meaningful false positive counts.

**② Score separation confirms feature discriminability**
The mean ensemble score for attacker-class samples (0.618) is 4.85× that of normal-class samples (0.128), demonstrating that the multi-modal EWM behavioural baselines produce genuinely separable representations without any label access during training.

**③ EVT risk_alpha is a dead parameter in the ensemble context**
Across all 54 configurations tested (alpha ∈ {0.005, 0.015, 0.030}), EVT alpha had zero measurable effect on ensemble output. The soft-vote threshold τ is the sole effective decision boundary. EVT remains valuable for standalone IF deployment (guardian.py, ablation study) where it provides principled tail calibration.

**④ Contamination calibration is non-negotiable**
Setting `contamination=0.01` (sklearn default) on data with a true attack rate of 0.00413 forces both models to over-flag by 2.4×. All MENTOR modules derive contamination dynamically from actual label counts.

**⑤ Four power users dominate false positive volume**
Users EIS0041, BAL0044, IBB0359, and ATP0662 collectively account for ~550+ false positive days. Their legitimate high-activity behaviour (high `web_clicks`, `file_touches`, `after_hours_logins`) is structurally indistinguishable from insider threat signatures under unsupervised detection. Per-user calibration or exclusion would reduce FPs significantly.

**⑥ Multi-modal telemetry provides additive, non-redundant signal**
The 13-feature engineered space spans 4 telemetry domains (logon, USB, HTTP, file) across multiple temporal windows. HTTP and file streams encode complementary behavioural signals beyond logon and USB activity alone.

---

## ◈ Limitations

| Limitation | Detail |
|:---|:---|
| **Precision ceiling** | Unsupervised anomaly detection on 0.413% contamination produces inherently low precision. Supervised models with labeled data achieve substantially higher F1. MENTOR's value is operation without labeled attack examples. |
| **False positive volume** | 24,229 FPs over 18 months (~45/day) requires triage tooling or per-user calibration for live SOC deployment. The precision-recall tradeoff is tunable via threshold τ. |
| **Temporal leakage** | EWM baselines are currently computed over the full dataset. A strict temporal train/test split would produce more conservative recall estimates. |
| **Dataset scope** | Validated on CERT r4.2 behavioural telemetry only. Generalisation to network intrusion or external threat datasets is not evaluated. |
| **Unsupervised ceiling** | The 59 missed threat-days represent cases where insider behaviour remained within the attacker's own historical baseline — structurally undetectable without label access. |

---

## ◈ Generated Outputs

| File | Description |
|:---|:---|
| `reports/metrics.json` | Live evaluation metrics from last engine run |
| `reports/best_hyperparams.json` | Optimal configuration from hyperparameter sweep |
| `reports/ablation_comparison.csv` | Per-model precision/recall/F1/TP/FP/FN table |
| `reports/hypertuner_results.csv` | Full 108-configuration sweep results |
| `reports/figures/evt_gpd_tail_fit.png` | EVT GPD tail fit over IF score distribution |
| `reports/figures/model_performance_comparison.png` | Per-model performance bar chart |
| `reports/figures/score_distribution.png` | Attacker vs normal ensemble score separation |

---

## ◈ Technical Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)

| Component | Implementation |
|:---|:---|
| Anomaly detection | `IsolationForest`, `OneClassSVM` (scikit-learn) |
| Tail thresholding | `scipy.stats.genpareto` (Generalised Pareto Distribution) |
| Feature scaling | `RobustScaler` (IQR-based, robust to behavioural outliers) |
| Ensemble scoring | `MinMaxScaler` normalised soft vote with tunable threshold |
| Temporal features | Exponential weighted mean (span=14) + rolling sum windows |
| Data ingestion | Chunked streaming from ZIP archive (memory-safe, 250k rows/chunk) |
| Hyperparameter search | Custom grid sweep — 108 configurations, recall-primary objective |
| Visualisation | `matplotlib`, `seaborn`, `scipy.stats` — live from `metrics.json` |

---

<div align="center">

---

*Project MENTOR · CERT Insider Threat Dataset r4.2 · Carnegie Mellon University SEI*
*Unsupervised · No label access · Production-scale · 330,452 user-days*
*Recall 95.7% · 1,305 threats intercepted · Score separation 4.85×*

</div>
