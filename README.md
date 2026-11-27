# Vehicle Loan Default Prediction — Empirical Machine Learning Study

**Course:** Programming for Artificial Intelligence and Data Science (CO3135 — HCMUT CSE)  
**Dataset:** L&T Financial Services (LTFS) Vehicle Loan Default Prediction ([Kaggle: `mamtadhaker/lt-vehicle-loan-default-prediction`](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction))  
**Task:** Predict **First Payment Default (`loan_default` $\in \{0, 1\}$)** at loan origination (`DisbursalDate`) across **$N = 233{,}154$** labeled vehicle loan contracts (default rate `21.71%`, class imbalance ratio `3.61 : 1`).  
**Project Report (PDF):** [`reports/final_report_en.pdf`](reports/final_report_en.pdf) | **LaTeX Source:** [`reports/latex/main.tex`](reports/latex/main.tex)

---

## 1. Key Methodology & Findings

1. **Data Cleaning & Feature Engineering ([`src/features/engineering.py`](src/features/engineering.py), [`src/features/preprocessing.py`](src/features/preprocessing.py)):**
   - Dropped sequential ID (`UniqueID`, Spearman $\rho = 0.9998$ with `DisbursalDate`) and constant `MobileNo_Avl_Flag`.
   - Fixed 2-digit birth-year century rollover (`dd-mm-yy` `'00`–`'68` misparsed as `20xx`) for `19,866` rows (`8.52%`), restoring valid borrower ages in `[18.0, 69.1]` years.
   - Separated unscored thin-file bureau status codes (`PERFORM_CNS.SCORE < 300`, `55.66%` of rows) from numeric credit scores (`300–890`), and added leverage, delinquency, and ledger-inconsistency features (`F0` $\to$ `F1`).
   - Encoded high-cardinality channel/geography IDs (`branch_id`, `supplier_id`, `Current_pincode_ID`) via fold-isolated Out-of-Fold Empirical-Bayes target encoding + frequency encoding (`F2_safe`), gaining `+0.0115` to `+0.0180` OOT ROC-AUC over `F1`.
2. **Temporal vs. Random Validation ([`src/data/splits.py`](src/data/splits.py)):**
   - **Primary Out-of-Time (OOT) Split:** Development Set `2018-08-01` to `2018-10-22` ($N = 182{,}501$, default rate `20.63%`, 5-fold stratified CV) and Locked OOT Test Set `2018-10-23` to `2018-10-31` ($N = 50{,}653$, default rate `25.59%`).
   - **Comparison with Random 80/20 Split:** Standard random splitting inflates ROC-AUC by `+0.0091` to `+0.0106` and hides a `30x` increase in calibration error (`ECE: 0.0016 -> 0.0491`) caused by late-October default-rate drift.
3. **Class Imbalance, Cost Thresholds & Explainability:**
   - Fold-isolated `SMOTE` degrades `LightGBM` OOT ROC-AUC from `0.6621` down to `0.6252`, and balanced `class_weight` severely distorts calibration (`ECE = 0.2111`).
   - Training on natural class priors with a Bayesian `5:1` cost threshold ($\tau = 0.167$) cuts principal-weighted portfolio loss nearly in half (`7.13%` vs. `13.12%` at $\tau = 0.50$).
   - Exact **TreeSHAP** explanations identify `ltv`, dealer/branch channel risk, `bureau_score_clean`, `disbursed_amount`, and recent delinquencies as top drivers, with stable feature rankings from Development to OOT Test (Spearman $\rho = 0.973$).

---

## 2. Main Results on Locked OOT Test Set ($N = 50{,}653$, `F2_safe`)

Extracted from [`reports/tables/m6_final_oot_comparison.csv`](reports/tables/m6_final_oot_comparison.csv):

| Rank | Model | Dev 5-Fold CV AUC | OOT ROC-AUC | OOT PR-AUC (Lift vs. `0.2559`) | KS | Brier Score | OOT ECE | F1@Opt ($\tau_{\text{Dev}}$) | Fit Time (s) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`LightGBM` (Tuned — Champion)** | 0.6748 | **0.6621** | **0.3910** (`1.53x`) | 0.2368 | 0.1810 | 0.0491 | **0.4540** (`0.205`) | 10.9 |
| 2 | **`XGBoost` (Tuned)** | **0.6749** | 0.6620 | 0.3907 (`1.53x`) | **0.2369** | **0.1810** | **0.0489** | 0.4526 (`0.209`) | 13.3 |
| 3 | `RandomForest` (Baseline) | 0.6688 | 0.6554 | 0.3820 (`1.49x`) | 0.2275 | 0.1826 | 0.0499 | 0.4505 (`0.205`) | 8.9 |
| 4 | `LogisticRegression` (`C=0.1`) | 0.6653 | 0.6525 | 0.3762 (`1.47x`) | 0.2232 | 0.1826 | 0.0490 | 0.4516 (`0.192`) | **1.7** |
| 5 | `MLP` (`128, 64` Baseline) | 0.6658 | 0.6525 | 0.3770 (`1.47x`) | 0.2221 | 0.1834 | 0.0573 | 0.4475 (`0.195`) | 7.5 |
| 6 | `DecisionTree` (Baseline) | 0.6389 | 0.6270 | 0.3501 (`1.37x`) | 0.1905 | 0.1861 | 0.0516 | 0.4306 (`0.195`) | 3.4 |

---

## 3. Repository Structure

```text
.
├── .gitignore                      # Excludes .venv, caches, raw/split CSVs, .npz archives, LaTeX build files
├── btl.md                          # Course project requirements
├── README.md                       # Project summary, results & reproduction instructions
├── requirements.txt                # Pinned Python 3.12 dependencies
├── configs/
│   ├── base.yaml                   # Seed=42, paths, column schema, split & preprocessing config
│   └── models.yaml                 # Baseline parameters, Optuna search spaces & cost ratios (6 models)
├── data/
│   ├── raw/                        # data_dictionary.csv (tracked); train.csv & test.csv downloaded via script
│   └── interim/splits/             # split_metadata.json (tracked); oot_split.csv & random_split.csv generated via script
├── notebooks/
│   ├── 02_eda.ipynb                # Dev-only Exploratory Data Analysis notebook
│   └── 03_preprocessing.ipynb      # Feature engineering & anti-leakage verification notebook
├── src/
│   ├── data/                       # download_dataset.py, audit_dataset.py, verify_audit.py, load_data.py, splits.py
│   ├── features/                   # engineering.py, preprocessing.py
│   ├── models/                     # registry.py, train_benchmark.py, tune_models.py, imbalance_calibration.py, optimize_models.py
│   ├── evaluation/                 # metrics.py, tracker.py, threshold_analysis.py, final_test_evaluation.py
│   ├── explainability/             # analyze_explainability.py (TreeSHAP & feature importance)
│   └── utils/                      # build_notebooks.py, execute_notebook.py, verify_literature.py
├── scripts/
│   └── run_benchmark_pipeline.py   # CLI orchestrator for baseline, tuning, imbalance & plots
├── experiments/
│   ├── figures/                    # Diagnostic, threshold & SHAP figures
│   └── results/                    # Consolidated Dev benchmark & optimization CSV/JSON tables
├── reports/
│   ├── final_report_en.pdf         # Compiled HCMUT LaTeX project report (PDF)
│   ├── latex/                      # Modular XeLaTeX source (main.tex, sections/*.tex, figures/)
│   ├── audit/                      # dataset_metadata.json, audit_summary.json, verification_report.json
│   ├── figures/                    # Publication figures included in the report
│   └── tables/                     # Consolidated audit, EDA, and final OOT evaluation tables
└── tests/                          # Automated unit & anti-leakage pytest suite
```

---

## 4. Setup & Reproducibility

### 4.1. Environment Setup & Data Download
The project uses Python `3.12` with dependencies pinned in [`requirements.txt`](requirements.txt):
```powershell
# 1. Create and activate Python virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Download raw Kaggle dataset and generate deterministic splits
# (Required before running tests or training; raw CSVs are gitignored)
python -m src.data.download_dataset
python -m src.data.splits
```

### 4.2. Quick Verification
```powershell
# Run automated unit & anti-leakage tests
python -m pytest tests/ -v

# Verify raw dataset SHA-256 integrity and audit outputs
python -m src.data.verify_audit
```

### 4.3. Full Pipeline Execution Order
```powershell
# Stage 1: Raw Data Audit & Splits
python -m src.data.audit_dataset
python -m src.data.verify_audit
python -m src.data.splits

# Stage 2: EDA & Preprocessing Notebooks
python -m src.utils.build_notebooks
python -m src.utils.execute_notebook notebooks/02_eda.ipynb notebooks/03_preprocessing.ipynb

# Stage 3: Baseline Benchmark, Feature Ablation, Tuning & Imbalance (Dev CV)
python scripts/run_benchmark_pipeline.py --tracks all

# Stage 4: Stage-2 GBDT Optimization, Threshold Analysis & TreeSHAP (Dev)
python -m src.models.optimize_models
python -m src.evaluation.threshold_analysis
python -m src.explainability.analyze_explainability

# Stage 5: Final Evaluation on Locked OOT & Random Test Sets
python -m src.evaluation.final_test_evaluation

# Stage 6: Compile LaTeX Report (XeLaTeX)
Push-Location reports/latex
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
Pop-Location
Copy-Item reports/latex/main.pdf reports/final_report_en.pdf -Force
```
