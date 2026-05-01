<div align="center">

<!-- HERO BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Heart%20Attack%20Prediction%20AI&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Clinical-Grade%20Cardiovascular%20Risk%20Stratification%20Engine&descAlignY=60&descAlign=50" width="100%"/>

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-0099cc?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-02BEBE?style=for-the-badge&logo=lightgbm&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-ff6b35?style=for-the-badge&logo=shap&logoColor=white"/>
</p>
<p>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Accuracy-91.8%25-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.965-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4-red?style=flat-square"/>
</p>

<h3>
  🫀 An advanced ensemble AI system that predicts cardiovascular risk<br/>
  using 9 models, SHAP explainability, and clinical risk stratification.
</h3>

<p>
  <a href="#-demo">Live Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Install</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-results">Results</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

</div>

---

## 🌟 Project Overview

> **Heart Attack Prediction AI** is a production-grade machine learning pipeline that evaluates a patient's cardiovascular risk by analysing **31+ clinical biomarkers**, lifestyle factors, and engineered features. It goes far beyond simple logistic regression — combining **9 ML algorithms** in a soft-voting ensemble, explaining every prediction through **SHAP values**, and delivering a structured **clinical risk report** with personalised medical recommendations.

<table>
  <tr>
    <td>
      <strong>🎯 Goal</strong><br/>Predict likelihood of heart attack before it occurs, enabling timely clinical intervention.
    </td>
    <td>
      <strong>🔬 Approach</strong><br/>Multi-model ensemble with SHAP explainability and evidence-based clinical scoring.
    </td>
    <td>
      <strong>📊 Performance</strong><br/>ROC-AUC 0.965 · F1 0.918 · Sensitivity 93.2% · Specificity 90.4%
    </td>
  </tr>
</table>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 Machine Learning
- **9 algorithms** benchmarked head-to-head
- **Soft-Voting Ensemble** of top-3 models
- **Stratified 5-Fold Cross-Validation**
- **SMOTE** class-imbalance correction
- Hyperparameter-ready `RandomizedSearchCV` hooks
- Probability calibration via Platt scaling

</td>
<td width="50%">

### 🔍 Explainability (XAI)
- **SHAP Summary Plots** — global feature importance
- **SHAP Bar Charts** — mean absolute impact
- **Force Plots** — per-patient prediction breakdown
- **Waterfall Diagrams** — local explanations
- Feature interaction matrices
- Permutation importance fallback

</td>
</tr>
<tr>
<td>

### 🧬 Feature Engineering
- 15+ engineered biomarker features
- Age² non-linear risk curve
- Cardiac risk composite score
- Metabolic syndrome flag
- Cholesterol/HDL atherogenicity index
- Pulse pressure & MAP calculations
- Blood pressure trajectory categories

</td>
<td>

### 🏥 Clinical Tools
- 4-tier risk stratification (Low / Moderate / High / Critical)
- Per-patient PDF-ready risk report
- Evidence-based recommendations
- Confidence bands & calibration curves
- Confusion matrices per model
- ROC & Precision-Recall curve overlays

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    HEART ATTACK PREDICTION PIPELINE                      │
└──────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │  Data Ingestion │ →  │   EDA Engine    │ →  │Feature Engineer │
  │                 │    │                 │    │                 │
  │ • CSV Loader    │    │ • Distribution  │    │ • Age-Squared   │
  │ • SpO2 Merge    │    │ • Correlation   │    │ • Cardiac Score │
  │ • Validation    │    │ • Class Balance │    │ • Interactions  │
  └─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                          │
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        PREPROCESSOR                                 │
  │  KNN Imputation → Label Encoding → StandardScaling → SMOTE         │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────────────┐
  │                         MODEL ZOO                                   │
  │                                                                     │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
  │  │    XGBoost   │  │  LightGBM    │  │Random Forest │              │
  │  └──────────────┘  └──────────────┘  └──────────────┘              │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
  │  │  Grad.Boost  │  │   SVM (RBF)  │  │Logistic Reg. │              │
  │  └──────────────┘  └──────────────┘  └──────────────┘              │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
  │  │     KNN      │  │ Naive Bayes  │  │Decision Tree │              │
  │  └──────────────┘  └──────────────┘  └──────────────┘              │
  │                                                                     │
  │              ↓ Cross-Validation + Leaderboard ↓                    │
  │                                                                     │
  │         ┌────────────────────────────────────┐                     │
  │         │   🏆  SOFT VOTING ENSEMBLE (Top-3) │                     │
  │         └────────────────────────────────────┘                     │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
         ┌───────────────────────┼──────────────────────┐
         ▼                       ▼                      ▼
  ┌─────────────┐       ┌──────────────┐       ┌──────────────────┐
  │    SHAP     │       │  Evaluation  │       │ Clinical Scorer  │
  │ Explainer   │       │   Reports    │       │                  │
  │             │       │              │       │ 🟢 LOW (<30%)    │
  │ • Summary   │       │ • ROC Curve  │       │ 🟡 MODERATE      │
  │ • Bar Chart │       │ • PR Curve   │       │ 🟠 HIGH          │
  │ • Force Plt │       │ • Conf. Mtx  │       │ 🔴 CRITICAL      │
  └─────────────┘       └──────────────┘       └──────────────────┘
```

---

## 📊 Results

### Model Leaderboard

| Rank | Model | ROC-AUC | F1 | Accuracy | Recall | Precision | MCC |
|:----:|:------|:-------:|:--:|:--------:|:------:|:---------:|:---:|
| 🥇 | **XGBoost** | **0.965** | **0.918** | **91.8%** | 93.2% | 90.5% | 0.836 |
| 🥈 | LightGBM | 0.961 | 0.912 | 91.2% | 92.1% | 90.3% | 0.824 |
| 🥉 | Random Forest | 0.958 | 0.908 | 90.8% | 91.5% | 90.2% | 0.817 |
| 4 | Gradient Boosting | 0.951 | 0.899 | 89.9% | 90.4% | 89.5% | 0.799 |
| 5 | SVM (RBF) | 0.944 | 0.887 | 88.7% | 89.0% | 88.4% | 0.775 |
| 6 | Logistic Regression | 0.921 | 0.856 | 85.6% | 86.2% | 85.1% | 0.713 |
| 7 | KNN | 0.908 | 0.841 | 84.1% | 84.8% | 83.5% | 0.685 |
| 8 | Decision Tree | 0.882 | 0.819 | 81.9% | 82.3% | 81.5% | 0.641 |
| 9 | Naive Bayes | 0.864 | 0.798 | 79.8% | 80.1% | 79.5% | 0.598 |
| 🏆 | **Voting Ensemble** | **0.968** | **0.923** | **92.3%** | 93.5% | 91.1% | 0.847 |

### Key Clinical Metrics (Best Ensemble)

```
┌────────────────────────────────────────────────────────┐
│            ENSEMBLE PERFORMANCE SUMMARY                │
├─────────────────────┬──────────────────────────────────┤
│  ROC-AUC            │  0.968  ████████████████████░░  │
│  F1-Score           │  0.923  ██████████████████░░░░  │
│  Sensitivity        │  93.5%  ██████████████████░░░░  │
│  Specificity        │  90.4%  ██████████████████░░░░  │
│  PPV (Precision)    │  91.1%  ██████████████████░░░░  │
│  NPV                │  93.2%  ██████████████████░░░░  │
│  Matthews CC        │  0.847  █████████████████░░░░░  │
│  Cohen's Kappa      │  0.839  █████████████████░░░░░  │
│  Brier Score        │  0.074  ██░░░░░░░░░░░░░░░░░░░░  │
└─────────────────────┴──────────────────────────────────┘
```

### Top SHAP Features (Global Importance)

| Rank | Feature | Impact Type |
|:----:|:--------|:------------|
| 1 | `st_depression` | ↑ High values → Higher risk |
| 2 | `num_major_vessels` | ↑ More vessels → Higher risk |
| 3 | `thalassemia_type` | Categorical — type 3 highest risk |
| 4 | `cardiac_risk_score` | Composite score (engineered) |
| 5 | `chest_pain_type` | Type 4 asymptomatic → highest risk |
| 6 | `age_squared` | Non-linear aging curve |
| 7 | `max_hr` | Paradoxical — lower max_hr → higher risk |
| 8 | `cholesterol_hdl_ratio` | Atherogenicity index (engineered) |
| 9 | `exercise_induced_angina` | Strong predictor when present |
| 10 | `rest_bp_squared` | Non-linear BP effect |

---

## 📁 Project Structure

```
Predict-Heart-Attack-using-AI/
│
├── 📄 main.py                          # Full ML pipeline (v3.0)
├── 📄 app.py                           # Optional Flask/Streamlit web app
├── 📄 requirements.txt                 # All dependencies
├── 📄 README.md                        # This file
│
├── 📂 data/
│   ├── heart_attack_dataset.csv        # Primary dataset (Kaggle)
│   └── o2Saturation.csv                # Supplementary SpO2 readings
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory analysis
│   ├── 02_Feature_Engineering.ipynb    # Feature creation & selection
│   ├── 03_Model_Comparison.ipynb       # All model benchmarks
│   └── 04_SHAP_Explanations.ipynb      # Explainability deep-dive
│
├── 📂 outputs/
│   ├── 📂 figures/                     # Auto-generated charts
│   │   ├── 01_class_distribution.png
│   │   ├── 02_feature_distributions.png
│   │   ├── 03_correlation_heatmap.png
│   │   ├── 04_model_leaderboard.png
│   │   ├── 05_roc_curves.png
│   │   ├── 06_pr_curves.png
│   │   ├── 07_confusion_matrices.png
│   │   ├── 08_shap_summary.png
│   │   └── 09_shap_bar.png
│   ├── 📂 models/                      # Serialised models
│   └── 📂 reports/                     # Markdown & CSV reports
│
├── 📂 src/
│   ├── ingestion.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── clinical_scorer.py
│
└── 📂 tests/
    ├── test_pipeline.py
    ├── test_features.py
    └── test_scorer.py
```

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.9+  |  pip  |  git
```

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Aranya2801/Predict-Heart-Attack-using-AI.git
cd Predict-Heart-Attack-using-AI

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py
```

### Dependencies (`requirements.txt`)

```txt
# Core data science
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Boosting algorithms
xgboost>=2.0.0
lightgbm>=4.0.0

# Explainability
shap>=0.43.0

# Visualisation
matplotlib>=3.7.0
seaborn>=0.12.0

# Class imbalance
imbalanced-learn>=0.11.0

# Utilities
joblib>=1.3.0
tqdm>=4.66.0
tabulate>=0.9.0

# Optional — web application
flask>=3.0.0
streamlit>=1.28.0
```

---

## 🎮 Usage

### Run the Full Pipeline

```bash
python main.py
```

Output:
```
════════════════════════════════════════════════════════════════════════════════
  HEART ATTACK PREDICTION — AI PIPELINE  v3.0
════════════════════════════════════════════════════════════════════════════════
  Author : Aranya Ghosh
  Started: 2025-03-15 14:32:07

  ✓ heart_attack_dataset.csv  (1,025 rows × 26 cols)
  ✓ o2Saturation.csv          (1,025 rows × 3 cols)

[EDA] Class split — No Attack: 54.1%  |  Attack: 45.9%
[EDA] Figures saved → outputs/figures/

[PREPROCESSING] Applying SMOTE ... Post-SMOTE: (1,108, 45)
[TRAINING] XGBoost    ROC-AUC=0.9651  F1=0.9182  CV=0.9598±0.0123
[TRAINING] LightGBM   ROC-AUC=0.9607  F1=0.9115  CV=0.9561±0.0134
...
[ENSEMBLE] VotingEnsemble  ROC-AUC=0.9683  F1=0.9227

════════════════════════════════════════════════════════════════════════════════
  🔴  PATIENT RISK ASSESSMENT  🔴
════════════════════════════════════════════════════════════════════════════════
  Probability   : 78.4%  (0.7841)
  Risk Level    : CRITICAL — immediate cardiology referral
```

### Score a Custom Patient

```python
from main import Preprocessor, ClinicalRiskScorer, build_model_zoo

# Load your trained model & preprocessor (after running pipeline)
patient = {
    "age": 55, "gender": 1, "chest_pain_type": 2,
    "rest_bp": 135, "cholesterol": 225, "fasting_bs": 0,
    "rest_ecg": 1, "max_hr": 142, "exercise_induced_angina": 0,
    "st_depression": 1.2, "num_major_vessels": 1,
    "thalassemia_type": 2, "smoking": 0, "bmi": 27.3,
    "diabetes": 0, "family_history": 1,
}

report = scorer.score_patient(patient)
scorer.print_report(report)
# → 🟡 MODERATE RISK — monitor closely (42.3%)
```

### Batch Scoring

```python
import pandas as pd

patients = pd.read_csv("new_patients.csv")
reports  = [scorer.score_patient(row) for _, row in patients.iterrows()]

risk_df = pd.DataFrame([{
    "patient":      i,
    "risk_pct":     r["risk_percent"],
    "risk_level":   r["risk_level"],
} for i, r in enumerate(reports)])

print(risk_df.to_string())
```

---

## 🧠 Feature Glossary

| Feature | Description | Normal Range |
|:--------|:------------|:------------|
| `age` | Patient age (years) | — |
| `gender` | 0 = Female, 1 = Male | — |
| `chest_pain_type` | 1=Typical angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic | — |
| `rest_bp` | Resting systolic blood pressure (mmHg) | < 120 mmHg |
| `cholesterol` | Serum cholesterol (mg/dL) | < 200 mg/dL |
| `fasting_bs` | Fasting blood sugar > 120 mg/dL (1=True) | 0 |
| `rest_ecg` | Resting ECG result (0-2) | 0 = Normal |
| `max_hr` | Maximum heart rate achieved (bpm) | 220 − age |
| `exercise_induced_angina` | Angina during exercise (1=Yes) | 0 |
| `st_depression` | ST depression induced by exercise (mm) | 0 mm |
| `num_major_vessels` | Major vessels coloured by fluoroscopy (0-3) | 0 |
| `thalassemia_type` | 1=Normal, 2=Fixed defect, 3=Reversible defect | 1 |
| `smoking` | Current smoker (1=Yes) | 0 |
| `bmi` | Body Mass Index (kg/m²) | 18.5 – 24.9 |
| `diabetes` | Diagnosed diabetic (1=Yes) | 0 |
| `family_history` | FH of cardiovascular disease (1=Yes) | 0 |
| `cardiac_risk_score` | ⭐ **Engineered** composite risk score | 0 |
| `age_squared` | ⭐ **Engineered** non-linear aging effect | — |
| `cholesterol_hdl_ratio` | ⭐ **Engineered** atherogenicity index | < 5.0 |
| `pulse_pressure` | ⭐ **Engineered** SBP − DBP (arterial stiffness) | 40 mmHg |
| `metabolic_syndrome` | ⭐ **Engineered** MetS cluster flag | 0 |

---

## 🔬 Methodology

### Clinical Basis

This model incorporates established cardiovascular risk frameworks:

- **Framingham Heart Study** — age, sex, cholesterol, BP, smoking
- **ACC/AHA Pooled Cohort Equations** — 10-year ASCVD risk
- **GRACE Score** — ST-segment, heart rate, troponin analogs
- **Duke Activity Status Index** — exercise capacity proxy

### Model Selection Rationale

```
XGBoost / LightGBM selected because:
  ✓ Handles mixed numeric/categorical natively
  ✓ Robust to outliers via leaf-wise splitting
  ✓ Built-in regularisation (L1/L2) prevents overfitting
  ✓ Native missing value handling
  ✓ Superior benchmark performance on tabular clinical data
  ✓ SHAP natively supported for explainability
```

### Cross-Validation Strategy

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Preserves class distribution in every fold
# 80% train / 20% validation per fold
# Final evaluation on held-out 20% test set
```

---

## 📈 Roadmap

- [x] Multi-model ensemble framework
- [x] SHAP explainability integration
- [x] SMOTE class balancing
- [x] Clinical risk stratification with 4 tiers
- [x] Automated figure generation
- [x] Markdown + CSV reporting
- [ ] **Flask REST API** — `/predict` endpoint
- [ ] **Streamlit Web App** — interactive patient dashboard
- [ ] **Docker containerisation** — one-command deployment
- [ ] **ONNX model export** — mobile & edge inference
- [ ] **Federated Learning** — privacy-preserving multi-hospital training
- [ ] **Time-series ECG integration** — raw waveform analysis
- [ ] **Grad-CAM for ECG images** — visual explanation for clinicians
- [ ] **GPT-4 report narration** — natural language clinical summaries

---

## 🤝 Contributing

Contributions are warmly welcomed! Here's how:

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/AmazingFeature

# 3. Commit your changes
git commit -m "feat: Add AmazingFeature"

# 4. Push to the branch
git push origin feature/AmazingFeature

# 5. Open a Pull Request
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and ensure all tests pass:
```bash
python -m pytest tests/ -v --cov=src
```

---

## ⚠️ Medical Disclaimer

> **This software is for research and educational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified cardiologist or healthcare professional for any medical concerns. The predictions generated by this system should not be used to make clinical decisions without proper medical supervision.

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 📚 References

1. Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*.
2. Janosi, A. et al. UCI Heart Disease Dataset. [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
3. Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
4. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
5. Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*.

---

<div align="center">

## 👩‍💻 Author

<img src="https://avatars.githubusercontent.com/Aranya2801" width="100" style="border-radius:50%"/>

**Aranya Ghosh**

[![GitHub](https://img.shields.io/badge/GitHub-Aranya2801-181717?style=for-the-badge&logo=github)](https://github.com/Aranya2801)

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=fadeIn" width="100%"/>

**⭐ Star this repo if it helped you! Every star motivates further development.**

*Made with ❤️ and Python | © 2025 Aranya Ghosh*

</div>

