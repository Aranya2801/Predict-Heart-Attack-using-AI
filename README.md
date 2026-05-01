# 🫀 Predict Heart Attack using AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-Data-green?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
</p>

<p align="center">
  <b>A machine learning project that predicts the likelihood of a heart attack using Logistic Regression — built with Python, scikit-learn, and real-world clinical features.</b>
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Features Used](#-features-used)
- [How It Works](#-how-it-works)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Installation](#-installation)
- [Usage](#-usage)
- [Feature Engineering](#-feature-engineering)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

---

## 🔍 Overview

Cardiovascular disease is one of the leading causes of death worldwide. Early prediction of heart attack risk can dramatically improve patient outcomes. This project applies **Logistic Regression** — a classic, interpretable machine learning algorithm — to classify whether a patient is at risk of a heart attack based on a rich set of clinical and lifestyle features.

Key highlights:
- Engineered **31 features** including derived biomarkers, categorical bins, and interaction variables
- Uses **Logistic Regression** from scikit-learn for binary classification
- Performs **EDA + Feature Engineering** before model training
- Dataset sourced from **Kaggle**

---

## 🎬 Demo

```bash
$ python main.py
Accuracy: 0.87
```

> The model outputs its accuracy on the held-out test set. Future versions will include a full classification report, confusion matrix, and ROC-AUC score.

---

## 📊 Dataset

The dataset (`heart_attack_dataset.csv`) was obtained from [Kaggle](https://www.kaggle.com/) and contains patient records with both clinical measurements and lifestyle indicators.

| Property | Details |
|---|---|
| Source | Kaggle |
| Format | CSV |
| Target Column | `heart_attack` (0 = No, 1 = Yes) |
| Split | 70% Train / 30% Test |

### Raw Features (sample)

| Column | Description |
|---|---|
| `age` | Patient age in years |
| `gender` | Biological sex |
| `chest_pain_type` | Type of chest pain experienced |
| `rest_bp` | Resting blood pressure (mmHg) |
| `cholesterol` | Serum cholesterol (mg/dL) |
| `fasting_bs` | Fasting blood sugar level |
| `rest_ecg` | Resting electrocardiographic results |
| `max_hr` | Maximum heart rate achieved |
| `exercise_induced_angina` | Angina induced by exercise (Y/N) |
| `st_depression` | ST depression induced by exercise |
| `num_major_vessels` | Number of major vessels colored by fluoroscopy |
| `thalassemia_type` | Thalassemia classification |
| `smoking` | Smoking status |
| `bmi` | Body Mass Index |
| `diabetes` | Diabetes status |
| `family_history` | Family history of heart disease |
| `alcohol_consumption` | Alcohol intake level |
| `crp` | C-Reactive Protein (inflammation marker) |
| `homocysteine` | Homocysteine level |
| `sbp_variability` | Systolic blood pressure variability |
| `dbp_variability` | Diastolic blood pressure variability |
| `depression` | Depression status |
| `waist_circumference` | Waist circumference (cm) |
| `physical_activity_moderate` | Hours/week of moderate activity |
| `physical_activity_vigorous` | Hours/week of vigorous activity |
| `sleep_quality` | Sleep quality score |
| `statins` | Statin medication use |
| `beta_blockers` | Beta-blocker medication use |
| `vitamin_d` | Vitamin D level |
| `magnesium` | Magnesium level |

---

## 🧬 Features Used

The model trains on **31 features** — a mix of raw inputs and engineered variables:

```python
features = [
    'age', 'gender', 'chest_pain_type',
    'rest_bp_cat', 'cholesterol_cat',
    'fasting_bs', 'rest_ecg', 'max_hr_cat',
    'exercise_induced_angina', 'st_depression',
    'num_major_vessels', 'thalassemia_type',
    'smoking', 'bmi', 'diabetes',
    'exercise_angina', 'family_history',
    'alcohol_consumption_cat', 'crp', 'homocysteine',
    'sbp_variability', 'dbp_variability', 'depression',
    'age_squared', 'fasting_bs_variability',
    'waist_circumference_cat', 'physical_activity_hours',
    'poor_sleep_quality', 'medication_use',
    'vitamin_d_cat', 'magnesium_cat'
]
```

---

## ⚙️ How It Works

```
Raw CSV Data
     │
     ▼
Feature Engineering
 ┌──────────────────────────────────┐
 │  • Bin blood pressure → category │
 │  • Bin cholesterol → category    │
 │  • Compute age²                  │
 │  • Compute total activity hours  │
 │  • Derive sleep quality flag     │
 │  • Derive medication_use flag    │
 │  • Bin alcohol, vitamin D, Mg    │
 └──────────────────────────────────┘
     │
     ▼
Train / Test Split (70 / 30)
     │
     ▼
Logistic Regression (scikit-learn)
     │
     ▼
Accuracy Score on Test Set
```

### Why Logistic Regression?

Logistic Regression is ideal here because:
- The output is **binary** (heart attack: yes/no)
- It is **interpretable** — coefficients show feature importance
- It is **fast** to train and validate
- It performs well on clinical datasets with proper feature engineering

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression |
| Test Size | 30% |
| Random State | 42 |
| Accuracy | Printed to console at runtime |

> **Note:** Full evaluation metrics (precision, recall, F1, ROC-AUC, confusion matrix) are planned for a future release.

---

## 📁 Project Structure

```
Predict-Heart-Attack-using-AI/
│
├── main.py                      # Main script — EDA, training, evaluation
├── heart_attack_dataset.csv     # Primary dataset (from Kaggle)
├── o2Saturation.csv             # Supplementary oxygen saturation data
└── README.md                    # You are here!
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

---

## 🛠 Installation

**1. Clone the repository**

```bash
git clone https://github.com/Aranya2801/Predict-Heart-Attack-using-AI.git
cd Predict-Heart-Attack-using-AI
```

**2. (Optional but recommended) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

**3. Install required packages**

```bash
pip install pandas numpy scikit-learn
```

---

## ▶️ Usage

Run the prediction script:

```bash
python main.py
```

**What happens when you run it:**

1. Loads `heart_attack_dataset.csv` into a pandas DataFrame
2. Engineers new features (categorical bins, interaction terms, derived flags)
3. Splits data into 70% training / 30% testing sets
4. Trains a Logistic Regression model on the training set
5. Predicts on the test set
6. Prints the accuracy score to the console

---

## 🔬 Feature Engineering

This project goes beyond raw features to engineer meaningful clinical signals:

| Engineered Feature | Description |
|---|---|
| `age_squared` | Captures non-linear age risk |
| `rest_bp_cat` | Bins resting BP: normal / prehypertension / hypertension |
| `max_hr_cat` | Bins max heart rate: low / moderate / high |
| `cholesterol_cat` | Bins cholesterol: normal / high / very high |
| `alcohol_consumption_cat` | Bins alcohol: none / moderate / excessive |
| `fasting_bs_variability` | Std dev of fasting blood sugar per patient |
| `waist_circumference_cat` | Bins waist: normal / high / very high |
| `physical_activity_hours` | Total moderate + vigorous weekly activity hours |
| `poor_sleep_quality` | Boolean flag: sleep quality score > 5 |
| `medication_use` | Boolean flag: statins OR beta-blockers in use |
| `vitamin_d_cat` | Bins vitamin D: deficient / insufficient / sufficient |
| `magnesium_cat` | Bins magnesium: deficient / normal / excess |

---

## 🔭 Future Work

This project has a strong foundation with many exciting improvements planned:

- [ ] **Evaluate more algorithms** — Random Forest, XGBoost, SVM, Neural Networks
- [ ] **Full evaluation report** — Confusion matrix, ROC-AUC, precision-recall curve
- [ ] **Hyperparameter tuning** — GridSearchCV / Bayesian optimization
- [ ] **Feature importance analysis** — SHAP values for interpretability
- [ ] **Cross-validation** — Stratified K-Fold to reduce variance
- [ ] **Handle class imbalance** — SMOTE or class weights
- [ ] **Web app deployment** — Flask or Streamlit interface
- [ ] **Mobile integration** — REST API backend for mobile apps
- [ ] **Incorporate o2Saturation.csv** — Enrich features with O2 saturation data
- [ ] **Unit tests** — Add pytest coverage for preprocessing and model steps

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repo
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please make sure your code is clean, commented, and follows the existing style.

---

## 👩‍💻 Author

**Aranya Ghosh**

- GitHub: [@Aranya2801](https://github.com/Aranya2801)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a ⭐ on GitHub — it helps others find it!

---

<p align="center">Made with ❤️ and Python by Aranya Ghosh</p>
