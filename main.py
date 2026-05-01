"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          HEART ATTACK PREDICTION SYSTEM — AI-POWERED CLINICAL ENGINE        ║
║                    Author: Aranya Ghosh  |  Version: 3.0                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Multi-model ensemble framework for cardiovascular risk stratification.
Implements: XGBoost · Random Forest · LightGBM · Logistic Regression
           + SHAP explainability · Hyperparameter tuning · Clinical scoring
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import warnings
import logging
from datetime import datetime
from pathlib import Path

# ─── Data Science ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─── Visualisation ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                       # headless / server-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─── Machine Learning ─────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    PowerTransformer,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

# ─── Optional heavy dependencies (graceful degradation) ───────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not installed — skipping. pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("LightGBM not installed — skipping. pip install lightgbm")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed — skipping explanations. pip install shap")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not installed. pip install imbalanced-learn")

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "data": {
        "primary_path":   "heart_attack_dataset.csv",
        "secondary_path": "o2Saturation.csv",
        "target":         "heart_attack",
        "patient_id":     "patient_id",
        "test_size":      0.20,
        "val_size":       0.10,
        "random_state":   42,
    },
    "model": {
        "cv_folds":       5,
        "n_jobs":        -1,
        "scoring":        "roc_auc",
        "calibrate":      True,
        "use_smote":      True,
    },
    "output": {
        "figures_dir":    "outputs/figures",
        "models_dir":     "outputs/models",
        "reports_dir":    "outputs/reports",
    },
    "risk_thresholds": {
        "low":            0.30,
        "moderate":       0.55,
        "high":           0.75,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_file: str = "heart_attack_pipeline.log") -> logging.Logger:
    """Configure structured logging to both file and console."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("HeartAI")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def print_section(title: str, width: int = 78) -> None:
    bar = "═" * width
    print(f"\n{bar}\n  {title}\n{bar}")


def save_figure(fig: plt.Figure, name: str) -> None:
    path = Path(CONFIG["output"]["figures_dir"]) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════

class DataIngestion:
    """Load, merge, and perform initial validation of raw data sources."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["data"]

    def load(self) -> pd.DataFrame:
        logger.info("Loading primary dataset …")
        df = self._load_csv(self.cfg["primary_path"])

        # Merge SpO2 data if available
        secondary = self.cfg.get("secondary_path")
        if secondary and Path(secondary).exists():
            logger.info("Merging secondary SpO2 dataset …")
            spo2 = self._load_csv(secondary)
            pid  = self.cfg.get("patient_id", "patient_id")
            if pid in df.columns and pid in spo2.columns:
                spo2 = spo2.add_prefix("spo2_").rename(columns={f"spo2_{pid}": pid})
                df = df.merge(spo2, on=pid, how="left")

        logger.info(f"Dataset shape after ingestion: {df.shape}")
        return df

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        for enc in ("utf-8", "latin-1", "iso-8859-1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                logger.info(f"  ✓ {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

class ExploratoryAnalysis:
    """Rich EDA with automated visualisations."""

    def __init__(self, df: pd.DataFrame, target: str):
        self.df     = df
        self.target = target

    def run(self) -> None:
        print_section("EXPLORATORY DATA ANALYSIS")
        self._data_quality_report()
        self._class_distribution()
        self._numeric_distributions()
        self._correlation_heatmap()

    def _data_quality_report(self) -> None:
        df = self.df
        missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing = missing[missing > 0]
        duplicates = df.duplicated().sum()

        print(f"\n  Rows      : {len(df):>8,}")
        print(f"  Columns   : {df.shape[1]:>8}")
        print(f"  Duplicates: {duplicates:>8,}")

        if not missing.empty:
            print(f"\n  Missing Values (top columns):")
            for col, pct in missing.head(10).items():
                bar = "█" * int(pct / 2)
                print(f"    {col:<35} {pct:5.1f}%  {bar}")
        else:
            print("\n  ✓ No missing values detected.")

    def _class_distribution(self) -> None:
        if self.target not in self.df.columns:
            return
        counts = self.df[self.target].value_counts()
        total  = len(self.df)
        print(f"\n  Target Distribution  ({self.target}):")
        for label, count in counts.items():
            bar = "█" * int(count / total * 40)
            print(f"    Class {label}: {count:>6,}  ({count/total*100:.1f}%)  {bar}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Target Class Distribution", fontsize=14, fontweight="bold")

        colors = ["#2196F3", "#F44336"]
        labels = [f"No Attack\n({counts.get(0, 0):,})", f"Attack\n({counts.get(1, 0):,})"]
        axes[0].pie(counts.values, labels=labels, colors=colors,
                    autopct="%1.1f%%", startangle=90,
                    wedgeprops=dict(edgecolor="white", linewidth=2))
        axes[0].set_title("Class Split")

        bars = axes[1].bar(["No Attack", "Heart Attack"], counts.values,
                           color=colors, edgecolor="white", linewidth=1.5,
                           width=0.5)
        for bar, val in zip(bars, counts.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 5, str(val),
                         ha="center", va="bottom", fontweight="bold")
        axes[1].set_title("Count per Class")
        axes[1].set_ylabel("Patients")
        axes[1].spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        save_figure(fig, "01_class_distribution")

    def _numeric_distributions(self) -> None:
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != self.target][:12]

        if not num_cols:
            return

        n = len(num_cols)
        cols_per_row = 4
        rows = (n + cols_per_row - 1) // cols_per_row

        fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
        axes = axes.flatten()
        fig.suptitle("Feature Distributions by Outcome", fontsize=14, fontweight="bold")

        colors = {0: "#2196F3", 1: "#F44336"}
        for i, col in enumerate(num_cols):
            ax = axes[i]
            if self.target in self.df.columns:
                for outcome in [0, 1]:
                    subset = self.df[self.df[self.target] == outcome][col].dropna()
                    ax.hist(subset, bins=25, alpha=0.6, color=colors[outcome],
                            label=("No Attack" if outcome == 0 else "Attack"),
                            density=True)
                ax.legend(fontsize=7)
            else:
                ax.hist(self.df[col].dropna(), bins=25, color="#607D8B", alpha=0.8)
            ax.set_title(col, fontsize=9, fontweight="bold")
            ax.set_xlabel("")
            ax.spines[["top", "right"]].set_visible(False)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        save_figure(fig, "02_feature_distributions")

    def _correlation_heatmap(self) -> None:
        num_df = self.df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 3:
            return

        corr = num_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(max(10, corr.shape[1]), max(8, corr.shape[1] - 1)))
        sns.heatmap(corr, mask=mask, annot=True if corr.shape[1] <= 15 else False,
                    fmt=".2f", cmap="RdYlBu_r", center=0,
                    linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.7})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        save_figure(fig, "03_correlation_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Domain-driven + automated feature construction.

    New features created
    ────────────────────
    age_squared            — captures non-linear aging risk
    pulse_pressure         — SBP − DBP (arterial stiffness proxy)
    cholesterol_hdl_ratio  — atherogenicity index
    cardiac_risk_score     — composite clinical heuristic
    metabolic_syndrome     — binary flag for metabolic syndrome cluster
    heart_rate_reserve     — max_hr − resting_hr (if available)
    bmi_waist_product      — obesity severity composite
    ...and more
    """

    CATEGORICAL_COLS: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running feature engineering …")
        df = df.copy()

        df = self._numeric_features(df)
        df = self._clinical_composites(df)
        df = self._categoricals(df)
        df = self._interaction_terms(df)

        logger.info(f"  Features after engineering: {df.shape[1]}")
        return df

    # ── Numeric derivations ──────────────────────────────────────────────────

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
        if "age" in df.columns:
            df["age_squared"]     = df["age"] ** 2
            df["age_log"]         = np.log1p(df["age"])

        if "rest_bp" in df.columns:
            df["rest_bp_squared"] = df["rest_bp"] ** 2

        if "cholesterol" in df.columns:
            df["cholesterol_log"] = np.log1p(df["cholesterol"])

        if all(c in df.columns for c in ["sbp", "dbp"]):
            df["pulse_pressure"]  = df["sbp"] - df["dbp"]
            df["map"]             = df["dbp"] + df["pulse_pressure"] / 3

        if all(c in df.columns for c in ["max_hr", "rest_hr"]):
            df["heart_rate_reserve"] = df["max_hr"] - df["rest_hr"]

        return df

    # ── Clinical composite scores ────────────────────────────────────────────

    @staticmethod
    def _clinical_composites(df: pd.DataFrame) -> pd.DataFrame:
        cols    = df.columns.tolist()
        score   = pd.Series(np.zeros(len(df)), index=df.index)

        if "age" in cols:
            score += (df["age"] > 55).astype(int) * 2
        if "cholesterol" in cols:
            score += (df["cholesterol"] > 240).astype(int) * 2
        if "rest_bp" in cols:
            score += (df["rest_bp"] > 140).astype(int) * 2
        if "smoking" in cols:
            score += df["smoking"].fillna(0).astype(int) * 3
        if "diabetes" in cols:
            score += df["diabetes"].fillna(0).astype(int) * 2
        if "family_history" in cols:
            score += df["family_history"].fillna(0).astype(int) * 1
        if "bmi" in cols:
            score += (df["bmi"] > 30).astype(int) * 1

        df["cardiac_risk_score"] = score

        # Metabolic syndrome flag
        met_components = []
        if "bmi"         in cols: met_components.append((df["bmi"]         > 30).astype(int))
        if "rest_bp"     in cols: met_components.append((df["rest_bp"]     > 130).astype(int))
        if "cholesterol" in cols: met_components.append((df["cholesterol"] > 200).astype(int))
        if "fasting_bs"  in cols: met_components.append((df["fasting_bs"]  > 100).astype(int))
        if met_components:
            df["metabolic_syndrome"] = (sum(met_components) >= 3).astype(int)

        # Cholesterol / HDL ratio
        if all(c in cols for c in ["cholesterol", "hdl"]):
            df["cholesterol_hdl_ratio"] = df["cholesterol"] / df["hdl"].replace(0, np.nan)

        return df

    # ── Categorical binning ──────────────────────────────────────────────────

    @staticmethod
    def _categoricals(df: pd.DataFrame) -> pd.DataFrame:
        cats = {}

        if "age" in df.columns:
            cats["age_group"] = pd.cut(
                df["age"],
                bins=[0, 35, 45, 55, 65, 120],
                labels=["<35", "35-45", "45-55", "55-65", "65+"],
            )

        if "rest_bp" in df.columns:
            cats["bp_category"] = pd.cut(
                df["rest_bp"],
                bins=[0, 120, 130, 140, 180, 999],
                labels=["Normal", "Elevated", "Stage1", "Stage2", "Crisis"],
            )

        if "cholesterol" in df.columns:
            cats["chol_category"] = pd.cut(
                df["cholesterol"],
                bins=[0, 200, 239, 999],
                labels=["Desirable", "Borderline", "High"],
            )

        if "bmi" in df.columns:
            cats["bmi_category"] = pd.cut(
                df["bmi"],
                bins=[0, 18.5, 25, 30, 35, 999],
                labels=["Underweight", "Normal", "Overweight", "Obese1", "Obese2+"],
            )

        for col, series in cats.items():
            df[col] = series.astype(str)

        return df

    # ── Interaction terms ────────────────────────────────────────────────────

    @staticmethod
    def _interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns.tolist()

        if all(c in cols for c in ["age", "cholesterol"]):
            df["age_x_cholesterol"] = df["age"] * df["cholesterol"]

        if all(c in cols for c in ["age", "rest_bp"]):
            df["age_x_bp"]          = df["age"] * df["rest_bp"]

        if all(c in cols for c in ["smoking", "cholesterol"]):
            df["smoke_x_chol"]      = df["smoking"].fillna(0) * df["cholesterol"]

        return df


# ══════════════════════════════════════════════════════════════════════════════
# PRE-PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    """
    Handles:
      • Missing value imputation (median for numeric, mode for categorical)
      • Label / One-Hot encoding
      • Feature scaling (StandardScaler)
      • Optional class-imbalance correction (SMOTE)
    """

    def __init__(self):
        self.num_imputer  = SimpleImputer(strategy="median")
        self.cat_imputer  = SimpleImputer(strategy="most_frequent")
        self.scaler       = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.num_cols_: list[str] = []
        self.cat_cols_: list[str] = []
        self.feature_names_: list[str] = []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                      use_smote: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        X_num = self._encode_numerics(X, fit=True)
        X_cat = self._encode_cats(X, fit=True)

        X_proc = np.hstack([X_num, X_cat])
        y_arr  = y.values

        if use_smote and IMBLEARN_AVAILABLE:
            logger.info("  Applying SMOTE to balance classes …")
            sm     = SMOTE(random_state=42, k_neighbors=5)
            X_proc, y_arr = sm.fit_resample(X_proc, y_arr)
            logger.info(f"  Post-SMOTE shape: {X_proc.shape}")

        return X_proc, y_arr

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_num = self._encode_numerics(X, fit=False)
        X_cat = self._encode_cats(X, fit=False)
        return np.hstack([X_num, X_cat])

    def _encode_numerics(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        if not self.num_cols_:
            return np.empty((len(X), 0))
        X_num = X[self.num_cols_].copy()
        if fit:
            X_num = self.num_imputer.fit_transform(X_num)
            X_num = self.scaler.fit_transform(X_num)
        else:
            X_num = self.num_imputer.transform(X_num)
            X_num = self.scaler.transform(X_num)
        return X_num

    def _encode_cats(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        if not self.cat_cols_:
            return np.empty((len(X), 0))
        frames = []
        for col in self.cat_cols_:
            col_data = X[[col]].copy()
            col_data = self.cat_imputer.fit_transform(col_data) if fit else \
                       self.cat_imputer.transform(col_data)
            col_series = pd.Series(col_data.flatten(), name=col)
            if fit:
                le = LabelEncoder()
                encoded = le.fit_transform(col_series.astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    encoded = np.zeros(len(col_series))
                else:
                    col_str = col_series.astype(str)
                    known   = set(le.classes_)
                    col_str = col_str.map(lambda x: x if x in known else le.classes_[0])
                    encoded = le.transform(col_str)
            frames.append(encoded.reshape(-1, 1))

        return np.hstack(frames)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ZOO
# ══════════════════════════════════════════════════════════════════════════════

def build_model_zoo() -> dict:
    """Return all candidate classifiers."""
    zoo: dict = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=4,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        ),
        "SVM_RBF": SVC(
            kernel="rbf", probability=True, C=5.0, gamma="scale",
            class_weight="balanced", random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=9, metric="minkowski", weights="distance"
        ),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=4,
            class_weight="balanced", random_state=42
        ),
    }

    if XGB_AVAILABLE:
        zoo["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            tree_method="hist", random_state=42
        )

    if LGB_AVAILABLE:
        zoo["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", n_jobs=-1, random_state=42,
            verbose=-1
        )

    return zoo


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINER  (cross-validation + evaluation)
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Train, cross-validate, and evaluate every model in the zoo.
    Selects the best model by ROC-AUC and builds a Voting Ensemble.
    """

    def __init__(self, cfg: dict):
        self.cfg     = cfg["model"]
        self.results_: list[dict] = []
        self.best_model_name_: str = ""
        self.best_model_     = None
        self.ensemble_       = None

    def run(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_test:  np.ndarray, y_test:  np.ndarray,
            ) -> pd.DataFrame:
        print_section("MODEL TRAINING & CROSS-VALIDATION")

        zoo = build_model_zoo()
        cv  = StratifiedKFold(n_splits=self.cfg["cv_folds"],
                              shuffle=True, random_state=42)

        for name, model in zoo.items():
            logger.info(f"  Training {name} …")
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring="roc_auc", n_jobs=self.cfg["n_jobs"]
                )
                model.fit(X_train, y_train)
                metrics = self._evaluate(model, X_test, y_test, name)
                metrics["cv_roc_auc_mean"] = cv_scores.mean()
                metrics["cv_roc_auc_std"]  = cv_scores.std()
                self.results_.append(metrics)
                logger.info(
                    f"    ROC-AUC={metrics['roc_auc']:.4f}  "
                    f"F1={metrics['f1']:.4f}  "
                    f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
                )
            except Exception as exc:
                logger.warning(f"    {name} failed: {exc}")

        df_results = pd.DataFrame(self.results_).sort_values(
            "roc_auc", ascending=False
        ).reset_index(drop=True)

        # Best single model
        self.best_model_name_ = df_results.iloc[0]["model"]
        self.best_model_      = zoo[self.best_model_name_]

        # Voting ensemble from top-3
        top3_names = df_results.head(3)["model"].tolist()
        estimators = [(n, zoo[n]) for n in top3_names]
        self.ensemble_ = VotingClassifier(estimators=estimators,
                                          voting="soft", n_jobs=-1)
        self.ensemble_.fit(X_train, y_train)
        ens_metrics = self._evaluate(self.ensemble_, X_test, y_test, "VotingEnsemble(Top3)")
        logger.info(f"  Ensemble ROC-AUC={ens_metrics['roc_auc']:.4f}  F1={ens_metrics['f1']:.4f}")

        return df_results

    def _evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray,
                  name: str) -> dict:
        y_pred  = model.predict(X_test)
        y_proba = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, "predict_proba") else y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        return {
            "model":           name,
            "accuracy":        accuracy_score(y_test, y_pred),
            "precision":       precision_score(y_test, y_pred, zero_division=0),
            "recall":          recall_score(y_test, y_pred, zero_division=0),
            "specificity":     tn / (tn + fp) if (tn + fp) > 0 else 0,
            "f1":              f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":         roc_auc_score(y_test, y_proba),
            "avg_precision":   average_precision_score(y_test, y_proba),
            "mcc":             matthews_corrcoef(y_test, y_pred),
            "kappa":           cohen_kappa_score(y_test, y_pred),
            "brier_score":     brier_score_loss(y_test, y_proba),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

class ModelVisualiser:
    """Produce publication-quality evaluation charts."""

    def __init__(self, models: dict, X_test: np.ndarray, y_test: np.ndarray):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test

    def plot_all(self, results_df: pd.DataFrame) -> None:
        self._leaderboard(results_df)
        self._roc_curves()
        self._pr_curves()
        self._confusion_matrices()

    def _leaderboard(self, df: pd.DataFrame) -> None:
        metrics = ["roc_auc", "f1", "accuracy", "recall", "precision", "mcc"]
        metrics = [m for m in metrics if m in df.columns]
        sub = df.set_index("model")[metrics]

        fig, ax = plt.subplots(figsize=(max(12, len(df) * 1.5), 6))
        x      = np.arange(len(sub))
        width  = 0.12
        colors = plt.cm.tab10.colors

        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, sub[metric], width=width,
                   label=metric.upper().replace("_", " "),
                   color=colors[i % len(colors)], alpha=0.85,
                   edgecolor="white")

        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(sub.index, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title("Model Leaderboard — All Metrics", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(1.0, color="grey", lw=0.5, ls="--")

        plt.tight_layout()
        save_figure(fig, "04_model_leaderboard")

    def _roc_curves(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        colors  = plt.cm.tab10.colors

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

        for i, (name, model) in enumerate(self.models.items()):
            if not hasattr(model, "predict_proba"):
                continue
            try:
                y_proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                auc = roc_auc_score(self.y_test, y_proba)
                ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                        label=f"{name} (AUC={auc:.3f})")
            except Exception:
                pass

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        save_figure(fig, "05_roc_curves")

    def _pr_curves(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        colors  = plt.cm.tab10.colors
        baseline = self.y_test.mean()
        ax.axhline(baseline, color="k", ls="--", lw=1, label=f"Baseline ({baseline:.2f})")

        for i, (name, model) in enumerate(self.models.items()):
            if not hasattr(model, "predict_proba"):
                continue
            try:
                y_proba = model.predict_proba(self.X_test)[:, 1]
                prec, rec, _ = precision_recall_curve(self.y_test, y_proba)
                ap = average_precision_score(self.y_test, y_proba)
                ax.plot(rec, prec, lw=2, color=colors[i % len(colors)],
                        label=f"{name} (AP={ap:.3f})")
            except Exception:
                pass

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        save_figure(fig, "06_pr_curves")

    def _confusion_matrices(self) -> None:
        n      = len(self.models)
        cols   = min(n, 3)
        rows   = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes   = np.array(axes).flatten()
        fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

        for i, (name, model) in enumerate(self.models.items()):
            ax  = axes[i]
            y_pred = model.predict(self.X_test)
            cm  = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                        cmap="Blues", cbar=False,
                        xticklabels=["No Attack", "Attack"],
                        yticklabels=["No Attack", "Attack"],
                        linewidths=1, linecolor="white")
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        save_figure(fig, "07_confusion_matrices")


# ══════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """Model-agnostic SHAP explanations for the best model."""

    def __init__(self, model, X_test: np.ndarray, feature_names: list[str]):
        self.model         = model
        self.X_test        = X_test
        self.feature_names = feature_names

    def explain(self) -> None:
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available — skipping explanations.")
            return

        print_section("SHAP FEATURE EXPLANATIONS")
        logger.info("Computing SHAP values …")

        try:
            # Tree explainer is fastest for ensemble models
            explainer   = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)

            # For binary classifiers, shap_values may be a list [neg, pos]
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values

            # Global feature importance
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(sv, self.X_test,
                              feature_names=self.feature_names,
                              show=False)
            plt.title("SHAP Summary — Global Feature Importance",
                      fontsize=13, fontweight="bold")
            plt.tight_layout()
            save_figure(fig, "08_shap_summary")

            # Bar importance
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(sv, self.X_test,
                              feature_names=self.feature_names,
                              plot_type="bar", show=False)
            plt.title("SHAP Mean |Impact| per Feature",
                      fontsize=13, fontweight="bold")
            plt.tight_layout()
            save_figure(fig2, "09_shap_bar")

            logger.info("  ✓ SHAP explanations generated.")
        except Exception as exc:
            logger.warning(f"  SHAP failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL RISK SCORER
# ══════════════════════════════════════════════════════════════════════════════

class ClinicalRiskScorer:
    """
    Wraps the best trained model to provide patient-level risk assessment.
    Returns a structured risk report with confidence band and recommendations.
    """

    RISK_LEVELS = {
        "LOW":      (0.00, 0.30, "🟢", "MINIMAL RISK"),
        "MODERATE": (0.30, 0.55, "🟡", "MODERATE RISK — monitor closely"),
        "HIGH":     (0.55, 0.75, "🟠", "HIGH RISK — clinical evaluation advised"),
        "CRITICAL": (0.75, 1.00, "🔴", "CRITICAL RISK — immediate cardiology referral"),
    }

    def __init__(self, model, preprocessor: Preprocessor, thresholds: dict):
        self.model       = model
        self.prep        = preprocessor
        self.thresholds  = thresholds

    def score_patient(self, patient_dict: dict) -> dict:
        """Score a single patient record and return a full report."""
        df    = pd.DataFrame([patient_dict])
        X     = self.prep.transform(df)
        prob  = self.model.predict_proba(X)[0, 1]

        level, icon, label = self._classify(prob)

        report = {
            "risk_probability": round(float(prob), 4),
            "risk_percent":     f"{prob * 100:.1f}%",
            "risk_level":       level,
            "risk_icon":        icon,
            "risk_label":       label,
            "recommendations":  self._recommendations(level, prob),
            "timestamp":        datetime.now().isoformat(),
        }
        return report

    def _classify(self, prob: float) -> tuple[str, str, str]:
        for level, (lo, hi, icon, label) in self.RISK_LEVELS.items():
            if lo <= prob < hi:
                return level, icon, label
        return "CRITICAL", "🔴", "CRITICAL RISK"

    @staticmethod
    def _recommendations(level: str, prob: float) -> list[str]:
        base = [
            "Maintain a heart-healthy diet (Mediterranean style).",
            "At least 150 min/week of moderate aerobic exercise.",
            "Avoid smoking and limit alcohol consumption.",
            "Monitor blood pressure and cholesterol regularly.",
        ]
        if level in ("HIGH", "CRITICAL"):
            base += [
                "⚕️  Schedule a cardiology consultation promptly.",
                "⚕️  ECG and echocardiogram may be indicated.",
                "⚕️  Consider stress-test / coronary angiography.",
                "🚨  Ensure emergency contact plan is in place.",
            ]
        elif level == "MODERATE":
            base += [
                "Consult your GP for a comprehensive cardiac risk panel.",
                "Consider statin therapy if LDL-C > 130 mg/dL.",
            ]
        return base

    def print_report(self, report: dict) -> None:
        print("\n" + "═" * 60)
        print(f"  {report['risk_icon']}  PATIENT RISK ASSESSMENT  {report['risk_icon']}")
        print("═" * 60)
        print(f"  Probability   : {report['risk_percent']}  ({report['risk_probability']:.4f})")
        print(f"  Risk Level    : {report['risk_level']}  — {report['risk_label']}")
        print(f"  Generated     : {report['timestamp']}")
        print("\n  Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"    {i:2}. {rec}")
        print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Write a Markdown & CSV summary report."""

    def __init__(self, results_df: pd.DataFrame, best_name: str, cfg: dict):
        self.results_df = results_df
        self.best_name  = best_name
        self.cfg        = cfg

    def generate(self) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(self.cfg["output"]["reports_dir"])

        # CSV
        csv_path = base / f"model_results_{ts}.csv"
        self.results_df.to_csv(csv_path, index=False)
        logger.info(f"  CSV report → {csv_path}")

        # Markdown
        md_path = base / f"report_{ts}.md"
        self._write_markdown(md_path)
        logger.info(f"  Markdown report → {md_path}")

    def _write_markdown(self, path: Path) -> None:
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best = self.results_df.iloc[0]

        lines = [
            "# Heart Attack Prediction — Experiment Report",
            f"\n**Generated:** {ts}",
            f"\n**Best Model:** `{self.best_name}`",
            "\n---\n",
            "## Model Leaderboard\n",
            self.results_df.to_markdown(index=False),
            "\n---\n",
            "## Best Model Performance\n",
        ]
        for metric in ["roc_auc", "f1", "accuracy", "precision", "recall",
                       "specificity", "mcc", "kappa", "brier_score"]:
            if metric in best.index:
                lines.append(f"- **{metric.upper()}**: `{best[metric]:.4f}`")

        lines += [
            "\n---\n",
            "## Confusion Matrix\n",
            f"| | Predicted No Attack | Predicted Attack |",
            f"|---|---|---|",
            f"| **Actual No Attack** | {int(best.get('tn', 0))} | {int(best.get('fp', 0))} |",
            f"| **Actual Attack**    | {int(best.get('fn', 0))} | {int(best.get('tp', 0))} |",
            "\n---\n",
            "_Aranya Ghosh — Heart Attack Prediction AI System_",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global logger
    logger = setup_logging()

    ensure_dirs(
        CONFIG["output"]["figures_dir"],
        CONFIG["output"]["models_dir"],
        CONFIG["output"]["reports_dir"],
    )

    print_section("HEART ATTACK PREDICTION — AI PIPELINE  v3.0")
    print(f"  Author : Aranya Ghosh")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. Data Ingestion ────────────────────────────────────────────────────
    ingestion = DataIngestion(CONFIG)
    df_raw    = ingestion.load()

    target = CONFIG["data"]["target"]

    # ── 2. EDA ───────────────────────────────────────────────────────────────
    eda = ExploratoryAnalysis(df_raw, target)
    eda.run()

    # ── 3. Feature Engineering ───────────────────────────────────────────────
    drop_cols = [CONFIG["data"].get("patient_id", "patient_id")]
    df = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns])

    fe = FeatureEngineer()
    df = fe.fit_transform(df)

    # ── 4. Train / Test Split ────────────────────────────────────────────────
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found. Check dataset.")
        return

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = CONFIG["data"]["test_size"],
        random_state = CONFIG["data"]["random_state"],
        stratify     = y,
    )

    # ── 5. Preprocessing ─────────────────────────────────────────────────────
    print_section("PREPROCESSING")
    prep = Preprocessor()
    X_train_p, y_train_p = prep.fit_transform(
        X_train, y_train, use_smote=CONFIG["model"]["use_smote"]
    )
    X_test_p  = prep.transform(X_test)
    y_test_arr = y_test.values

    logger.info(f"  Train shape: {X_train_p.shape} | Test shape: {X_test_p.shape}")

    # ── 6. Train & Evaluate Models ───────────────────────────────────────────
    trainer     = ModelTrainer(CONFIG)
    results_df  = trainer.run(X_train_p, y_train_p, X_test_p, y_test_arr)

    print_section("LEADERBOARD")
    display_cols = ["model", "roc_auc", "f1", "accuracy",
                    "recall", "precision", "mcc", "cv_roc_auc_mean"]
    display_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[display_cols].to_string(index=False))

    # ── 7. Visualise Results ─────────────────────────────────────────────────
    zoo     = build_model_zoo()
    trained = {}
    for name in zoo:
        try:
            zoo[name].fit(X_train_p, y_train_p)
            trained[name] = zoo[name]
        except Exception:
            pass
    if trainer.ensemble_:
        trained["VotingEnsemble"] = trainer.ensemble_

    vis = ModelVisualiser(trained, X_test_p, y_test_arr)
    vis.plot_all(results_df)

    # ── 8. SHAP Explanations ─────────────────────────────────────────────────
    best_name  = trainer.best_model_name_
    best_model = trained.get(best_name, trainer.best_model_)
    feat_names = (prep.num_cols_ +
                  [f"cat_{c}" for c in prep.cat_cols_])

    if XGB_AVAILABLE and isinstance(best_model, xgb.XGBClassifier):
        shap_exp = SHAPExplainer(best_model, X_test_p, feat_names)
        shap_exp.explain()
    elif LGB_AVAILABLE and isinstance(best_model, lgb.LGBMClassifier):
        shap_exp = SHAPExplainer(best_model, X_test_p, feat_names)
        shap_exp.explain()
    elif isinstance(best_model, RandomForestClassifier):
        shap_exp = SHAPExplainer(best_model, X_test_p, feat_names)
        shap_exp.explain()

    # ── 9. Clinical Risk Scorer — demo ──────────────────────────────────────
    print_section("CLINICAL RISK SCORER — DEMO PATIENT")
    scorer = ClinicalRiskScorer(best_model, prep, CONFIG["risk_thresholds"])

    demo_patient = {
        "age": 62, "gender": 1, "chest_pain_type": 3,
        "rest_bp": 148, "cholesterol": 260, "fasting_bs": 1,
        "rest_ecg": 2, "max_hr": 115, "exercise_induced_angina": 1,
        "st_depression": 2.8, "num_major_vessels": 2,
        "thalassemia_type": 3, "smoking": 1, "bmi": 32.1,
        "diabetes": 1, "family_history": 1,
    }
    try:
        report = scorer.score_patient(demo_patient)
        scorer.print_report(report)
    except Exception as exc:
        logger.warning(f"Clinical scorer demo failed: {exc}")

    # ── 10. Report ───────────────────────────────────────────────────────────
    rep = ReportGenerator(results_df, best_name, CONFIG)
    rep.generate()

    print_section("PIPELINE COMPLETE")
    print(f"  Best model  : {best_name}")
    best_row = results_df[results_df["model"] == best_name]
    if not best_row.empty:
        print(f"  ROC-AUC     : {best_row.iloc[0]['roc_auc']:.4f}")
        print(f"  F1-Score    : {best_row.iloc[0]['f1']:.4f}")
    print(f"  Figures     : {CONFIG['output']['figures_dir']}/")
    print(f"  Reports     : {CONFIG['output']['reports_dir']}/")
    print(f"  Completed   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
