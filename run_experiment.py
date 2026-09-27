"""Student Dropout & Academic Success Prediction ML Pipeline

=========================================================
Trains, benchmarks, and interprets multi-class ML models on higher education
student retention outcomes: Dropout, Enrolled, and Graduate.

Features:
- Auto-fetches benchmark dataset from UCI if husnain.csv is not present locally.
- Two-stage evaluation: 'Enrollment-only' vs 'Full (with semester outcomes)'.
- Handles class imbalance with balanced sample and class weights.
- Generates 6 analytical visualization artifacts (PNGs) and metric CSV/JSON reports.
"""

from __future__ import annotations

import io
import json
import os
import ssl
import sys
import urllib.request
import warnings
import zipfile
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend suitable for CLI, VS Code, and servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Suppress benign scikit-learn convergence and feature name warnings during production runs
warnings.filterwarnings("ignore")

# 1. GLOBAL CONSTANTS & CONFIGURATION

RANDOM_STATE: int = 42  # Seed for strict reproducibility across data splits & models
CLASSES: List[str] = ["Dropout", "Enrolled", "Graduate"]

# Palette: Dropout (Alert Crimson), Enrolled (Muted Slate), Graduate (Forest Green)
COLOR_DROPOUT: str = "#C1121F"
COLOR_ENROLLED: str = "#9A8C98"
COLOR_GRADUATE: str = "#3A7D44"
COLOR_STAGE_ENROLL: str = "#3A7D44"
COLOR_STAGE_FULL: str = "#2E4B7A"

# High-resolution styling for exported publication/presentation charts
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
})


# 2. DATA LOADING & SCHEMA SEGMENTATION

def load_and_preprocess_dataset(
    filepath: str = "husnain.csv",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str], List[str], List[str]]:
    """
    Loads student academic dataset (UCI Student Dropout & Academic Success format),
    cleans column whitespace, segregates numeric vs categorical features, and
    partitions features into 'Enrollment-only' and 'Full' feature sets.

    Auto-heals missing dataset by fetching directly from the official UCI repository.
    """
    if not os.path.exists(filepath):
        alt_names = ["data.csv", "student_data.csv", "students.csv"]
        found = next((alt for alt in alt_names if os.path.exists(alt)), None)
        if found:
            filepath = found
            print(f"ℹ Using existing local dataset: '{filepath}'")
        else:
            print(f"⚠️  Dataset '{filepath}' not found locally.")
            print("🌐 Fetching 'Predict Students Dropout & Academic Success' benchmark from UCI repository...")
            
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            except Exception:
                ctx = None

            try:
                uci_url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
                req = urllib.request.Request(uci_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                    z = zipfile.ZipFile(io.BytesIO(resp.read()))
                    csv_bytes = z.read("data.csv")
                    with open(filepath, "wb") as out_f:
                        out_f.write(csv_bytes)
                    # Also create alias copy as husnain.csv if needed
                    if filepath != "husnain.csv" and not os.path.exists("husnain.csv"):
                        with open("husnain.csv", "wb") as f_alias:
                            f_alias.write(csv_bytes)
                print(f"✅ Successfully downloaded '{filepath}' ({len(csv_bytes) // 1024} KB, 4,424 students)!\n")
            except Exception as dl_err:
                raise FileNotFoundError(
                    f"Dataset '{filepath}' not found and automatic download failed ({dl_err}).\n"
                    f"Please place 'husnain.csv' in this folder or download from:\n"
                    f"https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
                )

    # Detect delimiter safely (UCI dataset uses semicolon ';')
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
        delimiter = ";" if ";" in first_line else ","

    df = pd.read_csv(filepath, sep=delimiter)

    # Strip any leading/trailing whitespace from column headers
    df.columns = [c.strip() for c in df.columns]

    if "Target" not in df.columns:
        raise KeyError(
            "Expected 'Target' column indicating academic outcome (Dropout/Enrolled/Graduate)."
        )

    # Filter out semester curricular units (1st & 2nd semester academic progress)
    sem_cols = [c for c in df.columns if c.startswith("Curricular units")]

    # Define numeric features (grades, ages, socioeconomic indices, and semester stats)
    numeric_cols = [
        "Previous qualification (grade)",
        "Admission grade",
        "Age at enrollment",
        "Application order",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ] + sem_cols

    # Ensure numeric columns actually exist in the provided dataset
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # All remaining non-target columns are categorical (e.g. Course, Marital status, Nationality)
    categorical_cols = [
        c for c in df.columns if c not in numeric_cols and c != "Target"
    ]

    y = df["Target"]
    X_full = df.drop(columns=["Target"])

    # 'Enrollment-only' features exclude post-enrollment curricular unit outcomes
    enroll_cols = [c for c in X_full.columns if c not in sem_cols]

    return df, y, X_full, numeric_cols, categorical_cols, enroll_cols

# 3. PIPELINE PREPROCESSOR & MODEL FACTORY

def create_preprocessor(
    cols: List[str], numeric_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """Constructs a ColumnTransformer that standardizes continuous variables
    and one-hot encodes categorical variables."""
    active_num = [c for c in cols if c in numeric_cols]
    active_cat = [c for c in cols if c in categorical_cols]

    transformers = []
    if active_num:
        transformers.append(("num", StandardScaler(), active_num))

    if active_cat:
        # Compatible across both older (<1.2) and modern (>=1.2) scikit-learn versions
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, active_cat))

    return ColumnTransformer(transformers)


def get_model_candidates() -> Dict[str, object]:
    """Suite of 4 multi-class classifiers with class balancing tuned for retention data."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.06,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }


# 4. FEATURE IMPORTANCE AGGREGATION

def aggregate_rf_feature_importance(
    cols: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.Series:
    """Fits Random Forest and aggregates dummy one-hot importances to parent features."""
    pre = create_preprocessor(cols, numeric_cols, categorical_cols)
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("clf", rf)])
    pipe.fit(X_train[cols], y_train)

    active_num = [c for c in cols if c in numeric_cols]
    active_cat = [c for c in cols if c in categorical_cols]

    feat_names: List[str] = list(active_num)
    if active_cat:
        ohe = pipe.named_steps["pre"].named_transformers_["cat"]
        feat_names.extend(list(ohe.get_feature_names_out(active_cat)))

    importances = pipe.named_steps["clf"].feature_importances_
    sorted_cats = sorted(active_cat, key=len, reverse=True)

    agg: Dict[str, float] = {}
    for fn, val in zip(feat_names, importances):
        if fn in active_num:
            base_col = fn
        else:
            base_col = next((c for c in sorted_cats if fn.startswith(c)), fn)
        agg[base_col] = agg.get(base_col, 0.0) + float(val)

    return pd.Series(agg).sort_values(ascending=False)


# 5. VISUALIZATION ARTIFACT GENERATOR

class ArtifactPlotter:
    """Encapsulates chart generation and saves standardized figure artifacts."""

    def __init__(self):
        self.saved_figures: List[str] = []

    def _save(self, fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        self.saved_figures.append(filename)

    def plot_class_distribution(
        self, y: pd.Series, filename: str = "fig_classdist.png"
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        counts = y.value_counts().reindex(CLASSES)
        colors = [COLOR_DROPOUT, COLOR_ENROLLED, COLOR_GRADUATE]

        bars = ax.bar(CLASSES, counts.values, color=colors, width=0.55, edgecolor="none")
        for bar, count in zip(bars, counts.values):
            pct = (count / len(y)) * 100.0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                count + (max(counts) * 0.02),
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="medium",
            )

        ax.set_ylabel("Number of Students", fontsize=10)
        ax.set_title(f"Target Outcome Distribution (Total N = {len(y):,})", fontsize=12, pad=12)
        ax.set_ylim(0, max(counts) * 1.22)
        self._save(fig, filename)

    def plot_eda_boxplots(
        self, df: pd.DataFrame, filename: str = "fig_eda.png"
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
        box_colors = [COLOR_DROPOUT, COLOR_ENROLLED, COLOR_GRADUATE]

        configs = [
            (axes[0], "Admission grade", "Admission Grade by Outcome"),
            (axes[1], "Curricular units 1st sem (approved)", "1st-Sem Approved Units by Outcome"),
        ]

        for ax, col, title in configs:
            if col not in df.columns:
                continue
            data = [df[df["Target"] == c][col].dropna().values for c in CLASSES]
            bp = ax.boxplot(data, labels=CLASSES, patch_artist=True, showfliers=False, widths=0.45)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor("#333333")
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.4)

            ax.set_title(title, fontsize=11, pad=10)
            ax.set_ylabel(col, fontsize=9.5)
            ax.tick_params(axis="x", labelsize=9.5)

        self._save(fig, filename)

    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        stages: List[str],
        filename: str = "fig_modelcompare.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        model_names = results_df["model"].unique().tolist()
        x = np.arange(len(model_names))
        width = 0.36
        palette = [COLOR_STAGE_ENROLL, COLOR_STAGE_FULL]

        for i, stage in enumerate(stages):
            stage_rows = results_df[results_df["stage"] == stage]
            f1_values = [
                (
                    stage_rows[stage_rows["model"] == m]["macro_f1"].iloc[0]
                    if not stage_rows[stage_rows["model"] == m].empty
                    else 0.0
                )
                for m in model_names
            ]
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset,
                f1_values,
                width,
                label=stage,
                color=palette[i % len(palette)],
                alpha=0.88,
            )
            for bar, val in zip(bars, f1_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val + 0.012,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="semibold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9.5)
        ax.set_ylabel("Macro-F1 (Holdout Test Set)", fontsize=10)
        ax.set_title("Predictive Performance: Enrollment-Only vs. Full Stage", fontsize=12, pad=12)
        ax.set_ylim(0, 0.90)
        ax.legend(frameon=True, fontsize=9)
        self._save(fig, filename)

    def plot_confusion_matrix(
        self, cm_matrix: List[List[int]], title: str, filename: str
    ) -> None:
        cm = np.array(cm_matrix)
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, fontsize=9)
        ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel("Predicted Outcome", fontsize=10, labelpad=8)
        ax.set_ylabel("Actual Outcome", fontsize=10, labelpad=8)
        ax.set_title(title, fontsize=11, pad=10)

        max_val = cm.max() if cm.max() > 0 else 1
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                count = cm[i, j]
                color = "white" if count > (max_val * 0.5) else "black"
                ax.text(
                    j,
                    i,
                    f"{count:,}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10.5,
                    fontweight="medium",
                )

        self._save(fig, filename)

    def plot_feature_importance(
        self,
        importance_series: pd.Series,
        title: str,
        color: str,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        top_12 = importance_series.head(12)[::-1]

        bars = ax.barh(top_12.index, top_12.values, color=color, alpha=0.88, height=0.62)
        for bar, val in zip(bars, top_12.values):
            ax.text(
                val + 0.003,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.3f}",
                ha="left",
                va="center",
                fontsize=8.5,
                color="#222222",
            )

        ax.set_title(title, fontsize=11.5, pad=10)
        ax.set_xlabel("Aggregated Relative Importance (Gini)", fontsize=10)
        ax.set_xlim(0, max(top_12.values) * 1.15)
        ax.tick_params(axis="y", labelsize=9)
        self._save(fig, filename)

# 6. MAIN ORCHESTRATION PIPELINE

def run_pipeline(csv_path: str = "husnain.csv") -> None:
    print("=" * 72)
    print("🚀 Higher Education Student Retention & Dropout ML Experiment Pipeline")
    print("=" * 72)

    # 1. Load and clean data
    print(f"\n[1/6] Loading and validating dataset: '{csv_path}'...")
    df, y, X_full, numeric_cols, categorical_cols, enroll_cols = load_and_preprocess_dataset(csv_path)
    print(f"      Total records : {len(df):,}")
    print(f"      Total features: {X_full.shape[1]} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    print(f"      Target breakdown: {dict(y.value_counts())}")

    # 2. Train-Test Stratified Split (80% train / 20% holdout test)
    print("\n[2/6] Performing 80/20 stratified holdout split (random_state=42)...")
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      Training samples: {len(X_train_full):,} | Test samples: {len(X_test_full):,}")

    # 3. Model Training & Evaluation Across Stages
    stages: Dict[str, List[str]] = {
        "Enrollment-only": enroll_cols,
        "Full (with semester results)": list(X_full.columns),
    }

    results: List[Dict[str, object]] = []
    reports: Dict[str, dict] = {}
    cms: Dict[str, List[List[int]]] = {}

    print("\n[3/6] Training and evaluating models across feature stages:")
    for stage_name, cols in stages.items():
        print(f"\n  --- Stage: {stage_name} ({len(cols)} active features) ---")
        X_train_stage = X_train_full[cols]
        X_test_stage = X_test_full[cols]

        for model_name, clf in get_model_candidates().items():
            preprocessor = create_preprocessor(cols, numeric_cols, categorical_cols)
            pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

            if model_name == "Gradient Boosting":
                sample_weights = compute_sample_weight("balanced", y_train)
                pipe.fit(X_train_stage, y_train, clf__sample_weight=sample_weights)
            else:
                pipe.fit(X_train_stage, y_train)

            y_pred = pipe.predict(X_test_stage)
            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")

            results.append({
                "stage": stage_name,
                "model": model_name,
                "accuracy": round(float(acc), 3),
                "macro_f1": round(float(macro_f1), 3),
            })

            key = f"{stage_name} | {model_name}"
            reports[key] = classification_report(
                y_test, y_pred, labels=CLASSES, output_dict=True, zero_division=0
            )
            cms[key] = confusion_matrix(y_test, y_pred, labels=CLASSES).tolist()

            print(f"      {model_name:24s} | Accuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f}")

    # 4. Save Metric Tables
    print("\n[4/6] Exporting metric summaries and detailed evaluation logs...")
    res_df = pd.DataFrame(results)
    res_df.to_csv("results_table.csv", index=False)

    with open("results_detail.json", "w", encoding="utf-8") as f:
        json.dump({"reports": reports, "cms": cms}, f, indent=2)

    best_df = res_df.loc[res_df.groupby("stage")["macro_f1"].idxmax()]
    print("\n🏆 Champion Model per Feature Stage:")
    for _, row in best_df.iterrows():
        print(f"      * {row['stage']}: {row['model']} (Macro-F1: {row['macro_f1']:.3f}, Acc: {row['accuracy']:.3f})")

    # 5. Feature Importance Aggregation (Random Forest)
    print("\n[5/6] Computing aggregated feature importances via Random Forest...")
    imp_full = aggregate_rf_feature_importance(
        list(X_full.columns),
        X_train_full,
        y_train,
        numeric_cols,
        categorical_cols,
    )
    imp_enroll = aggregate_rf_feature_importance(
        enroll_cols, X_train_full, y_train, numeric_cols, categorical_cols
    )

    imp_full.to_csv("imp_full.csv")
    imp_enroll.to_csv("imp_enroll.csv")

    # 6. Generate Figures
    print("\n[6/6] Generating analytical visualization artifacts...")
    plotter = ArtifactPlotter()
    plotter.plot_class_distribution(y, "fig_classdist.png")
    plotter.plot_eda_boxplots(df, "fig_eda.png")
    plotter.plot_model_comparison(res_df, list(stages.keys()), "fig_modelcompare.png")

    best_full_model = best_df[best_df["stage"] == "Full (with semester results)"]["model"].iloc[0]
    best_enroll_model = best_df[best_df["stage"] == "Enrollment-only"]["model"].iloc[0]

    plotter.plot_confusion_matrix(
        cms[f"Full (with semester results) | {best_full_model}"],
        f"Confusion Matrix: {best_full_model} (Full Stage)",
        "fig_cm_full.png",
    )
    plotter.plot_confusion_matrix(
        cms[f"Enrollment-only | {best_enroll_model}"],
        f"Confusion Matrix: {best_enroll_model} (Enrollment-only)",
        "fig_cm_enroll.png",
    )
    plotter.plot_feature_importance(
        imp_full,
        "Top Predictors (Full Feature Set with Semesters)",
        COLOR_STAGE_FULL,
        "fig_imp_full.png",
    )
    plotter.plot_feature_importance(
        imp_enroll,
        "Top Predictors at Matriculation (Enrollment-only)",
        COLOR_STAGE_ENROLL,
        "fig_imp_enroll.png",
    )

    meta = {
        "best_full": best_full_model,
        "best_enroll": best_enroll_model,
        "imp_full_top": imp_full.head(12).round(4).to_dict(),
        "imp_enroll_top": imp_enroll.head(12).round(4).to_dict(),
        "generated_figures": plotter.saved_figures,
    }
    with open("meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ All artifacts successfully exported:")
    for fig_path in plotter.saved_figures:
        print(f"      📊 {fig_path}")
    print("      📄 results_table.csv")
    print("      📄 results_detail.json")
    print("      📄 imp_full.csv & imp_enroll.csv")
    print("      📄 meta.json")
    print("=" * 72)


if __name__ == "__main__":
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "husnain.csv"
    try:
        run_pipeline(dataset_file)
    except FileNotFoundError as e:
        print(f"\n[Notice] {e}")