#  Predict Students Dropout and Academic Success (Machine Learning Pipeline)

An end-to-end multi-class machine learning benchmarking and early-warning pipeline to predict higher education student academic outcomes: **Dropout**, **Enrolled**, or **Graduate**.

The pipeline evaluates and compares models across two operational decision stages:
1. **Enrollment-Only Stage**: Uses intake demographics, socio-economic factors, and prior qualifications available at admission for early risk alerts.
2. **Full Stage (with semester progress)**: Incorporates 1st and 2nd semester curricular unit grades, evaluations, and approval counts to assess in-program academic trajectory.

---

##  Dataset Overview

* **Source**: UCI Machine Learning Repository ("Predict Students' Dropout and Academic Success")
* **Samples**: 4,424 student records
* **Features**: 36 academic, demographic, and macroeconomic indicators
* **Target Classes**:
  * `Dropout` — Discontinued studies before completion
  * `Enrolled` — Actively continuing academic program
  * `Graduate` — Successfully completed degree requirements

---

##  Machine Learning Models Evaluated

All models incorporate class balancing (`balanced` / `balanced_subsample` / weighted loss) to address class distribution skew:

* **Logistic Regression** (L2 Regularized Baseline)
* **Decision Tree** (Constrained Depth)
* **Random Forest** (400 Estimators)
* **HistGradientBoosting** (Gradient Boosted Decision Trees)

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/predict-student-dropout-and-success-ml.git
cd predict-student-dropout-and-success-ml

### 2. Install Dependencies
python -m pip install -r requirements.txt

### 3. Run the Experiment
python run_experiment.py